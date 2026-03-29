from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import torch
import os
import logging
from core import TimeSeriesModel, prepare_data_v2, calculate_sun_features, DEVICE
from utils.weather_api import fetch_open_meteo_forecast
from sklearn.preprocessing import MinMaxScaler

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SolarAI-API")

# --- Global Resources ---
RESOURCES = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan Manager: Initializes model and scalers once at startup.
    This replaces the deprecated @app.on_event("startup").
    """
    logger.info(f"System initializing on device: {DEVICE}")
    try:
        csv_path = "solar_data.csv"
        df_train, _, _ = prepare_data_v2(csv_path)
        
        train_cols = [
            "temp", "rad", "cloud", "hum", "sun_elevation", 
            "clearsky_ghi", "rad_roll_3h", "rad_roll_24h", "power_lag_1h", "hour"
        ]
        
        RESOURCES["sx"] = MinMaxScaler().fit(df_train[train_cols])
        RESOURCES["sy"] = MinMaxScaler().fit(df_train[["target"]])
        RESOURCES["cols"] = train_cols
        RESOURCES["bootstrap_power"] = df_train["target"].iloc[-1]
        
        model = TimeSeriesModel(input_size=len(train_cols), model_type='Transformer', output_size=3)
        model_path = "solar_transformer_best.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
            logger.info("Successfully loaded Transformer weights.")
        else:
            logger.warning("No weights found. Serving in UNTRAINED mode.")
            
        model.to(DEVICE).eval()
        RESOURCES["model"] = model
        
    except Exception as e:
        logger.error(f"Critical Startup Error: {e}")
        raise e
        
    yield # Server is running...
    logger.info("System shutting down...")
    RESOURCES.clear()

app = FastAPI(title="SolarAI Forecast API (Lifespan Version)", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "operational", "device": str(DEVICE)}

@app.post("/forecast")
async def get_forecast(lat: float = 9.575390, lon: float = 77.675835):
    """
    Provides a real-time probabilistic forecast for the next 24 hours.
    """
    if not RESOURCES:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        f_df = fetch_open_meteo_forecast(lat, lon)
        f_df = calculate_sun_features(f_df)
        f_df["hour"] = f_df["time"].dt.hour
        f_df["rad_roll_3h"] = f_df["shortwave_radiation (W/m²)"].rolling(window=3, min_periods=1).mean()
        f_df["rad_roll_24h"] = f_df["shortwave_radiation (W/m²)"].rolling(window=24, min_periods=1).mean()
        
        # --- Precise 24h Window Calibration (00:00 to 23:00) ---
        midnights = f_df[f_df['time'].dt.hour == 0].index
        if len(midnights) > 0:
            start_idx = midnights[0]
            f_df = f_df.iloc[start_idx : start_idx + 24].reset_index(drop=True)
            
        last_p = RESOURCES["bootstrap_power"]
        seq_len = 24
        weather_cols = [c for c in RESOURCES["cols"] if c != "power_lag_1h"]
        
        # Initial Window Setup
        results = []
        row0 = f_df[weather_cols].iloc[0].copy(); row0["power_lag_1h"] = last_p
        win = np.tile(RESOURCES["sx"].transform(pd.DataFrame([row0])[RESOURCES["cols"]])[0], (seq_len, 1))
        
        for i in range(24):
            row = f_df[weather_cols].iloc[i].copy(); row["power_lag_1h"] = last_p
            row_s = RESOURCES["sx"].transform(pd.DataFrame([row])[RESOURCES["cols"]])[0]
            win = np.vstack([win[1:], row_s])
            
            with torch.no_grad():
                preds_tensor = torch.tensor(win).unsqueeze(0).float().to(DEVICE)
                out = RESOURCES["model"](preds_tensor).cpu().numpy()[0]
            
            p_raw = RESOURCES["sy"].inverse_transform(out.reshape(-1, 1)).flatten()
            p_raw = np.maximum(p_raw, 0)
            
            # --- Night Physics Enforcement ---
            if f_df["sun_elevation"].iloc[i] <= 0:
                p_raw = np.zeros_like(p_raw)
            
            results.append({
                "time": f_df["time"].iloc[i].isoformat(),
                "p10": float(p_raw[0]), "p50": float(p_raw[1]), "p90": float(p_raw[2])
            })
            last_p = p_raw[1] # Recursive P50 Feedback
            
        return {"forecast": results, "metadata": {"peak_kw": max([r["p50"] for r in results]), "integrated_kwh": sum([r["p50"] for r in results])}}
        
    except Exception as e:
        logger.error(f"Inference Failure: {e}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
