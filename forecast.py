import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from core import TimeSeriesModel, prepare_data_v2, calculate_sun_features, DEVICE
from utils.weather_api import fetch_open_meteo_forecast
import os
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SolarAI-CLI")

def run_forecast_cli():
    logger.info(f"SolarAI Forecasting CLI - Model: Transformer | Device: {DEVICE}")
    
    # 1. Fetch Data
    try:
        logger.info("Connecting to Open-Meteo for real-time physics parameters...")
        f_df = fetch_open_meteo_forecast()
        f_df = calculate_sun_features(f_df)
        f_df["hour"] = f_df["time"].dt.hour
        f_df["rad_roll_3h"] = f_df["shortwave_radiation (W/m²)"].rolling(window=3, min_periods=1).mean()
        f_df["rad_roll_24h"] = f_df["shortwave_radiation (W/m²)"].rolling(window=24, min_periods=1).mean()
        
        # --- Precise 24h Window Calibration (00:00 to 23:00) ---
        midnights = f_df[f_df['time'].dt.hour == 0].index
        if len(midnights) > 0:
            start_idx = midnights[0]
            f_df = f_df.iloc[start_idx : start_idx + 24].reset_index(drop=True)
            logger.info(f"Calibrated 24h window starting at: {f_df['time'].iloc[0]}")
    except Exception as e:
        logger.error(f"Weather API Fetch Failed: {e}")
        return

    # 2. Prepare Inference Environment
    train_cols = [
        "temp", "rad", "cloud", "hum", "sun_elevation", 
        "clearsky_ghi", "rad_roll_3h", "rad_roll_24h", "power_lag_1h", "hour"
    ]
    
    df_train, _, _ = prepare_data_v2("solar_data.csv")
    sx = MinMaxScaler().fit(df_train[train_cols])
    sy = MinMaxScaler().fit(df_train[["target"]])
    
    model = TimeSeriesModel(input_size=len(train_cols), model_type='Transformer', output_size=3)
    model.to(DEVICE)
    model_path = "solar_transformer_best.pth"
    if not os.path.exists(model_path):
         logger.warning("No production weights found. Model is untrained.")
    else:
         model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
    model.eval()
    
    # 3. Recursive Forecasting (P50 Feedback)
    seq_len = 24
    last_p = df_train["target"].iloc[-1]
    
    preds_p10, preds_p50, preds_p90 = [], [], []
    weather_cols = [c for c in train_cols if c != "power_lag_1h"]
    
    # Bootstrap window setup (Standardized logic)
    f_p = f_df.rename(columns={"shortwave_radiation (W/m²)": "rad", "temperature_2m (°C)": "temp", "cloudcover": "cloud", "relative_humidity": "hum"})
    row0 = f_p[weather_cols].iloc[0].copy(); row0["power_lag_1h"] = last_p
    win_scaled = np.tile(sx.transform(pd.DataFrame([row0])[train_cols])[0], (seq_len, 1))
    
    logger.info("Executing recursive neural inference...")
    for i in range(24):
        row = f_p[weather_cols].iloc[i].copy(); row["power_lag_1h"] = last_p
        # Precise window update
        row_s = sx.transform(pd.DataFrame([row])[train_cols])[0]
        win_scaled = np.vstack([win_scaled[1:], row_s])
        
        with torch.no_grad():
            inp = torch.tensor(win_scaled).unsqueeze(0).float().to(DEVICE)
            out = model(inp).cpu().numpy()[0]
        
        p_raw = sy.inverse_transform(out.reshape(-1, 1)).flatten()
        p_raw = np.maximum(p_raw, 0)
        
        # --- Hybrid Night Physics Enforcement ---
        cur_rad = row['rad']
        cur_elev = row['sun_elevation']
        if cur_rad < 10 or cur_elev <= 0:
            p_raw = np.zeros_like(p_raw)
        
        preds_p10.append(p_raw[0]); preds_p50.append(p_raw[1]); preds_p90.append(p_raw[2])
        last_p = p_raw[1]
    
    # 4. Visualization & Reporting
    ft_times = f_df['time'].iloc[:24]
    
    # Clean Terminal Summary Table (High-Contrast IST)
    print("\n" + "█"*60)
    print(f" {'LOCAL TIME (IST)':<20} | {'ELEVATION':<10} | {'FORECAST (kW)'}")
    print(" " + "-"*58)
    for i in range(24):
        t_str = ft_times.iloc[i].strftime("%Y-%m-%d %H:%M")
        elev = f_p['sun_elevation'].iloc[i]
        p50 = preds_p50[i]
        indicator = "🌙" if elev <= 0 else "☀️"
        print(f" {t_str:<20} | {elev:>9.2f}° | {p50:>10.3f} kW {indicator}")
    print("█"*60 + "\n")

    plt.figure(figsize=(12, 6), dpi=120)
    plt.fill_between(ft_times, preds_p10, preds_p90, color='orange', alpha=0.3, label='90% Confidence Interval')
    plt.plot(ft_times, preds_p50, color='blue', linewidth=2.5, label='Transformer Median (P50)')
    plt.title(f"Airtight Probabilistic Solar Forecast - Peak: {max(preds_p50):.2f} kW")
    plt.ylabel("Solar Power (kW)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.savefig("forecast_v2.png", bbox_inches='tight')
    
    logger.info(f"Analysis complete. Plot saved to 'forecast_v2.png'. Peak Output: {max(preds_p50):.2f} kW")

if __name__ == "__main__":
    run_forecast_cli()
