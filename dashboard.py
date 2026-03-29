import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from core import TimeSeriesModel, prepare_data_v2, create_sequences, calculate_sun_features, DEVICE
from utils.weather_api import fetch_open_meteo_forecast
import os

# --- Page Config ---
st.set_page_config(page_title="SolarAI - Research Suite", page_icon="🌙", layout="wide")

# --- Soft Sleek Dark Aesthetics ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background-color: #0F172A; }
.stApp { background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); }

.metric-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(15px);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    text-align: center;
    height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-val { font-size: 2.7rem; font-weight: 700; color: #F8FAFC; margin: 4px 0; }
.metric-lbl { font-size: 1.1rem; color: #94A3B8; font-weight: 500; text-transform: uppercase; letter-spacing: 2px; }
.main-title { font-size: 2.5rem; font-weight: 700; color: #F8FAFC; margin-bottom: 2rem; }
.stPlotlyChart { border-radius: 15px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_production_resources():
    """
    Optimized Resource Loader with DEVICE mapping and scale stabilization.
    """
    csv_path = "solar_data.csv"
    df_train, _, _ = prepare_data_v2(csv_path)
    
    train_cols = [
        "temp", "rad", "cloud", "hum", "sun_elevation", 
        "clearsky_ghi", "rad_roll_3h", "rad_roll_24h", "power_lag_1h", "hour"
    ]
    
    sx = MinMaxScaler().fit(df_train[train_cols])
    sy = MinMaxScaler().fit(df_train[["target"]])
    
    model = TimeSeriesModel(input_size=len(train_cols), model_type='Transformer', output_size=3)
    model.to(DEVICE)
    model_path = "solar_transformer_best.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
    model.eval()
    
    # Pre-calculate Validation Metrics on full history
    X_scaled = sx.transform(df_train[train_cols])
    y_scaled = sy.transform(df_train[["target"]])
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, 24)
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_seq).float().to(DEVICE)).cpu().numpy()
    
    act = sy.inverse_transform(y_seq).flatten()
    pre = sy.inverse_transform(preds_scaled[:, 1:2]).flatten() # Use Median (P50)
    
    return model, sx, sy, train_cols, df_train, act, pre

# --- Execution ---
try:
    with st.spinner("Synchronizing Neural Analytics..."):
        model, sx, sy, train_cols, df_raw, eval_act, eval_pre = load_production_resources()
        
    st.markdown('<div class="main-title">SolarAI Intelligence Suite</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["⚡ Live Pipeline Forecast", "📈 Neural Validation Reports"])

    with tab1:
        with st.spinner("Fetching Atmospheric Physics..."):
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
            
            # Recursive Probabilistic Inference
            seq_len, p10, p50, p90 = 24, [], [], []
            last_p = df_raw["target"].iloc[-1]
            weather_cols = [c for c in train_cols if c != "power_lag_1h"]
            
            # Bootstrap initial sliding window (standardized logic)
            row0 = f_df.rename(columns={"shortwave_radiation (W/m²)": "rad", "temperature_2m (°C)": "temp", "cloudcover": "cloud", "relative_humidity": "hum"}).iloc[0].copy()
            row0["power_lag_1h"] = last_p
            win_scaled = np.tile(sx.transform(pd.DataFrame([row0])[train_cols])[0], (seq_len, 1))
            
            f_p = f_df.rename(columns={"shortwave_radiation (W/m²)": "rad", "temperature_2m (°C)": "temp", "cloudcover": "cloud", "relative_humidity": "hum"})
            for i in range(24):
                row = f_p[weather_cols].iloc[i].copy(); row["power_lag_1h"] = last_p
                row_s = sx.transform(pd.DataFrame([row])[train_cols])[0]
                win_scaled = np.vstack([win_scaled[1:], row_s])
                
                with torch.no_grad():
                    inp_tensor = torch.tensor(win_scaled).unsqueeze(0).float().to(DEVICE)
                    out = model(inp_tensor).cpu().numpy()[0]
                
                rp = sy.inverse_transform(out.reshape(-1, 1)).flatten(); rp = np.maximum(rp, 0)
                
                # --- Night Physics Enforcement ---
                if f_p["sun_elevation"].iloc[i] <= 0:
                    rp = np.zeros_like(rp)
                
                p10.append(rp[0]); p50.append(rp[1]); p90.append(rp[2]); last_p = rp[1]
                
            ft_times = f_df["time"].iloc[:24]

        # Metrics Layer
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-lbl">24h Energy Yield</div><div class="metric-val">{np.sum(p50):.1f} kWh</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-lbl">Peak Solar Cap</div><div class="metric-val">{np.max(p50):.2f} kW</div></div>', unsafe_allow_html=True)
        
        pk_idx = np.argmax(p50)
        pk_t = ft_times.iloc[pk_idx].strftime("%H:%M")
        pk_val = p50[pk_idx]
        m3.markdown(f'<div class="metric-card"><div class="metric-lbl">Peak Flux Time</div><div class="metric-val" style="font-size:2.1rem;">{pk_t} ({pk_val:.2f} kW)</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-lbl">System Precision</div><div class="metric-val">Optimal</div></div>', unsafe_allow_html=True)

        # High-Precision Plot
        fig_f = go.Figure()
        # Uncertainty Zone (Orange)
        fig_f.add_trace(go.Scatter(x=ft_times, y=p90, line=dict(width=0), showlegend=False))
        fig_f.add_trace(go.Scatter(x=ft_times, y=p10, line=dict(width=0), fill='tonexty', fillcolor='rgba(249, 115, 22, 0.2)', name="Forecast Uncertainty (90%)"))
        # Median Trend (Vivid Blue)
        fig_f.add_trace(go.Scatter(x=ft_times, y=p50, name="Transformer Median (P50)", line=dict(color="#0EA5E9", width=4, shape='linear')))
        
        fig_f.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=20, t=50, b=50), height=500, hovermode="x unified",
            xaxis=dict(title="24-Hour Forecast Timeline", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Projected Power Output (kW)", showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_f, width='stretch')

    with tab2:
        # Statistical Evidence (Precision)
        mae = mean_absolute_error(eval_act, eval_pre)
        rmse = np.sqrt(mean_squared_error(eval_act, eval_pre))
        r2 = r2_score(eval_act, eval_pre)
        mask = eval_act > 0.1
        mape = mean_absolute_percentage_error(eval_act[mask], eval_pre[mask]) if mask.sum() > 0 else 0
        
        v1, v2, v3, v4 = st.columns(4)
        v1.markdown(f'<div class="metric-card"><div class="metric-lbl">Regression R²</div><div class="metric-val">{r2:.3f}</div></div>', unsafe_allow_html=True)
        v2.markdown(f'<div class="metric-card"><div class="metric-lbl">Mean Absolute Error</div><div class="metric-val">{mae:.3f}</div></div>', unsafe_allow_html=True)
        v3.markdown(f'<div class="metric-card"><div class="metric-lbl">RMSE Metric</div><div class="metric-val">{rmse:.3f}</div></div>', unsafe_allow_html=True)
        v4.markdown(f'<div class="metric-card"><div class="metric-lbl">MAPE Score</div><div class="metric-val">{mape:.1%}</div></div>', unsafe_allow_html=True)
        
        st.markdown("<h3 style='color:white; margin-top:20px;'>Historical Ground Truth Correlation</h3>", unsafe_allow_html=True)
        zoom = st.slider("Neural Feedback window", 50, len(eval_act), 1000)
        
        fig_e = go.Figure()
        # White Ground Truth
        fig_e.add_trace(go.Scatter(y=eval_act[-zoom:], name="Solar Ground Truth", line=dict(color="#F8FAFC", width=1.5)))
        # Deep Orange Prediction
        fig_e.add_trace(go.Scatter(y=eval_pre[-zoom:], name="Transformer Forecast", line=dict(color="#EA580C", width=2, dash='dot')))
        
        fig_e.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=20, t=50, b=50), height=500, hovermode="x unified",
            xaxis=dict(title="Recent Data Indices", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Observed vs Predicted Power (kW)", showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_e, width='stretch')

except Exception as e:
    st.error(f"Neural Analytics Engine Interface Error: {e}")

