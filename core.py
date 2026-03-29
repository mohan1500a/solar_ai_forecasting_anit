import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pvlib.solarposition import get_solarposition
from pvlib.clearsky import simplified_solis
import warnings

# --- Suppress Technical Warnings for Clean Terminal Output ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pvlib")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# --- Global Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.eps = 1e-8

    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i:i+1]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())
        return torch.stack(losses).mean() + self.eps

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=3, output_size=1, model_type='Transformer'):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.LayerNorm(32), nn.Linear(32, output_size)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.out_projection(x[:, -1, :])

def calculate_sun_features(df, lat=9.575390, lon=77.675835):
    """
    Calculates precise Sun Elevation, Azimuth, and Clear Sky GHI.
    Calibrated to Asia/Kolkata (IST) to match the sensor data.
    """
    try:
        # Localize naive timestamps to project timezone (IST) for astronomical alignment
        times = pd.to_datetime(df['time'])
        if times.dt.tz is None:
            times = times.dt.tz_localize('Asia/Kolkata', ambiguous='infer')
            
        solpos = get_solarposition(times, lat, lon)
        
        # Robustly extract values whether Series or Numpy Array
        df['sun_elevation'] = np.array(solpos['elevation'])
        df['sun_azimuth'] = np.array(solpos['azimuth'])
        
        # Physics-informed clear sky baseline
        cs = simplified_solis(np.array(solpos['zenith']), 400)
        df['clearsky_ghi'] = np.array(cs['ghi'])
        df['clearsky_ghi'] = df['clearsky_ghi'].fillna(0)
    except Exception as e:
        import logging
        logging.getLogger("SolarAI-Core").error(f"PVLIB ERROR: {e}")
        df['sun_elevation'] = -10.0 # Strict night default to ensure 0.0 power
        df['sun_azimuth'] = 0.0
        df['clearsky_ghi'] = 0.0
    return df

def create_sequences(data, target, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len + 1):
        X.append(data[i:i + seq_len])
        y.append(target[i + seq_len - 1])
    return np.array(X), np.array(y)

def prepare_data_v2(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1')
    
    # Generic Column Matching to handle encoding variance in symbols
    cols = df.columns
    def find_col(substring):
        for c in cols:
            if substring in str(c): return c
        return None
    
    temp_col = find_col("temperature_2m")
    rad_col = find_col("shortwave_radiation")
    hum_col = find_col("relative_humidity")
    cloud_col = find_col("cloudcover")
    target_col = find_col("Solar_Power")
    
    # Rename to standardized internal names
    if temp_col: df['temp'] = df[temp_col]
    if rad_col: df['rad'] = df[rad_col]
    if hum_col: df['hum'] = df[hum_col]
    if cloud_col: df['cloud'] = df[cloud_col]
    if target_col: df['target'] = df[target_col]
    
    df = calculate_sun_features(df)
    df['power_lag_1h'] = df['target'].shift(1)
    df['rad_roll_3h'] = df['rad'].rolling(window=3, min_periods=1).mean()
    df['rad_roll_24h'] = df['rad'].rolling(window=24, min_periods=1).mean()
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    
    df = df.dropna(subset=['temp', 'rad', 'hum', 'cloud', 'target', 'power_lag_1h']).reset_index(drop=True)
    
    train_cols = [
        "temp", "rad", "cloud", "hum", "sun_elevation", 
        "clearsky_ghi", "rad_roll_3h", "rad_roll_24h", "power_lag_1h", "hour"
    ]
    
    data = df[train_cols].values
    target = df[["target"]].values
    return df, data, target
