import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core import TimeSeriesModel, QuantileLoss, prepare_data_v2, create_sequences, DEVICE
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SolarAI-Validator")

def walk_forward_validation(model_type='Transformer', n_splits=3, seq_len=24):
    """
    Advanced Walk-Forward Validation for Time-Series.
    Ensures no future data leakage and accounts for sequence windowing.
    """
    logger.info(f"Starting Walk-Forward Neural Validation - Model: {model_type} | Device: {DEVICE}")
    df, data, target = prepare_data_v2("solar_data.csv")
    
    # Recursive Split Logic
    total_len = len(data)
    split_size = total_len // (n_splits + 1)
    
    results = []
    
    for i in range(n_splits):
        train_end = (i + 1) * split_size
        test_end = min((i + 2) * split_size, total_len)
        
        logger.info(f"Neural Split {i+1}/{n_splits} | Train: {train_end} | Test: {test_end}")
        
        X_train_raw, y_train_raw = data[:train_end], target[:train_end]
        X_test_raw, y_test_raw = data[train_end:test_end], target[train_end:test_end]
        
        # Cross-Sequence Preparation (Ensures first test sample has 24 samples of lookback)
        val_lookback = data[train_end - seq_len : test_end]
        val_target_lookback = target[train_end - seq_len : test_end]
        
        scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_raw)
        y_train_scaled = scaler_y.fit_transform(y_train_raw)
        
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_len)
        X_test_seq, y_test_seq = create_sequences(scaler_X.transform(val_lookback), scaler_y.transform(val_target_lookback), seq_len)
        
        # Prepare Tensors
        bt_X_train = torch.tensor(X_train_seq, dtype=torch.float32).to(DEVICE)
        bt_y_train = torch.tensor(y_train_seq, dtype=torch.float32).to(DEVICE)
        bt_X_test = torch.tensor(X_test_seq, dtype=torch.float32).to(DEVICE)
        bt_y_test = torch.tensor(y_test_seq, dtype=torch.float32).to(DEVICE)
        
        loader = DataLoader(TensorDataset(bt_X_train, bt_y_train), batch_size=32, shuffle=True)
        model = TimeSeriesModel(input_size=bt_X_train.shape[2], model_type=model_type, output_size=3)
        model.to(DEVICE)
        
        criterion = QuantileLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Neural Fitting for current split (25 epochs fast train)
        model.train()
        for epoch in range(25):
            for bx, by in loader:
                optimizer.zero_grad()
                criterion(model(bx), by).backward()
                optimizer.step()
        
        # Precise Inference
        model.eval()
        with torch.no_grad():
            preds_scaled = model(bt_X_test).cpu().numpy()
            actual = scaler_y.inverse_transform(bt_y_test.cpu().numpy()).flatten()
            preds = scaler_y.inverse_transform(preds_scaled[:, 1:2]).flatten() # Median (P50)
            
            # --- Hybrid Night Physics Enforcement (Precision Fix) ---
            # Power is forced to 0 if radiation is low OR sun is below horizon
            rad_values = df['rad'].iloc[train_end:test_end].values
            elev_values = df['sun_elevation'].iloc[train_end:test_end].values
            
            # Align lengths
            if len(preds) > len(rad_values):
                preds = preds[-len(rad_values):]
                actual = actual[-len(rad_values):]
            
            # Force zero if dark or night
            preds[(rad_values < 10) | (elev_values <= 0)] = 0.0
            
        mae = mean_absolute_error(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        results.append({'mae': mae, 'rmse': rmse})
        logger.info(f"Split {i+1} Metric: MAE={mae:.5f} | RMSE={rmse:.5f}")
        
    # --- Final Comparative Report (Clean Terminal Output) ---
    print("\n" + "█"*70)
    print(f" {'SAMPLING INDEX':<15} | {'ACTUAL (kW)':<15} | {'PREDICTED (kW)':<15} | {'ERROR'}")
    print(" " + "-"*68)
    
    # Show last 24 records for inspection
    sample_size = min(24, len(actual))
    for j in range(len(actual)-sample_size, len(actual)):
        act_val = actual[j]
        pre_val = preds[j]
        err = abs(act_val - pre_val)
        indicator = "✅" if err < 0.02 else "⚠️"
        print(f" Index {j+train_end:<7} | {act_val:>13.3f} | {pre_val:>13.3f} | {err:>8.3f} {indicator}")
    
    print("█"*70 + "\n")
    
    avg_mae = np.mean([r['mae'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])
    logger.info(f"Final Validation Analytics: Avg MAE: {avg_mae:.5f} | Avg RMSE: {avg_rmse:.5f}")

if __name__ == "__main__":
    walk_forward_validation(model_type='Transformer', n_splits=3)
