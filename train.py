import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from core import TimeSeriesModel, QuantileLoss, prepare_data_v2, create_sequences, DEVICE
import os
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SolarAI-Trainer")

def train_model(csv_path="solar_data.csv", model_type='Transformer'):
    logger.info(f"Initiating Neural Training - Model: {model_type} | Device: {DEVICE}")
    
    # 1. Pipeline: Load & Sequence
    df, data, target = prepare_data_v2(csv_path)
    
    # Split 80/20 for training validation
    train_size = int(0.8 * len(data))
    X_train_raw, y_train_raw = data[:train_size], target[:train_size]
    X_val_raw, y_val_raw = data[train_size:], target[train_size:]
    
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    
    # Val set needs 24 previous samples for first sequence
    X_val_scaled = scaler_X.transform(data[train_size-24:])
    y_val_scaled = scaler_y.transform(target[train_size-24:])
    
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, 24)
    X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, 24)
    
    # Move to Device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    # 2. Network: Transformer Encoder
    model = TimeSeriesModel(input_size=X_train.shape[2], model_type=model_type, output_size=3)
    model.to(DEVICE)
    
    criterion = QuantileLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 3. Execution: Optimization Loop
    best_val_loss = float('inf')
    early_stop_patience = 12
    patience_counter = 0
    save_path = "solar_transformer_best.pth"
    
    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            outputs = model(bx)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()
            
        avg_train_loss = train_loss / len(train_loader)
        if True: # Modified to log every epoch
            logger.info(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Consistent State Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Neural Convergence at Epoch {epoch+1}. Early Stopping.")
                break
                
    logger.info(f"Training Complete. Global Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    train_model(model_type='Transformer')
