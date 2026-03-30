import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from core import TimeSeriesModel, QuantileLoss, prepare_data_v2, create_sequences, DEVICE
import os
import logging
import matplotlib.pyplot as plt

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SolarAI-Trainer")

def train_model(csv_path="solar_data.csv", model_type='Transformer'):
    logger.info(f"Initiating Neural Training - Model: {model_type} | Device: {DEVICE}")
    
    # 1. Pipeline: Load & Sequence
    df, data, target = prepare_data_v2(csv_path)
    
    # Split 70/15/15 for training, validation, testing
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_end = train_size
    val_end = train_size + val_size
    
    X_train_raw, y_train_raw = data[:train_end], target[:train_end]
    X_val_raw, y_val_raw = data[train_end:val_end], target[train_end:val_end]
    # Rest is test
    X_test_raw, y_test_raw = data[val_end:], target[val_end:]
    
    logger.info(f"Neural Splits -> Train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}")
    
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    
    # Val and Test sets need 24 previous samples for sequence continuity
    X_val_scaled = scaler_X.transform(data[train_end-24:val_end])
    y_val_scaled = scaler_y.transform(target[train_end-24:val_end])
    
    X_test_scaled = scaler_X.transform(data[val_end-24:])
    y_test_scaled = scaler_y.transform(target[val_end-24:])
    
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, 24)
    X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, 24)
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, 24)
    
    # Move Tensors to Device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)
    
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
    history_train_loss = []
    history_val_loss = []
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
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(val_loss)
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
    
    # 4. Neural Verification (Test Set)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test).item()
        logger.info(f"Neural Verification (Test Set) | Total Loss: {test_loss:.6f}")

    # Generate Graphs
    logger.info("Generating Neural Performance Graphics...")
    
    # Plot 1: Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(history_train_loss, label='Train Loss', color='blue')
    plt.plot(history_val_loss, label='Validation Loss', color='orange')
    plt.title('Transformer Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Quantile Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Inverse transform predictions for plotting
    try:
        val_preds_inv = scaler_y.inverse_transform(val_outputs[:, 1:2].cpu().numpy()).flatten()
        val_actual_inv = scaler_y.inverse_transform(y_val.cpu().numpy()).flatten()
        
        test_preds_inv = scaler_y.inverse_transform(test_outputs[:, 1:2].cpu().numpy()).flatten()
        test_actual_inv = scaler_y.inverse_transform(y_test.cpu().numpy()).flatten()

        # Plot 2: Validation Predictions (zoomed in to 500 records max for clarity)
        plot_len_val = min(500, len(val_actual_inv))
        plt.figure(figsize=(12, 5))
        plt.plot(val_actual_inv[:plot_len_val], label='Actual Val', color='lightgray', linewidth=2)
        plt.plot(val_preds_inv[:plot_len_val], label='Predicted Val', color='orange', dash_capstyle='round', dashes=[2, 2])
        plt.title('Validation Set - Ground Truth vs Prediction (Sample)')
        plt.xlabel('Time Steps')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('val_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 3: Test Predictions (zoomed in to 500 records max for clarity)
        plot_len_test = min(500, len(test_actual_inv))
        plt.figure(figsize=(12, 5))
        plt.plot(test_actual_inv[:plot_len_test], label='Actual Test', color='lightgray', linewidth=2)
        plt.plot(test_preds_inv[:plot_len_test], label='Predicted Test', color='green', dash_capstyle='round', dashes=[2, 2])
        plt.title('Test Set - Ground Truth vs Prediction (Sample)')
        plt.xlabel('Time Steps')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('test_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations generated and saved successfully as PNGs.")
    except Exception as e:
        logger.error(f"Failed to generate predictions plots: {e}")

if __name__ == "__main__":
    train_model(model_type='Transformer')
