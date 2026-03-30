#!/bin/bash
# SolarAI Automated Pipeline Execution

echo "========================================="
echo "🌙 Initiating SolarAI Forecasting Pipeline"
echo "========================================="

# Activate virtual environment
source .venv/bin/activate

# 1. Train the Neural Model and Generate Analytics Visualizations
echo ""
echo "[Step 1/3] 🧠 Training Transformer Neural Network..."
python train.py

# 2. Advanced Walk-Forward Validation
echo ""
echo "[Step 2/3] 🧪 Validating Diagnostics..."
python validate_model.py

# 3. Launching Intelligence Suite
echo ""
echo "[Step 3/3] 🚀 Launching Intelligence Suite Dashboard..."
streamlit run dashboard.py
