#!/bin/bash

# Training script for Evolutionary Heterogeneity Analysis Framework (EvoHAF)
# This script runs the training with K-fold cross-validation

# Configuration
CONFIG_PATH="config/config_gpu0.yaml"
GPU_ID=0

echo "=========================================="
echo "  EvoHAF Training Script"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "GPU ID: $GPU_ID"
echo "=========================================="

# Set GPU device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run training
cd train
python train.py --config ../$CONFIG_PATH

echo "=========================================="
echo "  Training Complete!"
echo "=========================================="
