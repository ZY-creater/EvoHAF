#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Subregion Prediction Module (predictor.py)

This file contains the prediction modules of the model:
1. Subregion feature extraction and scoring
2. Temporal attention correction
3. Different head modules based on prediction tasks
4. Subregion score visualization functions

Design Philosophy:
- Extract subregion scores from high-resolution feature maps
- Use attention mechanism for temporal correction
- Design specific prediction heads for different tasks
- Provide visualization interface for subregion scores
"""

from venv import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from .layers import TemporalAttention, SubregionFeatureExtractor


class SubregionScorer(nn.Module):
    """
    Subregion Scoring Module: Extract subregion features and convert to scores
    
    Args:
        feature_dim (int): Feature dimension
        hidden_dim (int): Hidden layer dimension
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super(SubregionScorer, self).__init__()
        
        self.feature_extractor = SubregionFeatureExtractor(feature_dim)
        
        # Score conversion network
        self.score_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features, segmentation):
        """
        Forward propagation
        
        Args:
            features (torch.Tensor): Feature map [batch_size, time_steps, channels, D, H, W]
            segmentation (torch.Tensor): Segmentation map [batch_size, 1, D, H, W], contains region IDs
        
        Returns:
            torch.Tensor: Subregion scores [batch_size, time_steps, num_regions, 1]
            torch.Tensor: Subregion features [batch_size, time_steps, num_regions, feature_dim]
        """
        # Extract subregion features
        region_features = self.feature_extractor(features, segmentation)
        
        # Calculate subregion scores
        batch_size, time_steps, num_regions, feature_dim = region_features.shape
        region_scores = self.score_mlp(region_features)
        
        return region_scores, region_features


class TemporalScoreCorrector(nn.Module):
    """
    Temporal Score Corrector: Adjust subregion scores based on temporal attention
    
    Args:
        feature_dim (int): Feature dimension
        max_time_steps (int): Maximum time steps (passed to TemporalAttention)
        dropout (float): Dropout probability
    """
    def __init__(self, feature_dim, max_time_steps, dropout=0.1):
        super(TemporalScoreCorrector, self).__init__()
        
        self.temporal_attention = TemporalAttention(feature_dim, max_time_steps=max_time_steps, dropout=dropout)
        
        self.score_refiner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, region_features, region_scores, mask=None):
        """
        Forward propagation
        
        Args:
            region_features (torch.Tensor): Subregion features [batch_size, time_steps, num_regions, feature_dim]
            region_scores (torch.Tensor): Initial subregion scores [batch_size, time_steps, num_regions, 1]
            mask (torch.Tensor, optional): Valid time step mask [batch_size, time_steps]
        
        Returns:
            torch.Tensor: Corrected subregion scores [batch_size, time_steps, num_regions, 1]
        """
        # Apply temporal attention
        attended_features = self.temporal_attention(region_features, mask)
        
        # Refine scores
        refined_scores = self.score_refiner(attended_features)
        
        # Residual connection: merge original scores and refined scores
        corrected_scores = region_scores + refined_scores
        
        return corrected_scores


class PredictionHead(nn.Module):
    """
    Prediction Head: Convert subregion features/scores to final predictions.
    Uses GRU to implement compensation mechanism based on historical predictions and labels.
    
    Args:
        feature_dim (int): Feature dimension
        time_steps (int): Maximum time steps (theoretically no longer strictly needed, but kept for shape inference)
        hidden_dim (int): MLP hidden layer dimension
        use_compensation (bool): Whether to use compensation module
        gru_hidden_dim (int): GRU hidden layer dimension (if use_compensation=True)
        compensation_mlp_dim (int): Compensation MLP hidden layer dimension
    """
    def __init__(self, feature_dim, time_steps=8, hidden_dim=64, 
                 use_compensation=True, gru_hidden_dim=32, compensation_mlp_dim=16):
        super(PredictionHead, self).__init__()
        
        self.feature_dim = feature_dim
        self.time_steps = time_steps
        self.use_compensation = use_compensation
        self.gru_hidden_dim = gru_hidden_dim
        
        if use_compensation:
            # GRU receives predictions and labels from previous time step, outputs hidden state
            # Input dimension: 1 (prediction) + 1 (label) = 2
            self.compensation_gru = nn.GRU(input_size=2, 
                                           hidden_size=gru_hidden_dim, 
                                           batch_first=True) 
                                           
            # MLP maps GRU output to compensation value
            self.compensation_mlp = nn.Sequential(
                nn.Linear(gru_hidden_dim, compensation_mlp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(compensation_mlp_dim, 1)
            )
    
    def forward(self, region_features, region_scores, targets=None):
        """
        Forward propagation
        
        Args:
            region_features (torch.Tensor): Subregion features [B, T, N_regions, F]
            region_scores (torch.Tensor): Subregion scores [B, T, N_regions, 1]
            targets (torch.Tensor, optional): Target labels [B, T], used for compensation module (provided during training)
                                                Can be omitted during inference, or provide previous prediction results
        
        Returns:
            torch.Tensor: Global prediction scores [B, T, 1]
            torch.Tensor: Subregion scores [B, T, N_regions, 1] (unchanged)
        """
        batch_size, time_steps_actual, num_regions, _ = region_scores.shape
        device = region_scores.device
        
        # Initialize output
        global_scores = torch.zeros(batch_size, time_steps_actual, 1, device=device)
        
        # --- Calculate base prediction for each step --- 
        # Here we use the average of region scores as base prediction, consistent with old logic
        # If projection based on region_features is needed, modify here
        base_predictions = region_scores.mean(dim=2) # [B, T, 1]
        
        global_scores[:, 0, :] = base_predictions[:, 0, :] # Time point 0 has no compensation
        
        # --- Apply GRU compensation (if enabled) --- 
        if self.use_compensation and time_steps_actual > 1:
            if targets is None:
                print("Warning: Compensation enabled but targets not provided during forward pass. Compensation will be skipped.")
                global_scores = base_predictions # No targets, cannot compensate, directly return base predictions
                return global_scores, region_scores

            # Get base predictions for first T-1 steps (without compensation, no detach)
            # detach would prevent gradient flow back to main model, if compensation should affect main model then don't detach
            gru_input_preds = base_predictions[:, :-1, 0] # [B, T_actual - 1]
            
            # Get labels for first T-1 steps
            # Need to ensure label time steps align with prediction time steps (take actual prediction steps T_actual - 1)
            num_pred_steps = gru_input_preds.shape[1] # T_actual - 1
            gru_input_labels = targets[:, :num_pred_steps].float() # [B, T_actual - 1]
            
            # Check again if label shape matches (theoretically should always match now)
            if gru_input_preds.shape != gru_input_labels.shape:
                print(f"ERROR: Shape mismatch persists after alignment! Preds: {gru_input_preds.shape}, Labels: {gru_input_labels.shape}. Skipping compensation.")
                global_scores = base_predictions
                return global_scores, region_scores

            gru_input = torch.stack([gru_input_preds, gru_input_labels], dim=-1) # [B, T_actual - 1, 2]
            
            # Through GRU (initial hidden state is 0)
            # gru_output: [B, T-1, gru_hidden_dim]
            gru_output, _ = self.compensation_gru(gru_input) 
            
            # Through MLP to calculate compensation value delta_t (t=1 to T-1)
            # compensation_deltas: [B, T-1, 1]
            compensation_deltas = self.compensation_mlp(gru_output)
            
            # Add compensation value to base predictions from T=1 to T-1
            global_scores[:, 1:, :] = base_predictions[:, 1:, :] + compensation_deltas
            
        else:
            # If not using compensation or only one time point
            global_scores = base_predictions
        
        return global_scores, region_scores

    def set_compensation_grad(self, requires_grad: bool):
        """Set whether compensation module (GRU and MLP) parameters require gradients"""
        if self.use_compensation:
             print(f"Setting compensation GRU/MLP requires_grad to: {requires_grad}")
             for param in self.compensation_gru.parameters():
                 param.requires_grad = requires_grad
             for param in self.compensation_mlp.parameters():
                 param.requires_grad = requires_grad
        else:
             print("Compensation module is not used, skipping set_compensation_grad.")


class LongitudinalPredictor(nn.Module):
    """
    Longitudinal Prediction Model: Prediction framework based on subregion analysis (supports GRU-based compensation)
    
    Args:
        feature_dim (int): Feature dimension
        max_time_steps (int): Maximum time steps
        hidden_dim (int): MLP hidden layer dimension (passed to Scorer and Head)
        use_compensation (bool): Whether to use compensation module in prediction head
        gru_hidden_dim (int): GRU hidden layer dimension
        compensation_mlp_dim (int): Compensation MLP hidden layer dimension
    """
    def __init__(self, feature_dim, max_time_steps=8, hidden_dim=64, 
                 use_compensation=True, gru_hidden_dim=32, compensation_mlp_dim=16):
        super(LongitudinalPredictor, self).__init__()
        
        self.feature_dim = feature_dim
        self.max_time_steps = max_time_steps
        
        # Subregion scorer
        self.subregion_scorer = SubregionScorer(feature_dim, hidden_dim)
        
        # Temporal score corrector - pass max_time_steps
        self.temporal_corrector = TemporalScoreCorrector(feature_dim, max_time_steps=self.max_time_steps)
        
        # Prediction head (now uses new PredictionHead)
        self.prediction_head = PredictionHead(feature_dim, max_time_steps, hidden_dim, 
                                              use_compensation, gru_hidden_dim, compensation_mlp_dim)
    
    def forward(self, features, segmentation, targets=None, mask=None):
        """
        Forward propagation
        
        Args:
            features (torch.Tensor): Feature map [B, T, C, D, H, W]
            segmentation (torch.Tensor): Segmentation map [B, 1, D, H, W]
            targets (torch.Tensor, optional): Target labels [B, T], used for compensation
            mask (torch.Tensor, optional): Valid time step mask [B, T]
        
        Returns:
            dict: Dictionary containing 'predictions' and 'subregion_scores'
        """
        # 1. Get initial subregion scores and features
        region_scores, region_features = self.subregion_scorer(features, segmentation)
        
        # 2. Apply temporal correction
        corrected_scores = self.temporal_corrector(region_features, region_scores, mask)
        
        # 3. Get final predictions through prediction head (compensation handled internally)
        
        global_predictions, final_region_scores = self.prediction_head(region_features, corrected_scores, targets)
        
        return {
            'predictions': global_predictions, # [B, T, 1]
            'subregion_scores': final_region_scores # [B, T, N_regions, 1]
        }

    def get_subregion_scores(self, features, segmentation, mask=None):
        """
        Helper function: Only get corrected subregion scores, used for visualization etc.
        
        Args:
            features (torch.Tensor): Feature map [B, T, C, D, H, W]
            segmentation (torch.Tensor): Segmentation map [B, 1, D, H, W]
            mask (torch.Tensor, optional): Valid time step mask [B, T]
        
        Returns:
            torch.Tensor: Corrected subregion scores [B, T, N_regions, 1]
        """
        region_scores, region_features = self.subregion_scorer(features, segmentation)
        corrected_scores = self.temporal_corrector(region_features, region_scores, mask)
        return corrected_scores 