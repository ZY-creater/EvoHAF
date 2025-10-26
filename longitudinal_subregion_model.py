#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Longitudinal Subregion Parsing Prediction Model (longitudinal_subregion_model.py)

This file contains the main model architecture, integrating various components:
1. Encoder
2. Subregion feature extraction
3. Subregion scoring
4. Temporal attention correction
5. Prediction head
6. Loss function

Model design philosophy:
- Maintain high-resolution features for spatial interpretability
- Use subregion representation for parsing prediction
- Support different types of prediction tasks
- Longitudinal data analysis considering temporal relationships
- Interpretability and visualization support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .components.encoder import LongitudinalEncoder
from .components.predictor import LongitudinalPredictor
from .components.losses import LongitudinalPredictionLoss


class LongitudinalSubregionModel(nn.Module):
    """
    Longitudinal Subregion Prediction Model (with GRU-based compensation support)

    Args:
        num_modalities (int): Number of modalities (e.g., T1CE+FLAIR=2)
        max_time_steps (int): Maximum number of time steps
        feature_dim (int): Feature dimension
        hidden_dim (int): Predictor MLP hidden layer dimension
        use_compensation (bool): Whether to use compensation module in prediction head
        gru_hidden_dim (int): GRU hidden layer dimension (if use_compensation=True)
        compensation_mlp_dim (int): Compensation MLP hidden layer dimension
        encoder_feature_channels (list): Encoder feature channels at each layer
        encoder_dilations (list): Encoder dilation rates at each layer
        smooth_weight (float): Smooth loss weight
        smooth_threshold (float): Smooth threshold
        class_weight (float or list): Class weight
    """
    def __init__(self, 
                 num_modalities=2, 
                 max_time_steps=8,
                 feature_dim=64, 
                 hidden_dim=32, 
                 use_compensation=True,
                 gru_hidden_dim=32, # New parameter
                 compensation_mlp_dim=16, # New parameter
                 encoder_feature_channels=[16, 32, 48, 64], 
                 encoder_dilations=[1, 2, 4, 8],
                 smooth_weight=0.2, 
                 smooth_threshold=0.2,
                 class_weight=None):
        super(LongitudinalSubregionModel, self).__init__()
        
        # Ensure the last layer of encoder_feature_channels matches feature_dim
        if not encoder_feature_channels or encoder_feature_channels[-1] != feature_dim:
             print(f"Warning: Last layer of encoder_feature_channels ({encoder_feature_channels[-1] if encoder_feature_channels else 'None'}) does not match feature_dim ({feature_dim}).\
                      Adjusting the last layer of encoder_feature_channels to {feature_dim}.")
             if not encoder_feature_channels:
                  encoder_feature_channels = [16, 32, feature_dim] 
             else:
                  encoder_feature_channels[-1] = feature_dim
            
        self.num_modalities = num_modalities
        self.max_time_steps = max_time_steps
        self.feature_dim = feature_dim
        self.use_compensation = use_compensation # Save use_compensation flag
        
        # Encoder
        self.encoder = LongitudinalEncoder(
            num_modalities=num_modalities,
            feature_channels=encoder_feature_channels,
            dilations=encoder_dilations
        )
        
        # Predictor (pass GRU-related parameters)
        self.predictor = LongitudinalPredictor(
            feature_dim=self.feature_dim, 
            max_time_steps=max_time_steps,
            hidden_dim=hidden_dim,
            use_compensation=use_compensation,
            gru_hidden_dim=gru_hidden_dim, # Pass
            compensation_mlp_dim=compensation_mlp_dim # Pass
        )
        
        # Loss function 
        self.loss_fn = LongitudinalPredictionLoss(
            smooth_weight=smooth_weight,
            smooth_threshold=smooth_threshold,
            class_weight=class_weight
        )
    
    def set_encoder_predictor_grad(self, requires_grad: bool):
        """Set whether encoder and predictor parameters (except compensation module GRU/MLP) require gradients"""
        print(f"Setting encoder & predictor (non-compensation) requires_grad to: {requires_grad}")
        # Freeze/unfreeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad
        # Freeze/unfreeze non-compensation parts of predictor
        for name, module in self.predictor.named_modules():
            # Skip entire prediction_head as its gradients are controlled separately by set_compensation_grad
            if 'prediction_head' in name:
                continue
            # Set gradients for other modules (e.g., subregion_scorer, temporal_corrector)
            # Ensure we only set parameters of top-level modules to avoid duplication
            # Check if it's a direct submodule (avoid going into nn.Sequential internals)
            is_direct_submodule = '.' not in name 
            if is_direct_submodule and name and name != 'prediction_head': # Exclude root module and prediction_head
                 print(f"  Setting grad for predictor submodule: {name}")
                 for param in module.parameters():
                      param.requires_grad = requires_grad
            elif name == "": # For predictor itself (if it has direct parameters)
                 pass # Usually predictor itself has no direct parameters
                 
        # Old logic (filtering by parameter name may not be robust enough)
        # for name, param in self.predictor.named_parameters():
        #     is_compensation_param = False
        #     if hasattr(self.predictor.prediction_head, 'compensation_gru') and \
        #        param in self.predictor.prediction_head.compensation_gru.parameters():
        #         is_compensation_param = True
        #     if hasattr(self.predictor.prediction_head, 'compensation_mlp') and \
        #        param in self.predictor.prediction_head.compensation_mlp.parameters():
        #         is_compensation_param = True
                
        #     if not is_compensation_param:
        #          param.requires_grad = requires_grad

    def set_compensation_grad(self, requires_grad: bool):
        """Set whether compensation module parameters require gradients (by calling PredictionHead's method)"""
        if hasattr(self.predictor.prediction_head, 'set_compensation_grad'):
            self.predictor.prediction_head.set_compensation_grad(requires_grad)
        else:
             print("Predictor head does not have set_compensation_grad method.")

    def get_compensation_parameters(self):
        """Get parameter list of compensation module (GRU and MLP) if used"""
        params = []
        # Check model's own use_compensation flag
        if self.use_compensation and hasattr(self.predictor, 'prediction_head'):
            head = self.predictor.prediction_head
            if hasattr(head, 'compensation_gru'):
                params.extend(list(head.compensation_gru.parameters()))
            if hasattr(head, 'compensation_mlp'):
                params.extend(list(head.compensation_mlp.parameters()))
        return params

    def forward(self, modality_inputs, segmentation, targets=None, mask=None):
        """
        Forward propagation

        Args:
            modality_inputs (torch.Tensor): Modality inputs [batch_size, time_steps, channels, D, H, W]
            segmentation (torch.Tensor): Segmentation map [batch_size, 1, D, H, W]
            targets (torch.Tensor, optional): Target labels [batch_size, time_steps], for compensation module
            mask (torch.Tensor, optional): Valid time step mask [batch_size, time_steps]

        Returns:
            dict: Dictionary containing prediction results and subregion information
        """
        # Encode multi-modal inputs
        features = self.encoder(modality_inputs)
        
        # Predict (pass targets and mask)
        outputs = self.predictor(features, segmentation, targets, mask)
        
        return outputs
    
    def calculate_loss(self, outputs, targets, valid_mask=None):
        """
        Calculate loss

        Args:
            outputs (dict): Model outputs
            targets (torch.Tensor): Target labels [batch_size, time_steps]
            valid_mask (torch.Tensor, optional): Valid sample mask [batch_size, time_steps]

        Returns:
            torch.Tensor: Total loss
            dict: Loss components
        """
        predictions = outputs["predictions"]
        region_scores = outputs["subregion_scores"]
        
        # Align targets with predictions' time_steps
        pred_time_steps = predictions.shape[1]
        if targets.shape[1] > pred_time_steps:
             targets = targets[:, :pred_time_steps]
             if valid_mask is not None and valid_mask.shape[1] > pred_time_steps:
                 valid_mask = valid_mask[:, :pred_time_steps]

        # Adjust if valid_mask is None or has fewer time steps than predictions
        if valid_mask is None or valid_mask.shape[1] < pred_time_steps:
             if valid_mask is not None:
                 padding_size = pred_time_steps - valid_mask.shape[1]
                 padding = torch.ones(valid_mask.shape[0], padding_size, device=valid_mask.device)
                 valid_mask = torch.cat([valid_mask, padding], dim=1)
             else:
                 valid_mask = torch.ones(targets.shape[0], pred_time_steps, device=targets.device)
        
        # Ensure valid_mask matches predictions' time dimension
        if valid_mask.shape[1] > pred_time_steps:
            valid_mask = valid_mask[:, :pred_time_steps]

        return self.loss_fn(predictions, region_scores, targets, valid_mask)
    
    def predict(self, modality_inputs, segmentation, targets=None, mask=None):
        """
        Prediction function

        Args:
            modality_inputs (torch.Tensor): Modality inputs [batch_size, time_steps, channels, D, H, W]
            segmentation (torch.Tensor): Segmentation map [batch_size, 1, D, H, W]
            targets (torch.Tensor, optional): Target labels [batch_size, time_steps], for compensation module (optional during inference)
            mask (torch.Tensor, optional): Valid time step mask [batch_size, time_steps]

        Returns:
            torch.Tensor: Prediction probabilities [batch_size, time_steps, 1]
        """
        with torch.no_grad():
            outputs = self.forward(modality_inputs, segmentation, targets, mask)
            predictions = outputs["predictions"]
            return torch.sigmoid(predictions)
    
    def get_subregion_scores(self, modality_inputs, segmentation, mask=None):
        """
        Get subregion scores for visualization and interpretation

        Args:
            modality_inputs (torch.Tensor): Modality inputs [batch_size, time_steps, channels, D, H, W]
            segmentation (torch.Tensor): Segmentation map [batch_size, 1, D, H, W]
            mask (torch.Tensor, optional): Valid time step mask [batch_size, time_steps]

        Returns:
            torch.Tensor: Subregion scores [batch_size, time_steps, num_regions, 1]
        """
        with torch.no_grad():
            features = self.encoder(modality_inputs)
            return self.predictor.get_subregion_scores(features, segmentation, mask)
    
    def visualize_subregion_scores(self, modality_inputs, segmentation, mask=None):
        """
        Visualize subregion scores

        Args:
            modality_inputs (torch.Tensor): Modality inputs [batch_size, time_steps, channels, D, H, W]
            segmentation (torch.Tensor): Segmentation map [batch_size, 1, D, H, W]
            mask (torch.Tensor, optional): Valid time step mask [batch_size, time_steps]

        Returns:
            None (displays image directly)
        """
        with torch.no_grad():
            features = self.encoder(modality_inputs)
            # Visualization is now done inside predictor
            self.predictor.visualize_subregion_scores(features, segmentation, mask)


# Model configuration for glioma post-radiotherapy progression prediction
class GliomaProgressionModel(LongitudinalSubregionModel):
    """
    Glioma Post-Radiotherapy Progression Prediction Model (using LongitudinalSubregionModel base class)

    Args (inherited from base class with default values or passed through):
        num_modalities (int): Number of modalities, default is 2 (T1CE, FLAIR)
        max_time_steps (int): Maximum number of time steps
        feature_dim (int): Feature dimension
        hidden_dim (int): Predictor MLP hidden layer dimension
        use_compensation (bool): Whether to use compensation module
        gru_hidden_dim (int): GRU hidden layer dimension
        compensation_mlp_dim (int): Compensation MLP hidden layer dimension
        encoder_feature_channels (list): Encoder feature channels at each layer
        encoder_dilations (list): Encoder dilation rates at each layer
        smooth_weight (float): Smooth loss weight
        smooth_threshold (float): Smooth threshold
        class_weight (float): Positive class weight
    """
    def __init__(self, 
                 num_modalities=2, 
                 max_time_steps=8, 
                 feature_dim=128, 
                 hidden_dim=64,
                 use_compensation=True,
                 gru_hidden_dim=32, # New parameter
                 compensation_mlp_dim=16, # New parameter
                 encoder_feature_channels=[32, 64, 96, 128],
                 encoder_dilations=[1, 2, 4, 8],
                 smooth_weight=0.2, 
                 smooth_threshold=0.2, 
                 class_weight=None):
        
        # Call base class constructor, passing all parameters
        super(GliomaProgressionModel, self).__init__(
            num_modalities=num_modalities,
            max_time_steps=max_time_steps,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            use_compensation=use_compensation,
            gru_hidden_dim=gru_hidden_dim,
            compensation_mlp_dim=compensation_mlp_dim,
            encoder_feature_channels=encoder_feature_channels,
            encoder_dilations=encoder_dilations,
            smooth_weight=smooth_weight,
            smooth_threshold=smooth_threshold,
            class_weight=class_weight
        )
        
        # GliomaProgressionModel-specific initialization (if needed) can be placed here
        pass