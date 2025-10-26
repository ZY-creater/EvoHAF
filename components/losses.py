#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Loss Function Module (losses.py)

This file contains loss functions used by the model:
1. Loss function for sequential prediction tasks
2. Smooth prediction loss function

Design Philosophy:
- Sequential prediction tasks use binary cross-entropy loss
- Smooth prediction loss function prevents drastic fluctuations in prediction scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialPredictionLoss(nn.Module):
    """
    Loss function for sequential prediction tasks: Used for post-radiotherapy progression prediction
    
    Args:
        smooth_weight (float): Weight of smooth loss
        smooth_threshold (float): Smooth threshold
        class_weight (list): Class weights (usually weight for positive class)
    """
    def __init__(self, smooth_weight=0.2, smooth_threshold=0.2, class_weight=None):
        super(SequentialPredictionLoss, self).__init__()
        
        self.smooth_weight = smooth_weight
        self.smooth_threshold = smooth_threshold
        
        # Binary cross-entropy loss (compute element-wise loss)
        pos_weight_tensor = None
        if class_weight is not None:
            # Assume class_weight is a single value representing positive class weight
            pos_weight_tensor = torch.tensor(class_weight).float()
            
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_tensor)
    
    def forward(self, predictions, region_scores, targets, valid_mask=None):
        """
        Calculate loss
        
        Args:
            predictions (torch.Tensor): Global predictions [batch_size, time_steps, 1]
            region_scores (torch.Tensor): Subregion scores [batch_size, time_steps, num_regions, 1]
            targets (torch.Tensor): Target labels [batch_size, time_steps]
            valid_mask (torch.Tensor, optional): Valid sample mask [batch_size, time_steps]
        
        Returns:
            torch.Tensor: Total loss
            dict: Loss components
        """
        batch_size, time_steps, _ = predictions.shape
        
        # Process target labels
        targets = targets.float().view(batch_size, time_steps, 1)
        
        # Valid sample mask
        if valid_mask is None:
            valid_mask = torch.ones_like(targets)
        else:
            # Ensure valid_mask has correct shape and move to device
            valid_mask = valid_mask.float().view(batch_size, time_steps, 1).to(predictions.device)
        
        # Calculate unreduced classification loss
        loss_unreduced = self.ce_loss(predictions, targets)
        
        # Apply mask to calculate average classification loss
        # Only sum over valid time steps and take average
        class_loss = (loss_unreduced * valid_mask).sum() / valid_mask.sum().clamp(min=1e-8) 
        
        # Calculate smooth loss
        smooth_loss = 0.0
        num_steps = 0
        
        # Calculate score differences between adjacent time steps
        for t in range(1, time_steps):
            # Check if current and previous time steps are both valid
            curr_valid = valid_mask[:, t, 0] > 0.5 # Use > 0.5 to handle possible floating point issues
            prev_valid = valid_mask[:, t-1, 0] > 0.5
            
            # Find batch indices where both time steps are valid
            valid_indices = (curr_valid & prev_valid).nonzero(as_tuple=True)[0]

            if len(valid_indices) == 0:
                continue
                
            # Get region scores for current and previous time steps of valid samples
            curr_scores = region_scores[valid_indices, t]  # [num_valid, num_regions, 1]
            prev_scores = region_scores[valid_indices, t-1]  # [num_valid, num_regions, 1]
            
            # Calculate change in region scores
            score_diff = curr_scores - prev_scores  # [num_valid, num_regions, 1]
            
            # Only penalize changes exceeding threshold
            smooth_threshold = self.smooth_threshold
            penalty = F.relu(torch.abs(score_diff) - smooth_threshold) ** 2
            
            # Calculate average penalty
            step_smooth_loss = penalty.mean()
            smooth_loss += step_smooth_loss
            num_steps += 1
        
        # If there are more than one valid time step pairs, calculate average smooth loss
        if num_steps > 0:
            smooth_loss = smooth_loss / num_steps
        # If no valid time step pairs, ensure smooth_loss is a 0 tensor
        elif isinstance(smooth_loss, float):
             smooth_loss = torch.tensor(0.0, device=predictions.device)

        # Total loss
        total_loss = class_loss + self.smooth_weight * smooth_loss
        
        # Return total loss and loss components
        return total_loss, {
            "class_loss": class_loss.item(),
            "smooth_loss": smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else smooth_loss,
            "total_loss": total_loss.item()
        }


class LongitudinalPredictionLoss(nn.Module):
    """
    Loss function for longitudinal prediction model (only supports sequential prediction)
    
    Args:
        smooth_weight (float): Weight of smooth loss
        smooth_threshold (float): Smooth threshold
        class_weight (float or list): Class weights
    """
    def __init__(self, smooth_weight=0.2, smooth_threshold=0.2, class_weight=None):
        super(LongitudinalPredictionLoss, self).__init__()
        
        self.loss_fn = SequentialPredictionLoss(
            smooth_weight=smooth_weight,
            smooth_threshold=smooth_threshold,
            class_weight=class_weight
        )
    
    def forward(self, predictions, region_scores, targets, valid_mask=None):
        """
        Calculate loss
        
        Args:
            predictions (torch.Tensor): Global predictions
            region_scores (torch.Tensor): Subregion scores [batch_size, time_steps, num_regions, 1]
            targets (torch.Tensor): Target labels
            valid_mask (torch.Tensor, optional): Valid sample mask
        
        Returns:
            torch.Tensor: Total loss
            dict: Loss components
        """
        return self.loss_fn(predictions, region_scores, targets, valid_mask) 