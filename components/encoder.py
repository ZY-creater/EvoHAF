#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
High-Resolution Encoder Module (encoder.py)

This file contains the encoder part of the model:
1. Multi-modal 3D feature extractor
2. High-resolution encoder (using dilated convolution and hybrid 2D-3D convolution)
3. Modality fusion module

Design philosophy:
- Maintain high spatial resolution for subsequent subregion feature extraction
- Use dilated convolution to increase receptive field
- Hybrid 2D-3D convolution to reduce computational cost
- Support multi-modal data processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DilatedConvBlock, Hybrid2D3DConvBlock, ChannelAttention


class HighResolutionEncoder(nn.Module):
    """
    High-resolution encoder: uses dilated convolution and hybrid 2D-3D convolution to maintain spatial resolution

    Args:
        in_channels (int): Number of input channels (now represents number of modalities, e.g., T1CE+FLAIR=2)
        feature_channels (list): List of feature channels at each layer
        dilations (list): List of dilation rates
    """
    def __init__(self, in_channels, feature_channels=[32, 64, 128, 128], dilations=[1, 2, 4, 8]):
        super(HighResolutionEncoder, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, feature_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Create dilated convolution blocks
        self.dilated_blocks = nn.ModuleList()
        for i in range(len(feature_channels)-1):
            block = DilatedConvBlock(
                feature_channels[i], 
                feature_channels[i+1],
                dilation=dilations[i]
            )
            self.dilated_blocks.append(block)
        
        # Create hybrid 2D-3D convolution blocks
        self.hybrid_blocks = nn.ModuleList()
        for i in range(len(feature_channels)-1):
            block = Hybrid2D3DConvBlock(
                feature_channels[i+1], 
                feature_channels[i+1]
            )
            self.hybrid_blocks.append(block)
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Conv3d(feature_channels[-1], feature_channels[-1], kernel_size=1),
            nn.BatchNorm3d(feature_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Feature dimension (final output channels)
        self.feature_dim = feature_channels[-1]
    
    def forward(self, x):
        """
        Forward propagation

        Args:
            x (torch.Tensor): Input tensor [batch_size, time_steps, channels, D, H, W]
                             where channels now represents number of modalities (e.g., T1CE+FLAIR=2)

        Returns:
            torch.Tensor: Encoded features [batch_size, time_steps, feature_dim, D_out, H_out, W_out]
        """
        batch_size, time_steps, channels, D, H, W = x.shape
        
        # Initialize output feature list
        feature_maps = []
        
        # Process each time step
        for t in range(time_steps):
            # Input at current time step
            current_input = x[:, t]  # [batch_size, channels, D, H, W]
            
            # Initial convolution
            features = self.initial_conv(current_input)
            
            # Pass through dilated convolution blocks and hybrid convolution blocks sequentially
            for i in range(len(self.dilated_blocks)):
                # Dilated convolution
                features = self.dilated_blocks[i](features)
                # Hybrid 2D-3D convolution
                features = self.hybrid_blocks[i](features)
            
            # Final output layer
            features = self.output_layer(features)
            
            # Add to output list (unsqueeze(1) restores time dimension)
            feature_maps.append(features.unsqueeze(1))
        
        # Concatenate along time dimension
        output_features = torch.cat(feature_maps, dim=1)
        
        return output_features


class LongitudinalEncoder(nn.Module):
    """
    Longitudinal sequence encoder: processes multi-modal longitudinal data

    Args:
        num_modalities (int): Number of modalities (e.g., T1CE+FLAIR=2)
        feature_channels (list): List of feature channels at each layer
        dilations (list): List of dilation rates
    """
    def __init__(self, num_modalities=2, feature_channels=[32, 64, 128, 128], dilations=[1, 2, 4, 8]):
        super(LongitudinalEncoder, self).__init__()
        
        self.high_res_encoder = HighResolutionEncoder(
            in_channels=num_modalities, # Input channels equal to number of modalities
            feature_channels=feature_channels,
            dilations=dilations
        )
        
        # Feature dimension (final output channels)
        self.feature_dim = feature_channels[-1]
    
    def forward(self, x):
        """
        Forward propagation

        Args:
            x (torch.Tensor): Input tensor [batch_size, time_steps, channels, D, H, W]
                              where channels represents number of modalities (e.g., T1CE+FLAIR=2)

        Returns:
            torch.Tensor: Encoded features [batch_size, time_steps, feature_dim, D_out, H_out, W_out]
        """
        # Directly use high-resolution encoder for processing
        output_features = self.high_res_encoder(x)
        
        return output_features 