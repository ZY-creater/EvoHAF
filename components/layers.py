#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic Layers and Module Components (layers.py)

This file contains basic layers and components used by the model:
1. Dilated Convolution Block (DilatedConvBlock)
2. Channel Attention Mechanism (ChannelAttention)
3. Hybrid 2D-3D Convolution Block (Hybrid2D3DConvBlock)
4. Temporal Attention Module (TemporalAttention)
5. Subregion Feature Extraction Module (SubregionFeatureExtractor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DilatedConvBlock(nn.Module):
    """
    Dilated Convolution Block: Use dilated convolution to increase receptive field without reducing resolution
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size
        dilation (int): Dilation rate
        stride (int): Stride
        padding (int): Padding
        use_batch_norm (bool): Whether to use batch normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, 
                 stride=1, padding=1, use_batch_norm=True):
        super(DilatedConvBlock, self).__init__()
        
        # Calculate padding for dilated convolution to maintain output size
        if padding == 'same':
            padding = (dilation * (kernel_size - 1)) // 2
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                             dilation=dilation, stride=stride, padding=padding)
        
        layers = []
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.additional_layers = nn.Sequential(*layers) if layers else nn.Identity()
    
    def forward(self, x):
        """Forward propagation"""
        x = self.conv(x)
        x = self.additional_layers(x)
        return x


class ChannelAttention(nn.Module):
    """
    Channel Attention Mechanism: Weight channel dimensions
    
    Args:
        channels (int): Number of input channels
        reduction_ratio (int): Reduction ratio
    """
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        reduced_channels = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward propagation"""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        
        return x * channel_attention


class Hybrid2D3DConvBlock(nn.Module):
    """
    Hybrid 2D-3D Convolution Block: Combine 2D and 3D convolutions to reduce computational cost
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        use_attention (bool): Whether to use channel attention
    """
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(Hybrid2D3DConvBlock, self).__init__()
        
        self.conv_d = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), 
                               padding=(1, 0, 0))
        self.conv_h = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 1), 
                               padding=(0, 1, 0))
        self.conv_w = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 3), 
                               padding=(0, 0, 1))
        
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_fuse = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn_fuse = nn.BatchNorm3d(out_channels)
        self.relu_fuse = nn.ReLU(inplace=True)
        
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention(in_channels)
    
    def forward(self, x):
        """Forward propagation"""
        # Decompose into 2D convolutions in three directions
        x_d = self.conv_d(x)
        x_h = self.conv_h(x)
        x_w = self.conv_w(x)
        
        # Fuse features from three directions
        x = x_d + x_h + x_w
        x = self.bn(x)
        x = self.relu(x)
        
        # Apply channel attention (if enabled)
        if self.use_attention:
            x = self.channel_attention(x)
        
        # 1x1 convolution fusion
        x = self.conv_fuse(x)
        x = self.bn_fuse(x)
        x = self.relu_fuse(x)
        
        return x


class TemporalAttention(nn.Module):
    """
    Temporal Attention Module: Process longitudinal time series relationships with relative position encoding.
    
    Args:
        feature_dim (int): Feature dimension
        max_time_steps (int): Maximum expected time steps, used for relative position embedding table
        num_heads (int): Number of heads in multi-head attention
        dropout (float): Dropout probability
    """
    def __init__(self, feature_dim, max_time_steps, num_heads=4, dropout=0.1):
        super(TemporalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.max_time_steps = max_time_steps
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
        self.layer_norm = nn.LayerNorm(feature_dim)

        # --- New: Relative position bias embedding table ---
        # Range from -(max_time_steps - 1) to +(max_time_steps - 1), total 2 * max_time_steps - 1 values
        self.relative_position_bias = nn.Embedding(2 * max_time_steps - 1, 1) 
        # Initialize bias to 0 is a reasonable starting point
        nn.init.zeros_(self.relative_position_bias.weight)
        # -----------------------------
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量 [batch_size, time_steps, num_regions, feature_dim]
            mask (torch.Tensor, optional): 掩码，标记有效/无效时间步 [batch_size, time_steps_padded]
                                           (注意：这里的 time_steps 可能比 x 的长，需要裁剪)
        
        返回:
            torch.Tensor: 注意力加权后的特征
        """
        batch_size, time_steps_feat, num_regions, feature_dim = x.size() # 获取特征的实际时间步 T_feat
        device = x.device
        
        # 重塑输入以便进行多头注意力计算
        x_reshaped = x.view(batch_size * num_regions, time_steps_feat, feature_dim)
        residual = x_reshaped
        
        # 投影查询、键、值
        q = self.query(x_reshaped) # [B*num_regions, T_feat, feature_dim]
        k = self.key(x_reshaped)   # [B*num_regions, T_feat, feature_dim]
        v = self.value(x_reshaped) # [B*num_regions, T_feat, feature_dim]
        
        # 计算注意力分数
        # (B*num_regions, T_feat, feature_dim) x (B*num_regions, feature_dim, T_feat) -> (B*num_regions, T_feat, T_feat)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.feature_dim)

        # --- 新增：计算并添加相对位置偏置 ---
        # 1. 生成相对位置索引矩阵 [T_feat, T_feat]
        position_ids_l = torch.arange(time_steps_feat, device=device).view(-1, 1)
        position_ids_r = torch.arange(time_steps_feat, device=device).view(1, -1)
        relative_position_ids = position_ids_l - position_ids_r  # 范围 [-T_feat+1, T_feat-1]

        # 2. 将相对位置索引映射到嵌入表的有效索引 [0, 2*max_time_steps - 2]
        # 注意加上 self.max_time_steps - 1 进行偏移
        relative_position_indices = relative_position_ids + self.max_time_steps - 1
        
        # 3. 裁剪索引以确保在 [0, 2*max_time_steps - 2] 范围内
        relative_position_indices = torch.clamp(
            relative_position_indices, 
            min=0, 
            max=2 * self.max_time_steps - 2
        )

        # 4. 从嵌入表中查找偏置 [T_feat, T_feat, 1]
        relative_bias = self.relative_position_bias(relative_position_indices)

        # 5. 调整形状并添加到注意力分数中
        # attn_scores: [B*num_regions, T_feat, T_feat]
        # relative_bias: [T_feat, T_feat, 1] -> [1, T_feat, T_feat]
        attn_scores = attn_scores + relative_bias.squeeze(-1).unsqueeze(0)
        # -----------------------------------

        # 应用掩码（如果提供）- 处理padding
        if mask is not None:
            # 确保 mask 的 batch_size 匹配
            assert mask.shape[0] == batch_size, f"Mask batch size {mask.shape[0]} != Feature batch size {batch_size}"
            
            # --- 关键：裁剪 mask 以匹配特征的时间维度 --- 
            current_mask_time_steps = mask.shape[1]
            if current_mask_time_steps > time_steps_feat:
                mask = mask[:, :time_steps_feat] # 裁剪 [B, T_padded] -> [B, T_feat]
            elif current_mask_time_steps < time_steps_feat:
                # 如果 mask 比特征短 (理论上不应发生)，进行填充
                pad_size = time_steps_feat - current_mask_time_steps
                mask = F.pad(mask, (0, pad_size), mode='constant', value=0)
            # --- 裁剪结束，现在 mask shape is [B, T_feat] --- 

            # 为注意力分数创建掩码
            # attn_scores shape: [B*num_regions, T_feat, T_feat]
            # 我们需要一个掩码来屏蔽掉无效的 key 列
            # mask shape: [B, T_feat]
            attn_mask = mask.unsqueeze(1).repeat(1, num_regions, 1) # [B, num_regions, T_feat]
            attn_mask = attn_mask.view(batch_size * num_regions, time_steps_feat) # [B*num_regions, T_feat]
            attn_mask = attn_mask.unsqueeze(1) # 扩展为 [B*num_regions, 1, T_feat] -> 用于屏蔽 key 列
            
            # 创建用于屏蔽 query 行的掩码 (如果 query 时间步无效，其注意力也应屏蔽)
            attn_mask_query = attn_mask.transpose(-1, -2) # [B*num_regions, T_feat, 1]
            
            # 应用掩码：将无效 key 列 和无效 query 行 对应的分数设置为负无穷
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
            attn_scores = attn_scores.masked_fill(attn_mask_query == 0, -1e9)

        # --- 新增：应用因果掩码 (Causal Mask) ---
        # 创建一个上三角矩阵（不包括对角线），True 代表未来的位置
        # causal_mask shape: [T_feat, T_feat]
        causal_mask = torch.triu(torch.ones(time_steps_feat, time_steps_feat, device=device, dtype=torch.bool), diagonal=1)
        # 将 attn_scores 中对应未来的位置设置为负无穷
        # causal_mask shape: [T_feat, T_feat] -> [1, T_feat, T_feat] for broadcasting
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0), -1e9)
        # --------------------------------------

        # 注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1) # [B*num_regions, T_feat, T_feat]
        attn_weights = self.attn_dropout(attn_weights)
        
        # 应用注意力权重
        # (B*num_regions, T_feat, T_feat) x (B*num_regions, T_feat, feature_dim) -> (B*num_regions, T_feat, feature_dim)
        context = torch.matmul(attn_weights, v)
        
        # 投影输出
        output = self.proj(context) # [B*num_regions, T_feat, feature_dim]
        output = self.proj_dropout(output)
        
        # 残差连接和层归一化
        output = self.layer_norm(residual + output)
        
        # 重塑回原始维度
        output = output.view(batch_size, num_regions, time_steps_feat, feature_dim).permute(0, 2, 1, 3) # [B, T_feat, num_regions, feature_dim]
        
        return output


# --- 新增：基于 GRU 的时序模块 ---
class TemporalGRU(nn.Module):
    """
    基于 GRU 的时序处理模块。
    处理 [B, T, N, F] 的输入，其中 T 是时间步。
    使用 GRU 处理每个区域的时间序列。

    参数:
        feature_dim (int): 特征维度
        hidden_dim (int): GRU 隐藏层维度
        num_layers (int): GRU 层数
        dropout (float): Dropout 概率
    """
    def __init__(self, feature_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(TemporalGRU, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # GRU 层，注意 batch_first=True
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0, # GRU 内置 dropout 只在多层时有效
            bidirectional=False # 通常时序预测用单向
        )

        # 如果 hidden_dim 与 feature_dim 不同，需要一个线性层来匹配维度以进行残差连接
        if hidden_dim != feature_dim:
            self.proj = nn.Linear(hidden_dim, feature_dim)
        else:
            self.proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入张量 [batch_size, time_steps, num_regions, feature_dim]
            mask (torch.Tensor, optional): 掩码

        返回:
            torch.Tensor: GRU 处理后的特征 [batch_size, time_steps, num_regions, feature_dim]
        """
        batch_size, time_steps, num_regions, feature_dim = x.size()
        residual = x

        # 重塑以便 GRU 处理: [B, T, N, F] -> [B*N, T, F]
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(batch_size * num_regions, time_steps, feature_dim)

        # 通过 GRU
        # 输出形状: [B*N, T, hidden_dim]
        gru_output, _ = self.gru(x_reshaped) # 我们只需要输出序列，不需要最后的隐藏状态

        # 投影回 feature_dim (如果需要)
        projected_output = self.proj(gru_output) # [B*N, T, feature_dim]

        # 重塑回原始维度: [B*N, T, F] -> [B, N, T, F] -> [B, T, N, F]
        output_reshaped = projected_output.view(batch_size, num_regions, time_steps, feature_dim).permute(0, 2, 1, 3)

        # 应用 Dropout, 残差连接 和 LayerNorm
        output = self.layer_norm(residual + self.dropout(output_reshaped))

        return output

# --- 新增：基于 LSTM 的时序模块 ---
class TemporalLSTM(nn.Module):
    """
    基于 LSTM 的时序处理模块。
    处理 [B, T, N, F] 的输入。

    参数:
        feature_dim (int): 特征维度
        hidden_dim (int): LSTM 隐藏层维度
        num_layers (int): LSTM 层数
        dropout (float): Dropout 概率
    """
    def __init__(self, feature_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(TemporalLSTM, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        if hidden_dim != feature_dim:
            self.proj = nn.Linear(hidden_dim, feature_dim)
        else:
            self.proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入张量 [B, T, N, F]
            mask (torch.Tensor, optional): 掩码 (当前实现未使用)

        返回:
            torch.Tensor: LSTM 处理后的特征 [B, T, N, F]
        """
        batch_size, time_steps, num_regions, feature_dim = x.size()
        residual = x

        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(batch_size * num_regions, time_steps, feature_dim)
        lstm_output, _ = self.lstm(x_reshaped) # 输出: [B*N, T, hidden_dim]
        projected_output = self.proj(lstm_output) # [B*N, T, feature_dim]
        output_reshaped = projected_output.view(batch_size, num_regions, time_steps, feature_dim).permute(0, 2, 1, 3)
        output = self.layer_norm(residual + self.dropout(output_reshaped))

        return output

# --- 新增：基于 RNN 的时序模块 ---
class TemporalRNN(nn.Module):
    """
    基于 RNN 的时序处理模块。
    处理 [B, T, N, F] 的输入。

    参数:
        feature_dim (int): 特征维度
        hidden_dim (int): RNN 隐藏层维度
        num_layers (int): RNN 层数
        dropout (float): Dropout 概率
        nonlinearity (str): 'tanh' 或 'relu'
    """
    def __init__(self, feature_dim, hidden_dim, num_layers=1, dropout=0.1, nonlinearity='tanh'):
        super(TemporalRNN, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.RNN(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        if hidden_dim != feature_dim:
            self.proj = nn.Linear(hidden_dim, feature_dim)
        else:
            self.proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入张量 [B, T, N, F]
            mask (torch.Tensor, optional): 掩码 (当前实现未使用)

        返回:
            torch.Tensor: RNN 处理后的特征 [B, T, N, F]
        """
        batch_size, time_steps, num_regions, feature_dim = x.size()
        residual = x

        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(batch_size * num_regions, time_steps, feature_dim)
        rnn_output, _ = self.rnn(x_reshaped) # 输出: [B*N, T, hidden_dim]
        projected_output = self.proj(rnn_output) # [B*N, T, feature_dim]
        output_reshaped = projected_output.view(batch_size, num_regions, time_steps, feature_dim).permute(0, 2, 1, 3)
        output = self.layer_norm(residual + self.dropout(output_reshaped))

        return output

# --- 新增：基于 MLP 的时序模块 (Time-Distributed MLP) ---
class TemporalMLP(nn.Module):
    """
    基于 MLP 的时序处理模块 (Time-Distributed MLP)。
    独立地对每个时间步的特征应用 MLP。不直接建模时间依赖性，可作为基线。
    处理 [B, T, N, F] 的输入。

    参数:
        feature_dim (int): 特征维度
        mlp_hidden_dim (int): MLP 隐藏层维度 (通常是 feature_dim 的倍数)
        dropout (float): Dropout 概率
    """
    def __init__(self, feature_dim, mlp_hidden_dim, dropout=0.1):
        super(TemporalMLP, self).__init__()
        self.feature_dim = feature_dim
        self.mlp_hidden_dim = mlp_hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden_dim),
            nn.ReLU(), # 或者 nn.GELU()
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, feature_dim)
        )

        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout) # 输出前的 Dropout

    def forward(self, x, mask=None):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入张量 [B, T, N, F]
            mask (torch.Tensor, optional): 掩码 (当前实现未使用)

        返回:
            torch.Tensor: MLP 处理后的特征 [B, T, N, F]
        """
        batch_size, time_steps, num_regions, feature_dim = x.size()
        residual = x

        # 重塑以便 MLP 处理: [B, T, N, F] -> [B*T*N, F]
        x_reshaped = x.contiguous().view(batch_size * time_steps * num_regions, feature_dim)

        # 通过 MLP
        mlp_output = self.mlp(x_reshaped) # [B*T*N, F]

        # 重塑回原始维度: [B*T*N, F] -> [B, T, N, F]
        output_reshaped = mlp_output.view(batch_size, time_steps, num_regions, feature_dim)

        # 应用 Dropout, 残差连接 和 LayerNorm
        # 注意：这里的 Dropout 应用在 MLP 内部以及最终输出前
        output = self.layer_norm(residual + self.dropout(output_reshaped))

        return output


class SubregionFeatureExtractor(nn.Module):
    """
    子区域特征提取模块：从特征图中提取子区域表示
    
    参数:
        feature_dim (int): 特征维度
    """
    def __init__(self, feature_dim):
        super(SubregionFeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
    
    def forward(self, features, segmentation):
        """
        从特征图提取子区域特征
        
        参数:
            features (torch.Tensor): 特征图 [batch_size, time_steps, channels, D, H, W]
            segmentation (torch.Tensor): 分割图 [batch_size, 1, D, H, W]，包含区域ID
        
        返回:
            torch.Tensor: 子区域特征 [batch_size, time_steps, num_regions, feature_dim]
        """
        batch_size, time_steps, channels, D, H, W = features.size()
        
        # 获取唯一区域ID（不包括背景0）
        # 假设segmentation是整数标签，背景是0，各子区域从1开始编号
        region_ids = torch.unique(segmentation)
        region_ids = region_ids[region_ids > 0]  # 排除背景
        num_regions = len(region_ids)
        
        # 初始化输出张量，设备与输入 features 保持一致
        region_features = torch.zeros(batch_size, time_steps, num_regions, self.feature_dim, device=features.device)
        
        # 对每个时间步骤处理
        for b in range(batch_size):
            for t in range(time_steps):
                feature_map = features[b, t]  # [channels, D, H, W]
                seg_map = segmentation[b, 0]  # [D, H, W]
                
                # 将分割图调整为与特征图相同的空间尺寸
                if seg_map.shape != feature_map.shape[1:]:
                    # 为3D体积数据创建适当的插值
                    target_size = feature_map.shape[1:]  # [D_out, H_out, W_out]
                    
                    # 创建一个临时的4D张量进行插值 (添加batch维度)
                    seg_map_4d = seg_map.unsqueeze(0).float()  # [1, D, H, W]
                    
                    # 对3D体积数据进行插值
                    seg_map_resized = F.interpolate(
                        seg_map_4d.unsqueeze(0),  # [1, 1, D, H, W]
                        size=target_size,
                        mode='nearest'
                    ).squeeze(0).squeeze(0).long()  # [D_out, H_out, W_out]
                    
                    seg_map = seg_map_resized
                
                # 对每个区域提取特征
                for i, region_id in enumerate(region_ids):
                    # 创建二进制掩码
                    mask = (seg_map == region_id).float()  # [D, H, W]
                    
                    # 防止空区域
                    if mask.sum() == 0:
                        continue
                    
                    # 扩展维度以匹配特征图
                    mask = mask.unsqueeze(0).expand_as(feature_map)  # [channels, D, H, W]
                    
                    # 提取区域内的特征并计算平均值
                    masked_features = feature_map * mask
                    region_features[b, t, i] = (masked_features.sum(dim=(1, 2, 3)) / mask[0].sum())
        
        return region_features 