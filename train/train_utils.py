#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Utility Functions (train_utils.py)

This file contains auxiliary utility functions used during training:
1. Set random seed to ensure reproducible results
2. Evaluation metrics calculation
3. Learning rate scheduler
4. Early stopping mechanism
5. Model checkpoint saving and loading
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
import yaml
import shutil
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    """
    Set random seed to ensure reproducible results
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set: {seed}")

def get_project_root():
    """Get project root directory"""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

def load_config(config_path):
    """
    Load YAML configuration file
    
    Args:
        config_path (str): Configuration file path
    
    Returns:
        dict: Configuration dictionary
    """
    try:
        # Check if it's an absolute path
        if os.path.isabs(config_path):
            full_path = config_path
        else:
            # If it's a relative path, make it relative to project root
            full_path = os.path.join(get_project_root(), config_path)
            
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration file: {full_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration file: {str(e)}")
        return None

def save_config(config, output_path):
    """
    Save configuration dictionary as YAML file
    
    Args:
        config (dict): Configuration dictionary
        output_path (str): Output file path
    """
    try:
        # Check if it's an absolute path
        if os.path.isabs(output_path):
            full_path = output_path
        else:
            # If it's a relative path, make it relative to project root
            full_path = os.path.join(get_project_root(), output_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Successfully saved configuration to: {full_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {str(e)}")

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """
    Save model checkpoint
    
    Args:
        state (dict): State dictionary containing model parameters and optimizer state
        is_best (bool): Whether this is the best model
        save_dir (str): Save directory
        filename (str): Filename
    """
    # Check if it's an absolute path
    if os.path.isabs(save_dir):
        full_dir = save_dir
    else:
        # If it's a relative path, make it relative to project root
        full_dir = os.path.join(get_project_root(), save_dir)
    
    # Ensure directory exists
    os.makedirs(full_dir, exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = os.path.join(full_dir, filename)
    torch.save(state, checkpoint_path)
    
    # If it's the best model, copy it as best_model.pth
    if is_best:
        best_path = os.path.join(full_dir, 'best_model.pth')
        shutil.copyfile(checkpoint_path, best_path)
        logger.info(f"Saved best model checkpoint: {best_path}")
    else:
        logger.info(f"Saved model checkpoint: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path (str): Checkpoint file path
        model (nn.Module): Model instance
        optimizer (torch.optim.Optimizer, optional): Optimizer instance
    
    Returns:
        int: Starting epoch
        float: Best evaluation score
    """
    try:
        # Check if it's an absolute path
        if os.path.isabs(checkpoint_path):
            full_path = checkpoint_path
        else:
            # If it's a relative path, make it relative to project root
            full_path = os.path.join(get_project_root(), checkpoint_path)
        
        # Load checkpoint
        checkpoint = torch.load(full_path, map_location=lambda storage, loc: storage)

        # Filter and compatibly load model parameters (ignore weights not involved in inference like loss functions)
        ckpt_state = checkpoint.get('model_state_dict', checkpoint)
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in ckpt_state.items() if k in model_state}
        dropped_keys = [k for k in ckpt_state.keys() if k not in model_state]
        missing_keys = [k for k in model_state.keys() if k not in filtered_state]

        load_res = model.load_state_dict(filtered_state, strict=False)
        if dropped_keys:
            logger.warning(f"Dropped keys from checkpoint that don't match current model: {len(dropped_keys)}, examples: {dropped_keys[:5]}")
        if missing_keys:
            logger.info(f"Missing keys in checkpoint from current model: {len(missing_keys)}, examples: {missing_keys[:5]}")
        # Compatible with PyTorch's IncompatibleKeys return object
        if hasattr(load_res, 'unexpected_keys') and load_res.unexpected_keys:
            logger.warning(f"Unexpected keys when loading state_dict (ignored): {load_res.unexpected_keys[:5]}")
        if hasattr(load_res, 'missing_keys') and load_res.missing_keys:
            logger.info(f"Missing keys when loading state_dict: {load_res.missing_keys[:5]}")

        # If optimizer is provided, load optimizer state (if exists)
        if optimizer is not None and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Return starting epoch and best evaluation score
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_score = checkpoint.get('best_score', 0.0)
        
        logger.info(f"Loaded checkpoint: {full_path}, starting from epoch {start_epoch}, best evaluation score: {best_score}")
        
        return start_epoch, best_score
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        return 0, 0.0

class EarlyStopping:
    """
    Early stopping mechanism, stops training when validation performance no longer improves
    
    Args:
        patience (int): Number of epochs to tolerate, stops if validation performance doesn't improve for this many consecutive epochs
        verbose (bool): Whether to print logs
        delta (float): Minimum change threshold, changes below this threshold are not considered improvements
        mode (str): 'min' means lower is better, 'max' means higher is better
        init_best_score (float, optional): Initial best score
    """
    def __init__(self, patience=10, verbose=True, delta=0, mode='max', init_best_score=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None  # Set to None first, then set based on mode and init_best_score
        self.best_metrics = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        
        # Initialize comparison function
        if self.mode == 'min':
            self.is_better = lambda score, best: score <= (best - self.delta)
            # Set initial best score
            self.best_score = init_best_score if init_best_score is not None else float('inf')
        else:  # max
            self.is_better = lambda score, best: score >= (best + self.delta)
            # Set initial best score
            self.best_score = init_best_score if init_best_score is not None else float('-inf')
    
    def __call__(self, score, metrics=None):
        """
        Update early stopping state
        
        Args:
            score (float): Current evaluation score (for comparison)
            metrics (dict, optional): Dictionary of all current evaluation metrics
        
        Returns:
            bool: Whether this is the best model
        """
        is_best = False
        
        # If score is better than current best score
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_metrics = metrics
            self.counter = 0
            is_best = True
            if self.verbose:
                logger.info(f"Validation performance improved (best score: {self.best_score:.6f})")
        else:
            self.counter += 1
            # Check if early stopping is needed
            if self.patience is not None:
                if self.counter >= self.patience:
                    self.early_stop = True
                    logger.info(f"Early stopping triggered (exceeded patience {self.patience})!")
                else:
                     # Only print remaining patience if patience is not None
                     logger.info(f"Validation performance did not improve (best score: {self.best_score:.6f}, patience remaining: {self.patience - self.counter})")
            else:
                 # If patience is None, only print performance not improved message
                 logger.info(f"Validation performance did not improve (best score: {self.best_score:.6f})")
        
        return is_best

def get_lr_scheduler(optimizer, config):
    """
    Get learning rate scheduler
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        config (dict): Scheduler configuration
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
    """
    scheduler_type = config.get('type', 'step')
    
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    elif scheduler_type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get('milestones', [30, 60, 90]),
            gamma=config.get('gamma', 0.1)
        )
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 100),
            eta_min=config.get('eta_min', 0)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'max'),
            factor=config.get('factor', 0.1),
            patience=config.get('patience', 10),
            verbose=True
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using default StepLR")
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def compute_metrics(y_true, y_pred_prob):
    """
    Calculate common metrics for binary classification tasks.

    Args:
        y_true (np.ndarray): True labels (0 or 1).
        y_pred_prob (np.ndarray): Model predicted probabilities.

    Returns:
        dict: Dictionary containing AUC, Accuracy, F1, Sensitivity, Specificity.
    """
    metrics = {}

    # AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred_prob)
    except ValueError as e:
        # Handle cases where only one class is present in y_true
        # logger.warning(f"Cannot calculate AUC: {e}. Only one class may exist in labels. Setting AUC to 0.5.")
        metrics['auc'] = 0.5

    # Other metrics need binary predictions
    y_pred_binary = (y_pred_prob > 0.5).astype(int)

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)

    # F1 Score
    metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)

    # Sensitivity (Recall) and Specificity
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall / True Positive Rate
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # True Negative Rate
    except ValueError as e:
        # Handle cases where confusion matrix might not be 2x2 (e.g., only one class predicted)
        # logger.warning(f"Error calculating confusion matrix or SEN/SPE: {e}. Only one class may exist in predictions or labels. Setting SEN/SPE to 0.")
        # Check unique values to provide more context
        # logger.warning(f"Unique true labels: {np.unique(y_true)}, Unique predicted labels: {np.unique(y_pred_binary)}")
        # Fallback based on simple cases
        if len(np.unique(y_true)) == 1:
            if np.unique(y_true)[0] == 1: # Only positive class
                metrics['sensitivity'] = accuracy_score(y_true, y_pred_binary) # Accuracy is sensitivity
                metrics['specificity'] = 0.0
            else: # Only negative class
                metrics['sensitivity'] = 0.0
                metrics['specificity'] = accuracy_score(y_true, y_pred_binary) # Accuracy is specificity
        else:
            metrics['sensitivity'] = 0.0
            metrics['specificity'] = 0.0

    return metrics

def prepare_directories(output_dir):
    """
    Prepare output directory
    
    Args:
        output_dir (str): Output directory path
    
    Returns:
        str: Full output directory path
    """
    # Check if it's an absolute path
    if os.path.isabs(output_dir):
        full_dir = output_dir
    else:
        # If it's a relative path, make it relative to project root
        full_dir = os.path.join(get_project_root(), output_dir)
    
    # Create directory
    os.makedirs(full_dir, exist_ok=True)
    logger.info(f"Created output directory: {full_dir}")
    
    # Create subdirectories
    checkpoints_dir = os.path.join(full_dir, 'checkpoints')
    logs_dir = os.path.join(full_dir, 'logs')
    results_dir = os.path.join(full_dir, 'results')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return full_dir
