#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K-Fold Cross-Validation Training Script - train.py

This script trains longitudinal models using K-fold cross-validation.
Features include:
1. K-fold cross-validation split of the dataset
2. Two-stage training strategy:
   - Stage 1: Train encoder and predictor
   - Stage 2: Fine-tune compensation prediction head (optional)
3. Save best model and performance metrics for each fold
4. Summarize cross-validation results across all folds
"""

import argparse
import os
from pathlib import Path
import sys
import yaml
import logging
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from sklearn.model_selection import KFold
import shutil # For copying files

# Add opensource directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
opensource_root = os.path.abspath(os.path.join(current_dir, '..'))
if opensource_root not in sys.path:
    sys.path.insert(0, opensource_root)

# --- Project module imports ---
from dataset.longitudinal_dataset import LongitudinalGliomaDataset
from dataset.utils import collate_longitudinal_batch
from longitudinal_subregion_model import GliomaProgressionModel
from train.trainer import LongitudinalTrainer
from train.train_utils import (
    save_checkpoint, load_checkpoint, compute_metrics,
    EarlyStopping, get_lr_scheduler, prepare_directories
)
from utils.utils import set_seed, prepare_directories, save_config, load_config, get_project_root

# --- Logging setup ---
# Configure root logger using basicConfig
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Ensure output to console
# Get logger for current module
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Execute K-fold cross-validation training')
    parser.add_argument('--config', type=str, 
                        default='your_config_path',
                        help='Configuration file path (relative to project root)')

    return parser.parse_args()

def create_detailed_glioma_config():
    """Create default training configuration"""
    config = {}

    config['num_folds'] = 5
    config['seed'] = 42
    config['skip_existing_fold'] = True
    config['gpu_id'] = 0
    config['output_dir_base'] = 'your_output_dir_base' # Modify default output directory

    config['data'] = {
        'batch_size': 8,
        'lazy_loading': False,
        'max_time_points': 8,
        'modality_list': ['T1CE', 'FLAIR'],
        'num_workers': 8,
        'resample_size': [128, 128, 128],
        'target_key': 'responses',
        'train_json': 'your_train_json.json',
        'train_segmentation_key': 'your_train_segmentation_key',
    }
    config['model'] = {
        'class_weight': 1.0,
        'encoder_dilations': [1, 2, 4, 8],
        'encoder_feature_channels': [16, 32, 48, 64],
        'feature_dim': 64,
        'hidden_dim': 32,
        'smooth_threshold': 0.2,
        'smooth_weight': 0.1,
        'use_compensation': True,
        'gru_hidden_dim': 32,
        'compensation_mlp_dim': 16
    }
    config['training'] = {
        'stage1_epochs': 100,
        'stage1_patience': 15,
        'stage1_lr': 0.001,
        'stage1_weight_decay': 0.0001,
        'stage1_checkpoint_interval': 10,
        'stage2_epochs': 50,
        'stage2_lr': 0.0001
    }
    return config

def save_detailed_glioma_config(config_path):
    """Save default training configuration"""
    logger.info(f"Creating default configuration file at: {config_path}")
    config = create_detailed_glioma_config()
    save_config(config, config_path)
    return config

# For JSON serialization of numpy types (moved to front for use by run_fold)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def run_fold(fold_idx, train_indices, val_indices, full_dataset, config, fold_output_dir, fold_seed):
    """
    Run training and internal evaluation for a single cross-validation fold (two-stage).

    Args:
        fold_idx (int): Current fold number (0-based).
        train_indices (list): Sample indices for training in this fold.
        val_indices (list): Sample indices for internal validation in this fold.
        full_dataset (Dataset): Complete main dataset (for creating Subset).
        config (dict): Complete configuration dictionary.
        fold_output_dir (str): Output directory for current fold.
        fold_seed (int): Random seed used for current fold.

    Returns:
        tuple: (best_internal_metrics_stage1, best_internal_metrics_stage2)
               best_internal_metrics_stage1 (dict): Best internal validation metrics for Stage 1.
               best_internal_metrics_stage2 (dict or None): Best internal validation metrics for Stage 2, None if Stage 2 not executed or not improved.
               Returns (None, None) if training completely fails.
    """
    # Initialize return results
    best_internal_metrics_stage1 = None
    best_internal_metrics_stage2 = None
    
    # Path definitions
    checkpoints_dir = os.path.join(fold_output_dir, 'checkpoints')
    results_dir = os.path.join(fold_output_dir, 'results')
    best_ckpt_stage1_path = os.path.join(checkpoints_dir, 'best_model_stage1.pth')
    best_ckpt_stage2_path = os.path.join(checkpoints_dir, 'best_model_stage2.pth')
    best_ckpt_final_path = os.path.join(checkpoints_dir, 'best_model.pth') # Final best model
    metrics_stage1_path = os.path.join(results_dir, 'best_internal_metrics_stage1.json')
    metrics_stage2_path = os.path.join(results_dir, 'best_internal_metrics_stage2.json')
    stage2_completion_flag = os.path.join(results_dir, 'stage2_completed.flag') # Use this flag to determine if Stage 2 is completed
    
    logger.info(f"===== Start processing Fold {fold_idx + 1}/{config['num_folds']} (without independent test) ====")
    logger.info(f"Output directory: {fold_output_dir}")

    # Check whether to skip existing fold (based on final best model checkpoint)
    if config.get('skip_existing_fold', False) and os.path.exists(best_ckpt_final_path):
        logger.warning(f"Fold {fold_idx + 1}'s final best checkpoint {best_ckpt_final_path} already exists, try to load results and skip training.")
        try:
            # Try to load metrics from both stages
            if os.path.exists(metrics_stage1_path):
                with open(metrics_stage1_path, 'r') as f:
                    best_internal_metrics_stage1 = json.load(f)
            if os.path.exists(metrics_stage2_path):
                with open(metrics_stage2_path, 'r') as f:
                    best_internal_metrics_stage2 = json.load(f)
            
            if best_internal_metrics_stage1: # At least need Stage 1 results to skip
                logger.info(f"Successfully loaded existing metrics for Fold {fold_idx + 1}. Stage 1: {best_internal_metrics_stage1}, Stage 2: {best_internal_metrics_stage2}")
                return best_internal_metrics_stage1, best_internal_metrics_stage2
            else:
                logger.warning(f"Found final checkpoint but unable to load Stage 1 metrics ({metrics_stage1_path}), will retrain.")
        except Exception as e:
            logger.warning(f"Failed to load existing metrics: {e}, will retrain.")
        # If loading fails, continue with training

    # --- Prepare directories and configuration ---
    prepare_directories(fold_output_dir)
    prepare_directories(checkpoints_dir)
    prepare_directories(results_dir)
    save_config(config, os.path.join(fold_output_dir, 'config_fold.yaml'))
    
    # --- Save dataset indices ---
    try:
        train_indices_list = [int(idx) for idx in train_indices]
        val_indices_list = [int(idx) for idx in val_indices]
        train_indices_path = os.path.join(fold_output_dir, 'train_indices.json')
        val_indices_path = os.path.join(fold_output_dir, 'val_indices.json')
        with open(train_indices_path, 'w') as f: json.dump(train_indices_list, f)
        with open(val_indices_path, 'w') as f: json.dump(val_indices_list, f)
        logger.info(f"Saved training and validation set indices.")
    except Exception as e:
        logger.warning(f"Error saving dataset indices: {e}")

    # --- Set random seed and get configuration ---
    set_seed(fold_seed)
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    num_workers_to_use = data_config.get('num_workers', 0)
    gpu_id_to_use = config.get('gpu_id', 0)
    logger.info(f"Fold {fold_idx + 1}: Using num_workers = {num_workers_to_use}, GPU ID = {gpu_id_to_use}")

    # --- Create dataset and data loaders ---
    try:
        fold_train_dataset = Subset(full_dataset, train_indices)
        fold_val_dataset = Subset(full_dataset, val_indices)
        logger.info(f"Fold {fold_idx + 1}: Number of training samples={len(fold_train_dataset)}, Number of internal validation samples={len(fold_val_dataset)}")

        if len(fold_train_dataset) == 0 or len(fold_val_dataset) == 0:
             logger.error(f"Fold {fold_idx + 1}'s training or validation set is empty, cannot continue.")
             return None, None

        fold_train_loader = DataLoader(fold_train_dataset, batch_size=data_config['batch_size'], shuffle=True,
                                   num_workers=num_workers_to_use, collate_fn=collate_longitudinal_batch, pin_memory=True)
        fold_val_loader = DataLoader(fold_val_dataset, batch_size=data_config['batch_size'], shuffle=False,
                                 num_workers=num_workers_to_use, collate_fn=collate_longitudinal_batch, pin_memory=True)
        logger.info(f"Fold {fold_idx + 1}: Data loader created successfully.")
    except Exception as e:
        logger.error(f"Fold {fold_idx + 1}: Error creating dataset or DataLoader: {e}", exc_info=True)
        return None, None

    # --- Create model ---
    try:
        model = GliomaProgressionModel(
            num_modalities=len(data_config['modality_list']), max_time_steps=data_config['max_time_points'],
            feature_dim=model_config['feature_dim'], encoder_feature_channels=model_config['encoder_feature_channels'],
            encoder_dilations=model_config['encoder_dilations'], hidden_dim=model_config['hidden_dim'],
            use_compensation=model_config.get('use_compensation', True), gru_hidden_dim=model_config.get('gru_hidden_dim', 32),
            compensation_mlp_dim=model_config.get('compensation_mlp_dim', 16), smooth_weight=model_config['smooth_weight'],
            smooth_threshold=model_config['smooth_threshold'], class_weight=model_config.get('class_weight', None)
        )
        logger.info(f"Fold {fold_idx + 1}: Model created successfully.")
    except Exception as e:
        logger.error(f"Fold {fold_idx + 1}: Error creating model: {e}", exc_info=True)
        return None, None

    # --- Create optimizer (Stage 1) ---
    try:
        optimizer_stage1 = optim.Adam(model.parameters(), lr=training_config['stage1_lr'], weight_decay=training_config['stage1_weight_decay'])
        logger.info(f"Fold {fold_idx + 1}: Stage 1 optimizer created successfully.")
    except Exception as e:
        logger.error(f"Fold {fold_idx + 1}: Error creating Stage 1 optimizer: {e}", exc_info=True)
        return None, None

    # === Stage 1: Train Encoder and Predictor ===
    logger.info(f"Fold {fold_idx + 1}: === Start Stage 1 training ===")
    try:
        # Stage 1: Freeze compensation module if used
        if model_config.get('use_compensation', True):
            model.set_compensation_grad(requires_grad=False)
            model.set_encoder_predictor_grad(requires_grad=True)
            logger.info(f"Fold {fold_idx + 1}, Stage 1: Compensation module frozen.")
        
        # trainer_stage1's output_dir points to fold_output_dir, checkpoints saved in checkpoints subdirectory
        trainer_stage1 = LongitudinalTrainer(
            model=model, optimizer=optimizer_stage1, train_loader=fold_train_loader, valid_loader=fold_val_loader,
            scheduler_config=None, output_dir=fold_output_dir, # Trainer will automatically create checkpoints subdirectory
            max_epochs=training_config['stage1_epochs'], patience=training_config['stage1_patience'],
            checkpoint_interval=training_config['stage1_checkpoint_interval'],
            primary_metric='auc', metric_mode='max', gpu_id=gpu_id_to_use
        )
        logger.info(f"Fold {fold_idx + 1}: Stage 1 trainer created successfully.")

        # --- Train Stage 1 --- 
        logger.info(f"Fold {fold_idx + 1}: Start Stage 1 training... ({training_config['stage1_epochs']} epochs, patience={training_config['stage1_patience']})")
        history_stage1 = trainer_stage1.train() # Trainer internally saves best_model.pth and checkpoint_epoch_X.pth
        logger.info(f"Fold {fold_idx + 1}: Stage 1 training completed.")

        # Get Stage 1 best results
        # Trainer saves best checkpoint with fixed path
        temp_best_ckpt_path = os.path.join(checkpoints_dir, 'best_model.pth') 
        
        if temp_best_ckpt_path and os.path.exists(temp_best_ckpt_path):
            # Rename Stage 1 best checkpoint (Trainer saves best_model.pth -> best_model_stage1.pth)
            try:
                shutil.move(temp_best_ckpt_path, best_ckpt_stage1_path)
                logger.info(f"Fold {fold_idx + 1}, Stage 1: Best checkpoint renamed from {os.path.basename(temp_best_ckpt_path)} to {os.path.basename(best_ckpt_stage1_path)}")
            except Exception as e:
                logger.error(f"Renaming Stage 1 best checkpoint failed: {e}", exc_info=True)
                # If renaming fails, try copying
                try:
                    shutil.copyfile(temp_best_ckpt_path, best_ckpt_stage1_path)
                    logger.info(f"Fold {fold_idx + 1}, Stage 1: Best checkpoint copied as {os.path.basename(best_ckpt_stage1_path)}")
                except Exception as copy_e:
                     logger.error(f"Copying Stage 1 best checkpoint also failed: {copy_e}", exc_info=True)
                     # Log error but try to continue, as metrics may have been obtained

            # Get and save Stage 1 best metrics
            if hasattr(trainer_stage1, 'early_stopping') and trainer_stage1.early_stopping.best_metrics:
                 best_internal_metrics_stage1 = trainer_stage1.early_stopping.best_metrics
                 logger.info(f"Fold {fold_idx + 1}, Stage 1: Internal validation set best metrics: {best_internal_metrics_stage1}")
                 try:
                     with open(metrics_stage1_path, 'w') as f:
                         json.dump(best_internal_metrics_stage1, f, indent=4, cls=NpEncoder)
                     logger.info(f"Fold {fold_idx + 1}, Stage 1: Best internal metrics saved to {metrics_stage1_path}")
                 except Exception as save_err:
                      logger.warning(f"Error saving Stage 1 best internal validation metrics: {save_err}")
            else:
                logger.warning(f"Fold {fold_idx + 1}, Stage 1: Unable to get best metrics from EarlyStopping.")
                # Try to get from last history record (may not be best)
                if history_stage1 and 'valid_metrics' in history_stage1[-1]:
                    best_internal_metrics_stage1 = history_stage1[-1]['valid_metrics']
                    logger.warning(f"Fold {fold_idx + 1}, Stage 1: Using last epoch's validation metrics as alternative: {best_internal_metrics_stage1}")
                else:
                     best_internal_metrics_stage1 = {'auc': 0.0, 'accuracy': 0.0, 'f1': 0.0} # Provide default value
                     logger.error(f"Fold {fold_idx + 1}, Stage 1: Unable to determine best metrics!")
        else:
            logger.error(f"Fold {fold_idx + 1}, Stage 1: Expected best checkpoint file not found after training: {temp_best_ckpt_path}. Stage 1 failed.")
            return None, None # Stage 1 failed, cannot continue

    except Exception as e:
        logger.error(f"Fold {fold_idx + 1}, Stage 1: Error during training or evaluation: {e}", exc_info=True)
        return None, None

    # === Stage 2: Train Compensation Module (Optional) ===
    run_stage2 = model_config.get('use_compensation', True) and training_config.get('stage2_epochs', 0) > 0
    # temp_best_ckpt_stage2_path = None # 用于临时存储Trainer保存的best_model.pth
    stage2_improved = False # 标记Stage 2是否找到了更好的模型

    if run_stage2:
        logger.info(f"Fold {fold_idx + 1}: === Start Stage 2 training (Compensation Module) ===")
        logger.info(f"Fold {fold_idx + 1}: Using Stage 1 best checkpoint {best_ckpt_stage1_path} as starting point.")
        
        try:
            # 1. Reload Stage 1 best model state (just in case trainer internal state is inconsistent)
            load_checkpoint(best_ckpt_stage1_path, model=model) 
            logger.info(f"Fold {fold_idx + 1}, Stage 2: Loaded Stage 1 best weights.")

            # 2. Set gradients: only train compensation module
            # model.set_encoder_predictor_grad(requires_grad=False) # Don't freeze backbone, otherwise gradients cannot propagate
            model.set_compensation_grad(requires_grad=True)     # Ensure compensation module requires gradients
            # logger.info(f"Fold {fold_idx + 1}, Stage 2: Encoder/Predictor frozen, Compensation unfrozen.")
            logger.info(f"Fold {fold_idx + 1}, Stage 2: Compensation grad set to True. Encoder/Predictor grads remain True, but optimizer will only update compensation params.")

            # 3. Create Stage 2 optimizer (only optimize Compensation parameters)
            compensation_params = model.get_compensation_parameters()
            if not compensation_params:
                 logger.warning(f"Fold {fold_idx + 1}, Stage 2: Compensation module parameters not found, skip Stage 2.")
                 run_stage2 = False # Update flag
            else:
                 optimizer_stage2 = optim.Adam(compensation_params, lr=training_config['stage2_lr'])
                 logger.info(f"Fold {fold_idx + 1}: Stage 2 optimizer created successfully.")

                 # 4. Create Stage 2 trainer
                 initial_score_s1 = best_internal_metrics_stage1.get('auc', 0.0)
                 trainer_stage2 = LongitudinalTrainer(
                     model=model, optimizer=optimizer_stage2, train_loader=fold_train_loader, valid_loader=fold_val_loader,
                     scheduler_config=None, output_dir=fold_output_dir, # Points to same directory
                     max_epochs=training_config['stage2_epochs'],
                     patience=None, # Stage 2 usually fixed epochs
                     checkpoint_interval=training_config['stage2_epochs'], # Can only save at the end, or save by interval
                     primary_metric='auc', metric_mode='max', gpu_id=gpu_id_to_use,
                     initial_best_score=initial_score_s1 # Use Stage 1 best score as baseline
                 )
                 logger.info(f"Fold {fold_idx + 1}: Stage 2 trainer created successfully (will optimize based on Stage 1 best AUC {initial_score_s1:.4f}).")

                 # 5. Train Stage 2
                 logger.info(f"Fold {fold_idx + 1}: Start Stage 2 training... ({training_config['stage2_epochs']} epochs)")
                 history_stage2 = trainer_stage2.train() # Trainer will update or create best_model.pth
                 logger.info(f"Fold {fold_idx + 1}: Stage 2 training completed.")

                 # Get Stage 2 best results
                 # Trainer saved best checkpoint path is still checkpoints/best_model.pth
                 temp_best_ckpt_path_s2 = os.path.join(checkpoints_dir, 'best_model.pth')
                 
                 if temp_best_ckpt_path_s2 and os.path.exists(temp_best_ckpt_path_s2):
                    # Check if Stage 2 really improved the model (compare scores)
                    current_best_score_s2 = trainer_stage2.early_stopping.best_score if hasattr(trainer_stage2, 'early_stopping') else 0.0
                    if current_best_score_s2 > initial_score_s1:
                        stage2_improved = True
                        logger.info(f"Fold {fold_idx + 1}, Stage 2: Performance improved (AUC {current_best_score_s2:.4f} > {initial_score_s1:.4f}).")
                        # Rename Stage 2 best checkpoint (Trainer saves best_model.pth -> best_model_stage2.pth)
                        try:
                            shutil.move(temp_best_ckpt_path_s2, best_ckpt_stage2_path)
                            logger.info(f"Fold {fold_idx + 1}, Stage 2: Best checkpoint renamed from {os.path.basename(temp_best_ckpt_path_s2)} to {os.path.basename(best_ckpt_stage2_path)}")
                        except Exception as e:
                           logger.error(f"Renaming Stage 2 best checkpoint failed: {e}", exc_info=True)
                           # Try copying
                           try:
                               shutil.copyfile(temp_best_ckpt_path_s2, best_ckpt_stage2_path)
                               logger.info(f"Fold {fold_idx + 1}, Stage 2: Best checkpoint copied as {os.path.basename(best_ckpt_stage2_path)}")
                           except Exception as copy_e:
                               logger.error(f"Copying Stage 2 best checkpoint also failed: {copy_e}", exc_info=True)
                               stage2_improved = False # Mark unable to get Stage 2 checkpoint
                    else:
                         logger.info(f"Fold {fold_idx + 1}, Stage 2: Best score after training did not exceed Stage 1 (AUC {current_best_score_s2:.4f} <= {initial_score_s1:.4f}). Do not save Stage 2 specific checkpoint.")
                         # If no improvement, Trainer saved best_model.pth is actually old, or not saved, we don't need to move it
                         # If file exists, may need to delete it to avoid confusion, or ensure subsequent logic doesn't use it
                         if os.path.exists(temp_best_ckpt_path_s2):
                             try:
                                 os.remove(temp_best_ckpt_path_s2)
                                 logger.info(f"Fold {fold_idx + 1}, Stage 2: Removed unimproved {os.path.basename(temp_best_ckpt_path_s2)} file.")
                             except OSError as e:
                                 logger.warning(f"Unable to remove unimproved Stage 2 checkpoint file {temp_best_ckpt_path_s2}: {e}")
                                 

                    # Get and save Stage 2 best metrics (only when Stage 2 improved)
                    if stage2_improved:
                         if hasattr(trainer_stage2, 'early_stopping') and trainer_stage2.early_stopping.best_metrics:
                             best_internal_metrics_stage2 = trainer_stage2.early_stopping.best_metrics
                             logger.info(f"Fold {fold_idx + 1}, Stage 2: Internal validation set best metrics: {best_internal_metrics_stage2}")
                         else:
                             logger.warning(f"Fold {fold_idx + 1}, Stage 2: Unable to get best metrics from EarlyStopping, will re-evaluate {best_ckpt_stage2_path}")
                             # Need to manually evaluate stage 2 best checkpoint
                             if os.path.exists(best_ckpt_stage2_path):
                                 try:
                                     evaluator_s2 = LongitudinalTrainer(model=model, optimizer=None, train_loader=None, valid_loader=fold_val_loader, output_dir=fold_output_dir, gpu_id=gpu_id_to_use)
                                     load_checkpoint(best_ckpt_stage2_path, model=model) # Load Stage 2 best model
                                     _, best_internal_metrics_stage2, _, _, _ = evaluator_s2.evaluate(fold_val_loader, return_details=False)
                                     logger.info(f"Fold {fold_idx + 1}, Stage 2: Re-evaluated internal validation set best metrics: {best_internal_metrics_stage2}")
                                 except Exception as eval_e:
                                      logger.error(f"Fold {fold_idx + 1}, Stage 2: Re-evaluating checkpoint {best_ckpt_stage2_path} failed: {eval_e}", exc_info=True)
                                      best_internal_metrics_stage2 = None # Mark failure
                             else:
                                 logger.error(f"Fold {fold_idx + 1}, Stage 2: Best checkpoint {best_ckpt_stage2_path} does not exist, cannot re-evaluate.")
                                 best_internal_metrics_stage2 = None

                         # Save Stage 2 metrics
                         if best_internal_metrics_stage2:
                             try:
                                 with open(metrics_stage2_path, 'w') as f:
                                     json.dump(best_internal_metrics_stage2, f, indent=4, cls=NpEncoder)
                                 logger.info(f"Fold {fold_idx + 1}, Stage 2: Best internal metrics saved to {metrics_stage2_path}")
                             except Exception as save_err:
                                  logger.warning(f"Error saving Stage 2 best internal validation metrics: {save_err}")
                         else:
                              logger.error(f"Fold {fold_idx + 1}, Stage 2: Unable to determine best metrics!")
                    
                 else:
                     # If Trainer didn't save best_model.pth (possibly training ended early or other reasons)
                     logger.warning(f"Fold {fold_idx + 1}, Stage 2: Expected best checkpoint file not found after training: {temp_best_ckpt_path_s2}. Stage 2 may not have run or failed.")
                     # Ensure run_stage2 and stage2_improved are False
                     run_stage2 = False
                     stage2_improved = False

                 # Create Stage 2 completion flag file (mark as completed even without improvement)
                 Path(stage2_completion_flag).touch(exist_ok=True)

        except Exception as e:
            logger.error(f"Fold {fold_idx + 1}, Stage 2: Error during training or evaluation: {e}", exc_info=True)
            # Stage 2 failure does not affect returning Stage 1 results
            best_internal_metrics_stage2 = None
            run_stage2 = False # Mark Stage 2 failure
            stage2_improved = False

    else:
        # If skip Stage 2
        logger.info(f"Fold {fold_idx + 1}: Skip Stage 2 training.")
        best_internal_metrics_stage2 = None # Ensure return None

    # === Determine final best model and save as best_model.pth ===
    overall_best_metrics = None
    overall_best_checkpoint_to_copy = None
    
    primary_metric = 'auc' # Assume we mainly care about AUC
    
    s1_score = best_internal_metrics_stage1.get(primary_metric, 0.0) if best_internal_metrics_stage1 else 0.0
    s2_score = best_internal_metrics_stage2.get(primary_metric, 0.0) if best_internal_metrics_stage2 else 0.0
    
    # Only when Stage 2 successfully ran, metrics are valid, checkpoint exists and score is higher, then select Stage 2
    if stage2_improved and best_internal_metrics_stage2 and os.path.exists(best_ckpt_stage2_path) and s2_score >= s1_score:
        logger.info(f"Fold {fold_idx + 1}: Stage 2 model performance is better (AUC: {s2_score:.4f} >= {s1_score:.4f}). Will use {os.path.basename(best_ckpt_stage2_path)} as final best model.")
        overall_best_metrics = best_internal_metrics_stage2
        overall_best_checkpoint_to_copy = best_ckpt_stage2_path
    elif best_internal_metrics_stage1 and os.path.exists(best_ckpt_stage1_path):
        if run_stage2: # If ran Stage 2 but did not use its results
             logger.info(f"Fold {fold_idx + 1}: Stage 1 model performance is better (AUC: {s1_score:.4f} > {s2_score:.4f}) or Stage 2 failed/not improved. Will use {os.path.basename(best_ckpt_stage1_path)} as final best model.")
        else:
             logger.info(f"Fold {fold_idx + 1}: Stage 2 not run, use Stage 1 results. Will use {os.path.basename(best_ckpt_stage1_path)} as final best model.")
        overall_best_metrics = best_internal_metrics_stage1
        overall_best_checkpoint_to_copy = best_ckpt_stage1_path
    else:
        # If even Stage 1 checkpoint doesn't exist, there's a problem
        logger.error(f"Fold {fold_idx + 1}: Cannot find any valid best checkpoint (Stage 1 or Stage 2)!")

    # Copy final best checkpoint as best_model.pth
    if overall_best_checkpoint_to_copy:
        try:
            shutil.copyfile(overall_best_checkpoint_to_copy, best_ckpt_final_path)
            logger.info(f"Fold {fold_idx + 1}: Final best checkpoint ({os.path.basename(overall_best_checkpoint_to_copy)}) copied as {os.path.basename(best_ckpt_final_path)}")
        except Exception as e:
            logger.error(f"Copying final best checkpoint failed: {e}", exc_info=True)
    else:
        logger.error(f"Fold {fold_idx + 1}: Did not determine valid final best checkpoint path! Cannot create {os.path.basename(best_ckpt_final_path)}.")
        # Even if unable to copy, still return obtained metrics

    logger.info(f"===== Completed processing Fold {fold_idx + 1}/{config['num_folds']} ====")
    # Return best internal metrics from both stages
    return best_internal_metrics_stage1, best_internal_metrics_stage2

def main():
    """Main function, execute cross-validation process (without independent test)"""
    args = parse_args()
    project_root = get_project_root()

    # --- Load configuration --- 
    config_path_abs = os.path.join(project_root, args.config)
    if not os.path.exists(config_path_abs):
        # Remove logic for creating default configuration, force requirement for configuration file to exist
        logger.error(f"Configuration file does not exist: {config_path_abs}, please provide valid configuration file path. Exit.")
        return
        # logger.warning(f"Configuration file does not exist: {config_path_abs}, will create and use default configuration.")
        # config = save_detailed_glioma_config(config_path_abs) # No longer create default configuration
    
    logger.info(f"Loading configuration file: {config_path_abs}")
    config = load_config(config_path_abs)
    if config is None:
        logger.error(f"Loading configuration file failed: {config_path_abs}, exit.")
        return
    logger.info("Successfully loaded configuration from file.")
    print(config) # Print loaded configuration

    # --- Get configuration parameters ---
    seed = config.get('seed', 42)
    num_folds = config.get('num_folds', 5)
    # Base output directory read from configuration
    output_dir_base = os.path.join(project_root, config.get('output_dir_base', f'experiments/results/detailed_train_cv_2stage_no_test_{Path(args.config).stem}')) # Include configuration name in default directory
    num_workers = config['data'].get('num_workers', 0)
    gpu_id = config.get('gpu_id', 0)

    # --- Set random seed and prepare main output directory ---
    set_seed(seed)
    prepare_directories(output_dir_base)
    # Save final configuration used (base configuration)
    save_config(config, os.path.join(output_dir_base, 'config_base.yaml'))
    logger.info(f"Main output directory: {output_dir_base}")
    logger.info(f"Using parameters: seed={seed}, num_folds={num_folds}, gpu_id={gpu_id}, num_workers={num_workers}")

    # --- Load complete dataset --- 
    data_config = config['data']
    try:
        logger.info("Loading main dataset (for cross-validation)...")
        # Use train_json specified dataset for K-fold cross-validation
        main_dataset_json_path = os.path.join(project_root, data_config['train_json'])
        full_main_dataset = LongitudinalGliomaDataset(
            json_path=main_dataset_json_path,
            modality_list=data_config['modality_list'],
            segmentation_key=data_config.get('train_segmentation_key', data_config.get('segmentation_key')), # Prefer train_segmentation_key
            target_key=data_config['target_key'],
            max_time_points=data_config['max_time_points'],
            resample_size=data_config.get('resample_size'),
            lazy_loading=data_config.get('lazy_loading', True) # Read from configuration, default True to save memory
        )
        logger.info(f"Main dataset loaded successfully, number of samples: {len(full_main_dataset)}")
        if len(full_main_dataset) == 0:
             logger.error("Main dataset is empty, cannot perform cross-validation.")
             return

    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        return

    # --- Initialize cross-validation and result storage --- 
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    dataset_indices = list(range(len(full_main_dataset)))
    
    all_folds_metrics_s1 = [] # Store Stage 1 best internal validation metrics
    all_folds_metrics_s2 = [] # Store Stage 2 best internal validation metrics (if run)

    # --- Execute cross-validation --- 
    for fold_idx, (train_val_indices, test_indices) in enumerate(kf.split(dataset_indices)):
        # Split data into training and validation sets
        fold_train_indices = train_val_indices 
        fold_val_indices = test_indices 
        fold_output_dir = os.path.join(output_dir_base, f'fold_{fold_idx}')
        fold_seed = seed + fold_idx # Each fold uses different seed

        # Run single fold training and internal evaluation
        metrics_s1, metrics_s2 = run_fold(
            fold_idx, 
            fold_train_indices, 
            fold_val_indices, 
            full_main_dataset, 
            config, 
            fold_output_dir,
            fold_seed
        )
        
        # Collect results
        if metrics_s1: # Stage 1 must succeed
            all_folds_metrics_s1.append(metrics_s1)
            if metrics_s2: # Stage 2 is optional
                all_folds_metrics_s2.append(metrics_s2)
        else:
             logger.error(f"Fold {fold_idx + 1} did not successfully return Stage 1 metrics, will exclude from final statistics.")

    # --- Calculate and report final results --- 
    logger.info("=" * 30 + " Cross-validation completed " + "=" * 30)
    
    final_summary = {"num_total_folds": num_folds}
    
    # --- Stage 1 results summary ---
    if not all_folds_metrics_s1:
        logger.error("No successful folds or Stage 1 internal validation metrics collected, cannot calculate final statistics.")
    else:
        logger.info(f"Successfully completed {len(all_folds_metrics_s1)} folds of Stage 1 training and evaluation.")
        # Extract metrics
        internal_aucs_s1 = [m.get('auc', 0.0) for m in all_folds_metrics_s1]
        internal_accs_s1 = [m.get('accuracy', 0.0) for m in all_folds_metrics_s1]
        internal_f1s_s1 = [m.get('f1', 0.0) for m in all_folds_metrics_s1]
        
        # Calculate mean and standard deviation
        summary_s1 = {
            "num_successful_folds": len(all_folds_metrics_s1),
            "mean": {
                "auc": np.mean(internal_aucs_s1),
                "accuracy": np.mean(internal_accs_s1),
                "f1": np.mean(internal_f1s_s1)
            },
            "std": {
                "auc": np.std(internal_aucs_s1),
                "accuracy": np.std(internal_accs_s1),
                "f1": np.std(internal_f1s_s1)
            }
        }
        final_summary["internal_val_stage1"] = summary_s1

        # Print results
        logger.info("--- Internal validation set Stage 1 best performance (Mean ± Std) ---")
        logger.info(f"AUC:      {summary_s1['mean']['auc']:.4f} ± {summary_s1['std']['auc']:.4f}")
        logger.info(f"Accuracy: {summary_s1['mean']['accuracy']:.4f} ± {summary_s1['std']['accuracy']:.4f}")
        logger.info(f"F1 Score: {summary_s1['mean']['f1']:.4f} ± {summary_s1['std']['f1']:.4f}")
        
    # --- Stage 2 results summary (if run) ---
    if all_folds_metrics_s2: # Only when at least one fold successfully ran and returned Stage 2 metrics
        logger.info(f"Successfully completed {len(all_folds_metrics_s2)} folds of Stage 2 training and evaluation.")
        internal_aucs_s2 = [m.get('auc', 0.0) for m in all_folds_metrics_s2]
        internal_accs_s2 = [m.get('accuracy', 0.0) for m in all_folds_metrics_s2]
        internal_f1s_s2 = [m.get('f1', 0.0) for m in all_folds_metrics_s2]

        summary_s2 = {
            "num_successful_folds": len(all_folds_metrics_s2),
            "mean": {
                "auc": np.mean(internal_aucs_s2),
                "accuracy": np.mean(internal_accs_s2),
                "f1": np.mean(internal_f1s_s2)
            },
            "std": {
                "auc": np.std(internal_aucs_s2),
                "accuracy": np.std(internal_accs_s2),
                "f1": np.std(internal_f1s_s2)
            }
        }
        final_summary["internal_val_stage2"] = summary_s2

        logger.info("--- Internal validation set Stage 2 final performance (Mean ± Std) ---")
        logger.info(f"AUC:      {summary_s2['mean']['auc']:.4f} ± {summary_s2['std']['auc']:.4f}")
        logger.info(f"Accuracy: {summary_s2['mean']['accuracy']:.4f} ± {summary_s2['std']['accuracy']:.4f}")
        logger.info(f"F1 Score: {summary_s2['mean']['f1']:.4f} ± {summary_s2['std']['f1']:.4f}")
    else:
         logger.info("No Stage 2 results collected or Stage 2 not run/improved in any fold.")

    # --- Save detailed results and summary ---
    if all_folds_metrics_s1: # Only save when there are successful results
        detailed_results = {
            "fold_internal_metrics_stage1": all_folds_metrics_s1,
            "fold_internal_metrics_stage2": all_folds_metrics_s2, # May be empty list
            "summary": final_summary
        }
        results_path = os.path.join(output_dir_base, 'cross_validation_internal_results.json')
        try:
            with open(results_path, 'w') as f:
                json.dump(detailed_results, f, indent=4, cls=NpEncoder) 
            logger.info(f"Detailed cross-validation results saved to: {results_path}")
        except Exception as e:
            logger.error(f"Error saving CV results: {e}", exc_info=True)

if __name__ == "__main__":
    main()
