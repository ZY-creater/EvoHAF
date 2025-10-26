#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Trainer (trainer.py)

This file contains the trainer class for training longitudinal subregion parsing prediction models:
1. Support for training longitudinal medical imaging data
2. Support for sequential prediction tasks
3. Provide complete training, validation, and evaluation processes
4. Implement model saving, loading, and early stopping mechanisms
"""

import os
import time
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm import tqdm
from pathlib import Path

from ..dataset.utils import collate_longitudinal_batch
from .train_utils import (
    save_checkpoint, load_checkpoint, compute_metrics,
    EarlyStopping, get_lr_scheduler, prepare_directories
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LongitudinalTrainer:
    """
    Longitudinal Subregion Parsing Prediction Model Trainer (supports sequential prediction only)
    
    Args:
        model (nn.Module): Model instance
        optimizer (torch.optim.Optimizer): Optimizer
        train_loader (DataLoader): Training data loader
        valid_loader (DataLoader): Validation data loader
        scheduler_config (dict): Learning rate scheduler configuration
        device (torch.device): Training device
        output_dir (str): Output directory
        max_epochs (int): Maximum training epochs
        patience (int): Early stopping patience value
        checkpoint_interval (int): Checkpoint saving interval
        primary_metric (str): Metric for early stopping and best model selection
        metric_mode (str): Whether higher ('max') or lower ('min') metric values are better
        gpu_id (int): GPU ID
        initial_checkpoint_path (str, optional): Initial checkpoint path
        initial_best_score (float, optional): Initial best score
    """
    def __init__(
            self, 
            model, 
            optimizer, 
            train_loader, 
            valid_loader=None,
            scheduler_config=None,
            device=None,
            output_dir='experiments/results/our_longitudinal_model',
            max_epochs=100,
            patience=10,
            checkpoint_interval=5,
            primary_metric: str = 'auc',
            metric_mode: str = 'max',
            gpu_id: int = 0,
            initial_checkpoint_path=None,
            initial_best_score=None
        ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        # Set device (based on gpu_id)
        if device is not None:
            self.device = device # Prioritize passed device object
        elif torch.cuda.is_available() and gpu_id >= 0:
            self.device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Using GPU: {gpu_id}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
            
        self.model.to(self.device)
        
        # Training parameters
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_interval = checkpoint_interval
        self.primary_metric = primary_metric
        self.metric_mode = metric_mode
        self.initial_checkpoint_path = initial_checkpoint_path
        self.initial_best_score = initial_best_score  # Store initial best score
        
        # Learning rate scheduler
        if scheduler_config is not None:
            from .train_utils import get_lr_scheduler
            self.scheduler = get_lr_scheduler(optimizer, scheduler_config)
            self.scheduler_is_plateau = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        else:
            self.scheduler = None
            self.scheduler_is_plateau = False
        
        # Prepare output directory
        self.output_dir = prepare_directories(output_dir)
        self.checkpoints_dir = os.path.join(self.output_dir, 'checkpoints')
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        self.results_dir = os.path.join(self.output_dir, 'results')
        
        # Set up tensorboard
        self.writer = SummaryWriter(log_dir=self.logs_dir)
        
        # Training state - initialize best_score first, then create early_stopping
        self.start_epoch = 0
        self.best_score = initial_best_score if initial_best_score is not None else (0.0 if self.metric_mode == 'max' else float('inf'))  # Use initial best score
        self.current_epoch = 0
        self.global_step = 0
        
        # Early stopping mechanism (use passed parameters, but not pass primary_metric name)
        self.early_stopping = EarlyStopping(
            patience=patience, 
            verbose=True, 
            mode=self.metric_mode, # Use saved mode
            init_best_score=self.best_score # Now self.best_score is defined
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'valid_metrics': []
        }
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.max_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare data
            modality_inputs = batch['modality_inputs'].to(self.device)
            segmentation = batch['segmentation'].to(self.device)
            targets = batch['target'].to(self.device)
            mask = batch.get('mask').to(self.device)
            
            # Forward propagation
            outputs = self.model(modality_inputs, segmentation, targets=targets, mask=mask)
            
            # Calculate loss
            loss, loss_dict = self.model.calculate_loss(outputs, targets, mask)
            
            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss / (batch_idx + 1):.4f}"
            })
            
            # Record training information to tensorboard
            self.writer.add_scalar('Loss/train_batch', loss.item(), self.global_step)
            for loss_name, loss_value in loss_dict.items():
                self.writer.add_scalar(f'Loss/train_{loss_name}', loss_value, self.global_step)
            
            self.global_step += 1
        
        # Calculate average loss
        epoch_loss /= num_batches
        self.history['train_loss'].append(epoch_loss)
        
        # Record learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, self.current_epoch)
        
        return epoch_loss
    
    def validate(self):
        """Validate model"""
        if self.valid_loader is None:
            return None, None
        
        self.model.eval()
        epoch_loss = 0.0
        num_batches = len(self.valid_loader)
        
        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc="Validating"):
                # Prepare data
                modality_inputs = batch['modality_inputs'].to(self.device)
                segmentation = batch['segmentation'].to(self.device)
                targets = batch['target'].to(self.device)
                mask = batch.get('mask').to(self.device)
                
                # Forward propagation
                outputs = self.model(modality_inputs, segmentation, targets=targets, mask=mask)
                
                # Calculate loss
                loss, _ = self.model.calculate_loss(outputs, targets, mask)
                epoch_loss += loss.item()
                
                # Collect predictions and targets
                predictions = torch.sigmoid(outputs['predictions'])
                
                # Apply mask, only collect predictions and targets for valid time steps
                batch_size = predictions.size(0)
                max_pred_t = predictions.size(1)
                max_mask_t = mask.size(1)
                max_target_t = targets.size(1)
                
                # Determine minimum common time length for collection
                collect_t = min(max_pred_t, max_mask_t, max_target_t)
                
                mask_collect = mask[:, :collect_t].bool()
                
                for b in range(batch_size):
                    valid_indices = mask_collect[b].nonzero(as_tuple=True)[0]
                    if len(valid_indices) > 0:
                        b_pred = predictions[b, valid_indices, 0].cpu().numpy()
                        b_target = targets[b, valid_indices].cpu().numpy()
                        all_predictions.append(b_pred)
                        all_targets.append(b_target)
        
        # Calculate average loss
        epoch_loss /= num_batches
        self.history['valid_loss'].append(epoch_loss)
        
        # Record validation loss to tensorboard
        self.writer.add_scalar('Loss/valid', epoch_loss, self.current_epoch)
        
        # Merge all predictions and targets
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # Calculate evaluation metrics
        metrics = compute_metrics(all_targets, all_predictions)
        self.history['valid_metrics'].append(metrics)
        
        # Record metrics to tensorboard
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'Metrics/{metric_name}', value, self.current_epoch)
        
        # Print main metrics
        logger.info(f"Validation - Loss: {epoch_loss:.4f}")
        logger.info(f"Avg metrics - AUC: {metrics.get('auc', 0):.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}, F1: {metrics.get('f1', 0):.4f}")
        
        # Early stopping based on primary_metric
        # Use self.primary_metric to get correct metric value
        default_value = 0.0 if self.metric_mode == 'max' else float('inf')
    def train(self, resume_from=None):
        """Training main loop"""
        logger.info(f"Start training, total {self.max_epochs} epochs...")
        
        # Save original initial best_score (if set)
        original_best_score = self.best_score
        
        if resume_from is not None:
            if os.path.exists(resume_from):
                self.start_epoch, loaded_best_score = load_checkpoint(resume_from, self.model, self.optimizer)
                logger.info(f"Resume from checkpoint: {resume_from}, starting epoch: {self.start_epoch}, loaded best score: {loaded_best_score:.4f}")
                # Only use loaded score when initial best score is not set
                if self.initial_best_score is None:
                    self.best_score = loaded_best_score
                else:
                    # Keep initially set best_score
                    logger.info(f"Using initially set best score: {original_best_score:.4f} instead of loaded score: {loaded_best_score:.4f}")
            else:
                logger.warning(f"Specified resume checkpoint does not exist: {resume_from}, start training from scratch.")
        elif self.initial_checkpoint_path is not None and os.path.exists(self.initial_checkpoint_path):
            self.start_epoch, loaded_best_score = load_checkpoint(self.initial_checkpoint_path, self.model, self.optimizer)
            logger.info(f"Load from initial checkpoint: {self.initial_checkpoint_path}, starting epoch: {self.start_epoch}, loaded best score: {loaded_best_score:.4f}")
            # Only use loaded score when initial best score is not set
            if self.initial_best_score is None:
                self.best_score = loaded_best_score
            else:
                # Keep initially set best_score
                logger.info(f"Using initially set best score: {original_best_score:.4f} instead of loaded score: {loaded_best_score:.4f}")
        
        # Synchronize initial best score to early_stopping
        if hasattr(self, 'early_stopping') and hasattr(self, 'best_score'):
            self.early_stopping.best_score = self.best_score
            logger.info(f"Initial best score: {self.best_score:.4f} (based on {self.primary_metric}, mode: {self.metric_mode})")
        
        for epoch in range(self.start_epoch, self.max_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - start_time
            
            valid_loss, primary_metric_val = self.validate() # Receive returned metric value
            
            if self.scheduler is not None:
                if self.scheduler_is_plateau:
                    if primary_metric_val is not None:
                        self.scheduler.step(primary_metric_val) # Use metric value
                else:
                    self.scheduler.step()
            
            # If no validation loader, only save by interval
            if self.valid_loader is None:
                logger.info("No validation set, skip validation and early stopping.")
                if (epoch + 1) % self.checkpoint_interval == 0:
                    logger.info(f"Save checkpoint by interval Epoch {epoch+1}...")
                    self._save_checkpoint(epoch, 0.0, False)
            # If validation loader exists but metrics cannot be calculated (e.g., too few validation samples)
            elif primary_metric_val is None:
                logger.warning(f"Cannot calculate validation metrics ({self.primary_metric} is None).")
                if (epoch + 1) % self.checkpoint_interval == 0:
                    logger.info(f"Save checkpoint by interval Epoch {epoch+1}...")
                    self._save_checkpoint(epoch, self.best_score, False)
            # Normal validation process
            else:
                # Need to get complete metrics dict to pass to early_stopping
                # validate method now returns (loss, primary_metric_value)
                # metrics stored in self.history['valid_metrics'][-1]
                current_metrics = self.history['valid_metrics'][-1] if self.history['valid_metrics'] else None
                
                if current_metrics is not None:
                    is_best = self.early_stopping(primary_metric_val, current_metrics) # Pass metric value and complete dict
                    if is_best or (epoch + 1) % self.checkpoint_interval == 0:
                        self._save_checkpoint(epoch, primary_metric_val, is_best) # Save current metric value
                    if self.early_stopping.early_stop:
                        logger.info(f"Early stop at epoch {epoch+1}")
                        break
                else:
                     logger.warning(f"Epoch {epoch+1}: Cannot get current validation metrics, skip early stopping check and best model saving.")
                     # Still save by interval
                     if (epoch + 1) % self.checkpoint_interval == 0:
                         logger.info(f"Save checkpoint by interval Epoch {epoch+1}...")
                         self._save_checkpoint(epoch, self.best_score, False) # Use previous best_score
        
        self._save_history()
        self.writer.close()
        return self.history
    
    def test(self, test_loader, checkpoint_path=None):
        """
        测试模型
        
        参数:
            test_loader (DataLoader): 测试数据加载器
            checkpoint_path (str, optional): 模型检查点路径，'best' 或具体路径
        
        返回:
            dict: 测试指标
        """
        if checkpoint_path is not None:
            if checkpoint_path.lower() == 'best':
                load_path = os.path.join(self.checkpoints_dir, 'best_model.pth')
                logger.info("加载最佳检查点进行测试...")
            else:
                load_path = checkpoint_path
            if os.path.exists(load_path):
                _, _ = load_checkpoint(load_path, self.model)
            else:
                logger.error(f"测试检查点未找到: {load_path}。将使用当前模型状态。")
        else:
            logger.warning("未指定测试检查点，将使用当前模型状态。")

        self.model.eval()
        all_predictions = []
        all_targets = []
        patient_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                modality_inputs = batch['modality_inputs'].to(self.device)
                segmentation = batch['segmentation'].to(self.device)
                targets = batch['target'].to(self.device)
                mask = batch.get('mask').to(self.device)
                batch_patient_ids = batch.get('patient_id')
                if batch_patient_ids:
                    patient_ids.extend(batch_patient_ids)
                
                outputs = self.model(modality_inputs, segmentation, targets=targets, mask=mask)
                
                predictions = torch.sigmoid(outputs['predictions'])
                
                # 应用掩码
                batch_size = predictions.size(0)
                max_pred_t = predictions.size(1)
                max_mask_t = mask.size(1)
                max_target_t = targets.size(1)
                collect_t = min(max_pred_t, max_mask_t, max_target_t)
                mask_collect = mask[:, :collect_t].bool()
                
                for b in range(batch_size):
                    valid_indices = mask_collect[b].nonzero(as_tuple=True)[0]
                    if len(valid_indices) > 0:
                        b_pred = predictions[b, valid_indices, 0].cpu().numpy()
                        b_target = targets[b, valid_indices].cpu().numpy()
                        all_predictions.append(b_pred)
                        all_targets.append(b_target)
        
        # 计算指标
        if not all_predictions:
            logger.error("测试集中没有有效的预测或目标，无法计算指标。")
            return {}
        
        try:
            all_predictions_flat = np.concatenate(all_predictions)
            all_targets_flat = np.concatenate(all_targets).astype(int)
            
            if len(all_predictions_flat) == 0 or len(np.unique(all_targets_flat)) < 2:
                logger.warning(f"测试集中有效样本不足 ({len(all_predictions_flat)}) 或标签类别单一，指标可能无意义。")
                metrics = {'auc': 0.0, 'accuracy': 0.0, 'f1': 0.0}
            else:
                metrics = compute_metrics(all_targets_flat, all_predictions_flat)
        except Exception as e:
            logger.error(f"计算测试指标时出错: {e}")
            metrics = {'auc': 0.0, 'accuracy': 0.0, 'f1': 0.0}

        # 创建测试结果目录
        test_results_dir = os.path.join(self.results_dir, 'test_results')
        os.makedirs(test_results_dir, exist_ok=True)
        
        # 保存测试指标
        metrics_path = os.path.join(test_results_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 保存预测结果
        results = {
            'predictions': all_predictions_flat.tolist() if 'all_predictions_flat' in locals() else [],
            'targets': all_targets_flat.tolist() if 'all_targets_flat' in locals() else []
        }
        
        if patient_ids:
            results['patient_ids'] = patient_ids
        
        predictions_path = os.path.join(test_results_dir, 'test_predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info("测试结果:")
        logger.info(f"Metrics - AUC: {metrics.get('auc', 0):.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}, F1: {metrics.get('f1', 0):.4f}")
        
        return metrics
    
    def evaluate(self, data_loader, checkpoint_path=None, return_details=False):
        """
        评估模型并返回预测、标签和患者 ID。

        参数:
            data_loader (DataLoader): 数据加载器
            checkpoint_path (str, optional): 模型检查点路径，'best' 或具体路径
            return_details (bool): 是否返回预测、标签和 ID 列表

        返回:
            tuple: (loss, metrics) 或 (loss, metrics, predictions, targets, patient_ids) 如果 return_details=True
        """
        if checkpoint_path is not None:
            if checkpoint_path.lower() == 'best':
                load_path = os.path.join(self.checkpoints_dir, 'best_model.pth')
                logger.info("加载最佳检查点进行评估...")
            else:
                load_path = checkpoint_path
            if os.path.exists(load_path):
                # 仅加载模型状态，不加载优化器或 epoch 信息
                checkpoint = torch.load(load_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"成功从 {load_path} 加载模型状态。")
                else:
                    logger.warning(f"检查点 {load_path} 中未找到 'model_state_dict'。")
            else:
                logger.error(f"评估检查点未找到: {load_path}。将使用当前模型状态。")
        else:
            logger.info("未指定评估检查点，将使用当前模型状态。")

        self.model.eval()
        epoch_loss = 0.0
        num_batches = len(data_loader)
        all_predictions = []
        all_targets = []
        all_patient_ids = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                modality_inputs = batch['modality_inputs'].to(self.device)
                segmentation = batch['segmentation'].to(self.device)
                targets = batch['target'].to(self.device)
                mask = batch.get('mask').to(self.device)
                batch_patient_ids = batch.get('patient_id')

                outputs = self.model(modality_inputs, segmentation, targets=targets, mask=mask)
                loss, _ = self.model.calculate_loss(outputs, targets, mask)
                epoch_loss += loss.item()

                predictions = torch.sigmoid(outputs['predictions'])

                batch_size = predictions.size(0)
                max_pred_t = predictions.size(1)
                max_mask_t = mask.size(1) if mask is not None else 0
                max_target_t = targets.size(1)
                collect_t = min(max_pred_t, max_mask_t if max_mask_t > 0 else max_pred_t, max_target_t)
                mask_collect = mask[:, :collect_t].bool() if mask is not None else torch.ones(batch_size, collect_t, dtype=torch.bool, device=self.device)

                for b in range(batch_size):
                    valid_indices = mask_collect[b].nonzero(as_tuple=True)[0]
                    if len(valid_indices) > 0:
                        b_pred = predictions[b, valid_indices, 0].cpu().numpy()
                        b_target = targets[b, valid_indices].cpu().numpy()
                        all_predictions.append(b_pred)
                        all_targets.append(b_target)
                        # 记录对应的 patient_id
                        if batch_patient_ids and b < len(batch_patient_ids):
                            # 每个有效预测对应一个 patient_id
                            all_patient_ids.extend([batch_patient_ids[b]] * len(b_pred))
                        else:
                            all_patient_ids.extend([f"unknown_patient_{b}"] * len(b_pred))

        epoch_loss /= num_batches
        
        # 合并并计算指标
        if not all_predictions:
            logger.warning("评估集中没有有效的预测或目标，无法计算指标。")
            metrics = {'auc': 0.0, 'accuracy': 0.0, 'f1': 0.0}
            all_predictions_flat = np.array([])
            all_targets_flat = np.array([])
        else:
            try:
                all_predictions_flat = np.concatenate(all_predictions)
                all_targets_flat = np.concatenate(all_targets).astype(int)
                if len(all_predictions_flat) == 0 or len(np.unique(all_targets_flat)) < 2:
                    logger.warning(f"评估集中有效样本不足 ({len(all_predictions_flat)}) 或标签类别单一，指标可能无意义。")
                    metrics = {'auc': 0.0, 'accuracy': 0.0, 'f1': 0.0}
                else:
                    metrics = compute_metrics(all_targets_flat, all_predictions_flat)
            except Exception as e:
                logger.error(f"计算评估指标时出错: {e}")
                metrics = {'auc': 0.0, 'accuracy': 0.0, 'f1': 0.0}

        logger.info(f"Evaluation - Loss: {epoch_loss:.4f}")
        logger.info(f"Avg metrics - AUC: {metrics.get('auc', 0):.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}, F1: {metrics.get('f1', 0):.4f}")

        if return_details:
            return epoch_loss, metrics, all_predictions_flat, all_targets_flat, all_patient_ids
        else:
            return epoch_loss, metrics
    
    def _save_checkpoint(self, epoch, score, is_best):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score,  # 当前检查点的分数
            'best_score': self.best_score if hasattr(self, 'best_score') else score,  # 记录全局最佳分数
            'history': self.history
        }
        
        save_checkpoint(
            checkpoint, 
            is_best, 
            self.checkpoints_dir, 
            filename=f'checkpoint_epoch_{epoch+1}.pth'
        )
    
    def _save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.results_dir, 'training_history.json')
        
        # 将numpy数组转换为可序列化的形式
        serializable_history = {
            'train_loss': [float(loss) for loss in self.history['train_loss']],
            'valid_loss': [float(loss) if loss is not None else None for loss in self.history['valid_loss']],
            'valid_metrics': self.history['valid_metrics']
        }
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)
        
        logger.info(f"训练历史已保存到: {history_path}")
    
    def get_subregion_scores(self, data_loader, num_samples=5, checkpoint_path=None):
        """
        获取子区域评分可视化结果
        
        参数:
            data_loader (DataLoader): 数据加载器
            num_samples (int): 要可视化的样本数
            checkpoint_path (str, optional): 模型检查点路径，'best' 或具体路径
        
        返回:
            dict: 子区域评分可视化结果字典，包含 scores, segmentations, patient_ids
        """
        if checkpoint_path is not None:
            if checkpoint_path.lower() == 'best':
                load_path = os.path.join(self.checkpoints_dir, 'best_model.pth')
                logger.info("加载最佳检查点进行可视化...")
            else:
                load_path = checkpoint_path
            if os.path.exists(load_path):
                _, _ = load_checkpoint(load_path, self.model)
            else:
                logger.error(f"可视化检查点未找到: {load_path}。将使用当前模型状态。")
        else:
            logger.warning("未指定可视化检查点，将使用当前模型状态。")

        self.model.eval()
        region_scores_list = []
        segmentation_list = []
        patient_ids_vis = []
        count = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if count >= num_samples:
                    break
                
                modality_inputs = batch['modality_inputs'].to(self.device)
                segmentation = batch['segmentation'].to(self.device)
                mask = batch.get('mask').to(self.device)
                batch_patient_ids = batch.get('patient_id')
                
                score_maps = self.model.visualize_subregion_scores(modality_inputs, segmentation, mask)
                
                # 处理批次中的每个样本
                batch_size = score_maps.size(0)
                for b in range(batch_size):
                    if count < num_samples:
                        score_maps_np = score_maps[b].cpu().numpy()
                        segmentation_np = segmentation[b].cpu().numpy()
                        
                        region_scores_list.append(score_maps_np)
                        segmentation_list.append(segmentation_np)
                        if batch_patient_ids and b < len(batch_patient_ids):
                            patient_ids_vis.append(batch_patient_ids[b])
                        else:
                            patient_ids_vis.append(f"sample_{count}")
                        count += 1
                    else:
                        break 
        
        # 创建可视化结果目录
        vis_results_dir = os.path.join(self.results_dir, 'visualizations')
        os.makedirs(vis_results_dir, exist_ok=True)
        for i, (score_maps, segmentation) in enumerate(zip(region_scores_list, segmentation_list)):
            patient_id = patient_ids_vis[i] if i < len(patient_ids_vis) else f"sample_{i}"
            np.save(os.path.join(vis_results_dir, f"{patient_id}_scores.npy"), score_maps)
            np.save(os.path.join(vis_results_dir, f"{patient_id}_segmentation.npy"), segmentation)
            logger.info(f"已保存样本 {patient_id} 的可视化评分和分割。")

        return {
            'scores': region_scores_list,
            'segmentations': segmentation_list,
            'patient_ids': patient_ids_vis
        }
