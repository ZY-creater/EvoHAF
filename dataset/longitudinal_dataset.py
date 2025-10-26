#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Longitudinal Medical Imaging Dataset (longitudinal_dataset.py)

This file contains PyTorch dataset classes for processing longitudinal medical imaging data:
1. Glioma dataset (LongitudinalGliomaDataset)
2. General longitudinal medical imaging dataset base class (LongitudinalMedicalDataset)

Supported features:
- Multi-modal data loading
- Longitudinal time series processing
- Data preprocessing and augmentation
- Lazy loading (deferred evaluation)
- Data visualization
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import random
from pathlib import Path
import sys
# Set matplotlib backend to Agg before importing (non-GUI)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# Add opensource directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
opensource_root = os.path.abspath(os.path.join(current_dir, '..'))
if opensource_root not in sys.path:
    sys.path.insert(0, opensource_root)

from dataset.utils import (
    get_project_root, load_json_data, load_nifti_image, load_segmentation,
    get_patient_modalities, get_modality_images, convert_to_tensor,
    get_valid_time_points, find_bounding_box, crop_volume
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LongitudinalMedicalDataset(Dataset):
    """
    Base class for longitudinal medical imaging datasets
    
    Args:
        json_path (str): Path to dataset JSON file
        modality_list (list): List of required modalities
        segmentation_key (str): Key name for segmentation images
        task_type (str): Task type, 'sequential' or 'cumulative'
        max_time_points (int): Maximum number of time points
        transform (callable, optional): Data augmentation transform
        target_key (str, optional): Key name for target labels
        lazy_loading (bool): Whether to use lazy loading
        mask_region_only (bool): Whether to consider only masked regions
        resample_size (tuple, optional): Target size for resampling (if provided, resampling will be performed)
    """
    def __init__(
            self, 
            json_path, 
            modality_list,
            segmentation_key,
            task_type='sequential',
            max_time_points=8,
            transform=None,
            target_key=None,
            lazy_loading=True,
            mask_region_only=True,
            resample_size=None
        ):
        self.json_path = json_path
        self.modality_list = modality_list
        self.segmentation_key = segmentation_key
        self.task_type = task_type
        self.max_time_points = max_time_points
        self.transform = transform
        self.target_key = target_key
        self.lazy_loading = lazy_loading
        self.mask_region_only = mask_region_only
        self.resample_size = resample_size
        
        # Load JSON data
        self.data = load_json_data(json_path)
        if not self.data:
            logger.error(f"Failed to load dataset JSON file: {json_path}")
            raise FileNotFoundError(f"Dataset file does not exist: {json_path}")
        
        # Get valid patient ID list
        self.valid_patient_ids = self._get_valid_patients()
        logger.info(f"Loaded {len(self.valid_patient_ids)} valid patients")
        
        # If not using lazy loading, preload all data
        if not self.lazy_loading:
            self.preloaded_data = self._preload_data()
    
    def _get_valid_patients(self):
        """Get list of valid patient IDs"""
        valid_patients = []
        skipped_segmentation = 0
        skipped_modalities = 0
        # skipped_target = 0 # Target check is currently commented out
        
        for patient_id, patient_data in self.data.items():
            # Check if segmentation images exist
            if self.segmentation_key not in patient_data:
                logger.debug(f"[ValidCheck] Skipping patient {patient_id}: missing segmentation key '{self.segmentation_key}'")
                skipped_segmentation += 1
                continue
                
            # Check if there are sufficient time points with all specified modalities
            valid_time_points = get_valid_time_points(patient_data, self.modality_list)
            if len(valid_time_points) == 0:
                # Add more detailed logging: which modalities are missing?
                missing_modalities_info = ""
                if isinstance(patient_data.get('scans'), dict):
                    all_available_mods = set()
                    for tp_data in patient_data['scans'].values():
                        all_available_mods.update(tp_data.keys())
                    missing_modalities_info = f" (Available: {list(all_available_mods)}, Required: {self.modality_list})"
                
                logger.debug(f"[ValidCheck] Skipping patient {patient_id}: no time points with all required modalities.{missing_modalities_info}")
                skipped_modalities += 1
                continue
                
            # Check if target labels exist (if target_key is specified)
            # if self.target_key is not None and self.target_key not in patient_data:
            #     logger.debug(f"[ValidCheck] Skipping patient {patient_id}: missing target key '{self.target_key}'")
            #     skipped_target += 1
            #     continue
                
            valid_patients.append(patient_id)
            
        logger.info(f"[ValidCheck Summary] Total patients in JSON: {len(self.data)}. Valid: {len(valid_patients)}. Skipped due to missing segmentation: {skipped_segmentation}. Skipped due to missing modalities: {skipped_modalities}.")
        
        return valid_patients
    
    def _preload_data(self):
        """Preload all data, including calculating bounding boxes and performing cropping"""
        preloaded = {}
        logger.info("Start preloading data...") # Add logging
        num_processed = 0
        total_patients = len(self.valid_patient_ids)

        for patient_id in self.valid_patient_ids:
            try:
                patient_data = self.data[patient_id]
                patient_data['id'] = patient_id # Ensure ID exists

                # 1. Load segmentation images (original for bbox, resampled for subsequent processing)
                segmentation_path = patient_data[self.segmentation_key]
                # Loading original size for bbox calculation doesn't seem appropriate, as bbox should be based on processed images
                # Therefore, load directly processed (possibly resampled) segmentation map
                segmentation = load_segmentation(segmentation_path, resample_size=self.resample_size)
                if segmentation is None:
                    logger.warning(f"Preload skip patient {patient_id}: failed to load segmentation image")
                    continue

                # 2. Calculate bounding box (based on processed segmentation map)
                bbox = find_bounding_box(segmentation, margin=(5, 10, 10)) # Use same margin as _load_patient_data
                # If bbox is None (e.g., all-zero segmentation), continue but subsequent cropping will be skipped
                if bbox is None:
                    logger.warning(f"Preload patient {patient_id}: calculated bbox is empty, cropping will not be performed.")

                # 3. Get valid time points and modality paths
                valid_time_points = get_valid_time_points(patient_data, self.modality_list)
                if not valid_time_points:
                    logger.warning(f"Preload skip patient {patient_id}: no valid time points with modalities")
                    continue
                modality_paths = get_patient_modalities(patient_data, self.modality_list, valid_time_points)

                # 4. Load modality images (load resampled images)
                modality_images = get_modality_images(patient_data, modality_paths, resample_size=self.resample_size)
                if not modality_images: # Check if any modalities were successfully loaded
                     logger.warning(f"Preload skip patient {patient_id}: failed to load any modality images")
                     continue

                # 5. Perform cropping (if bbox is valid)
                if bbox is not None:
                    segmentation_cropped = crop_volume(segmentation, bbox)
                    modality_images_cropped = {}
                    for tp_str, mods in modality_images.items():
                        modality_images_cropped[tp_str] = {}
                        for mod_key, img_array in mods.items():
                            modality_images_cropped[tp_str][mod_key] = crop_volume(img_array, bbox)
                else:
                    segmentation_cropped = segmentation # Not cropped
                    modality_images_cropped = modality_images # Not cropped

                # 6. Load target labels
                target = None
                if self.target_key is not None and self.target_key in patient_data:
                    target = patient_data[self.target_key]

                # 7. Store preloaded and preprocessed data
                preloaded[patient_id] = {
                    'segmentation_cropped': segmentation_cropped, # Store cropped
                    'modality_images_cropped': modality_images_cropped, # Store cropped
                    'target': target,
                    'valid_time_points': valid_time_points,
                    'bbox': bbox # Store calculated bbox
                }
                num_processed += 1
                if num_processed % 50 == 0: # Print progress every 50 processed
                     logger.info(f"Preload progress: {num_processed}/{total_patients}")

            except Exception as e:
                 logger.error(f"Error preloading patient {patient_id}: {str(e)}")
                 import traceback
                 traceback.print_exc()
                 # Choose to skip this patient or interrupt, here choose to skip
                 continue
        
        logger.info(f"Preload completed. Successfully processed {len(preloaded)}/{total_patients} patients.")
        return preloaded
    
    def __len__(self):
        """Return dataset size"""
        return len(self.valid_patient_ids)
        
    def get_patient_id(self, idx):
        """
        Get patient ID by index
        
        Args:
            idx (int): Sample index
            
        Returns:
            str: Patient ID, returns None if index is invalid or error occurs
        """
        try:
            if not isinstance(idx, (int, np.integer)):
                logger.warning(f"get_patient_id: non-integer index {type(idx)}")
                return None
                
            if idx < 0 or idx >= len(self.valid_patient_ids):
                logger.warning(f"get_patient_id: index {idx} out of valid range [0-{len(self.valid_patient_ids)-1}]")
                return None
                
            # Get patient ID
            patient_id = self.valid_patient_ids[idx]
            return patient_id
            
        except Exception as e:
            logger.error(f"Error getting patient ID: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _load_patient_data(self, patient_id):
        """Load single patient's data and calculate bounding box"""
        patient_data = self.data[patient_id]
        patient_data['id'] = patient_id

        # Load segmentation images (do not crop immediately, first get original segmentation for bbox calculation)
        segmentation_path = patient_data[self.segmentation_key]
        segmentation_orig = load_segmentation(segmentation_path, resample_size=None) # Get original size segmentation
        if segmentation_orig is None:
            raise ValueError(f"Failed to load original segmentation image for patient {patient_id}")

        # If resampling is needed, resample segmentation map first
        if self.resample_size:
            segmentation_resampled = load_segmentation(segmentation_path, resample_size=self.resample_size)
            if segmentation_resampled is None:
                 raise ValueError(f"Failed to load or resample segmentation image for patient {patient_id} to {self.resample_size}")
            segmentation = segmentation_resampled
        else:
            segmentation = segmentation_orig # If no resampling, use original segmentation

        # Calculate bounding box based on resampled (or original) segmentation map
        # !! Note: margin needs to be adjusted according to resample_size, or use fixed pixel values !!
        # Temporarily use fixed pixel margin here, which may have different effects at different resolutions
        bbox = find_bounding_box(segmentation, margin=(5, 10, 10))

        # Get valid time points and modality paths
        valid_time_points = get_valid_time_points(patient_data, self.modality_list)
        modality_paths = get_patient_modalities(patient_data, self.modality_list, valid_time_points)

        # Load modality images (load resampled images, but do not crop)
        modality_images = get_modality_images(patient_data, modality_paths, resample_size=self.resample_size)

        # Load target labels
        target = None
        if self.target_key is not None and self.target_key in patient_data:
            target = patient_data[self.target_key]

        return {
            'segmentation': segmentation, # Return resampled (or original) segmentation map
            'modality_images': modality_images, # Return resampled (or original) modality maps
            'target': target,
            'valid_time_points': valid_time_points,
            'patient_data': patient_data,
            'bbox': bbox # Return calculated bounding box
        }
    
    def _prepare_sample(self, patient_data, segmentation_processed, modality_images_processed, valid_time_points, target=None, bbox=None):
        """
        准备胶质瘤样本数据。
        假设输入的 segmentation_processed 和 modality_images_processed 已经是经过重采样和裁剪（如果 applicable）的数据。
        """
        # 确定有效且有标签的时间点
        time_points = sorted(valid_time_points)
        if target is not None and isinstance(target, dict):
            labeled_time_points = {int(tp) for tp in target.keys()}
            time_points = sorted([tp for tp in time_points if tp in labeled_time_points])
        else:
            if not isinstance(target, dict):
                 logger.warning(f"患者 {patient_data.get('id', 'Unknown')} 缺少有效的 target 字典 ('{self.target_key}')，无法生成样本。")
                 return None

        time_points = time_points[:self.max_time_points]
        num_time_points = len(time_points)
        if num_time_points == 0:
            logger.warning(f"患者 {patient_data.get('id', 'Unknown')} 没有有效的、带标签的时间点。")
            return None

        # --- 裁剪逻辑已移至 _preload_data (for non-lazy) 或 _load_patient_data (for lazy) ---
        # segmentation_cropped = segmentation_processed
        # modality_images_cropped = modality_images_processed

        # 获取处理后图像的形状以初始化正确的张量
        img_shape_processed = None
        # 从处理后的分割图获取形状 (更可靠，因为它总是存在)
        if segmentation_processed is not None:
             img_shape_processed = segmentation_processed.shape # (D, H, W)
        # 如果分割图异常，尝试从模态图获取
        elif modality_images_processed:
             for tp_str, mods in modality_images_processed.items():
                  for mod_key, img_array in mods.items():
                       img_shape_processed = img_array.shape # (D, H, W)
                       if img_shape_processed: break
                  if img_shape_processed: break

        if img_shape_processed is None:
             logger.warning(f"患者 {patient_data.get('id', 'Unknown')} 无法确定有效的处理后图像形状。")
             return None

        # 创建模态输入张量：[T, C, D_proc, H_proc, W_proc]
        modality_tensor = torch.zeros((num_time_points, len(self.modality_list), *img_shape_processed))

        # 填充处理后的模态数据
        for t_idx, tp in enumerate(time_points):
            tp_str = str(tp)
            if tp_str in modality_images_processed:
                for m_idx, modality in enumerate(self.modality_list):
                    if modality in modality_images_processed[tp_str]:
                        img_data_processed = modality_images_processed[tp_str][modality]
                        # 检查形状是否匹配 (理论上应该匹配，除非预处理/裁剪逻辑有误)
                        if img_data_processed.shape == img_shape_processed:
                             modality_tensor[t_idx, m_idx] = convert_to_tensor(img_data_processed)
                        else:
                             logger.warning(f"患者 {patient_data.get('id', 'Unknown')}, TP {tp_str}, Mod {modality}: 形状不匹配 ({img_data_processed.shape} vs {img_shape_processed})，跳过填充。")

        # 创建处理后的分割掩膜张量 [1, D_proc, H_proc, W_proc]
        segmentation_tensor = convert_to_tensor(segmentation_processed).unsqueeze(0)

        # --- 目标标签和Mask的创建保持不变 ---
        progression_labels = []
        valid_steps_mask = torch.zeros(self.max_time_points, dtype=torch.float32)
        for tp_idx, tp in enumerate(time_points):
            tp_str = str(tp)
            label_text = target.get(tp_str, 'stable') # 默认标签
            if isinstance(label_text, str):
                prog_label = self.label_mapping.get(label_text, 0.0)
                if label_text not in self.label_mapping:
                     logger.warning(f"未知标签: {label_text} ... 使用默认 0.0")
            else:
                prog_label = float(label_text)
            progression_labels.append(prog_label)
            if tp_idx < self.max_time_points:
                 valid_steps_mask[tp_idx] = 1.0
        while len(progression_labels) < self.max_time_points:
            progression_labels.append(0.0)
        target_tensor = torch.tensor(progression_labels, dtype=torch.float32)

        # --- 创建最终样本字典 ---
        sample = {
            'modality_inputs': modality_tensor.unsqueeze(0),  # [1, T, C, D_proc, H_proc, W_proc]
            'segmentation': segmentation_tensor, # [1, D_proc, H_proc, W_proc]
            'target': target_tensor,
            'mask': valid_steps_mask,
            'patient_id': patient_data.get('id', 'Unknown')
        }

        return sample
    
    def __getitem__(self, idx):
        """Get sample from dataset"""
        patient_id = self.valid_patient_ids[idx]
        
        # If using lazy loading, load data in real-time
        if self.lazy_loading:
            try:
                patient_data = self._load_patient_data(patient_id)
                
                # Prepare sample
                sample = self._prepare_sample(
                    patient_data['patient_data'],
                    patient_data['segmentation'],
                    patient_data['modality_images'],
                    patient_data['valid_time_points'],
                    patient_data['target'],
                    patient_data['bbox']
                )
                
                # Add patient ID
                sample['patient_id'] = patient_id
                
                return sample
            except Exception as e:
                logger.error(f"Error loading patient {patient_id} data: {str(e)}")
                # If error occurs, return another sample
                if len(self.valid_patient_ids) > 1:
                    return self.__getitem__((idx + 1) % len(self.valid_patient_ids))
                else:
                    raise e
        else:
            # Use preloaded data
            if patient_id not in self.preloaded_data:
                logger.error(f"Patient {patient_id} data not in preloaded data")
                # If error occurs, return another sample
                if len(self.valid_patient_ids) > 1:
                    return self.__getitem__((idx + 1) % len(self.valid_patient_ids))
                else:
                    raise ValueError(f"Patient {patient_id} data not in preloaded data")
            
            preloaded = self.preloaded_data[patient_id]
            
            # Prepare sample - use preloaded and pre-cropped data
            # Note: now passing already cropped images
            sample = self._prepare_sample(
                self.data[patient_id], # patient_data still needed for getting ID and other metadata
                preloaded['segmentation_cropped'], # Use cropped segmentation
                preloaded['modality_images_cropped'], # Use cropped modalities
                preloaded['valid_time_points'],
                preloaded['target'],
                preloaded['bbox'] # Pass bbox
            )
            
            # Add patient ID
            sample['patient_id'] = patient_id
            
            return sample
    
    def visualize_sample(self, idx, slice_idx=None, save_path=None):
        """
        Visualize sample from dataset
        
        Args:
            idx (int): Sample index
            slice_idx (int, optional): Slice index to display, if None auto-select center slice
            save_path (str, optional): Save path, if None use default path
        """
        # Get sample
        sample = self.__getitem__(idx)
        patient_id = sample['patient_id']
        
        # Get modality inputs and segmentation
        modality_inputs = sample['modality_inputs'][0]  # [T, C, D, H, W]
        segmentation = sample['segmentation'][0]  # [D, H, W]
        
        # Get 3D volume shape
        T, C, D, H, W = modality_inputs.shape
        
        # If slice index not specified, select center slice
        if slice_idx is None:
            # Find indices of non-zero values in segmentation mask
            non_zero_slices = torch.nonzero(segmentation.sum(dim=(1, 2)) > 0).squeeze().cpu().numpy()
            if len(non_zero_slices) > 0:
                # Select center of non-zero slices
                slice_idx = non_zero_slices[len(non_zero_slices) // 2]
            else:
                # If all slices are zero, select center slice
                slice_idx = D // 2
        
        # Set visualization parameters
        fig_rows = T
        fig_cols = C + 1  # C modalities + 1 segmentation
        
        # Create red semi-transparent colormap for segmentation mask
        alpha_red_cmap = LinearSegmentedColormap.from_list(
            'alpha_red', [(0, 0, 0, 0), (1, 0, 0, 0.5)])
        
        # Set matplotlib font to global default font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Bitstream Vera Sans']
        
        # Create figure
        plt.figure(figsize=(fig_cols * 4, fig_rows * 4))
        
        for t in range(T):
            for c in range(C):
                # Plot modality images
                plt.subplot(fig_rows, fig_cols, t * fig_cols + c + 1)
                plt.imshow(modality_inputs[t, c, slice_idx].cpu().numpy(), cmap='gray')
                plt.title(f"Timepoint {t}, {self.modality_list[c]}")
                plt.axis('off')
                
                # Overlay segmentation mask on same subplot
                plt.imshow(segmentation[slice_idx].cpu().numpy(), 
                          cmap=alpha_red_cmap, alpha=0.5)
            
            # Plot segmentation mask separately
            plt.subplot(fig_rows, fig_cols, t * fig_cols + C + 1)
            plt.imshow(segmentation[slice_idx].cpu().numpy(), cmap='hot')
            plt.title(f"Segmentation {self.segmentation_key}")
            plt.axis('off')
        
        plt.suptitle(f"Patient {patient_id} - Slice {slice_idx}/{D-1}")
        plt.tight_layout()
        
        # If save path not specified, create default path
        if save_path is None:
            # Ensure output directory exists
            output_dir = os.path.join(get_project_root(), "output", "visualizations")
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"patient_{patient_id}_slice_{slice_idx}.png")
        
        # Save image
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Visualization saved to: {save_path}")
        return save_path


class LongitudinalGliomaDataset(LongitudinalMedicalDataset):
    """
    Longitudinal glioma dataset

    Args:
        json_path (str): Path to dataset JSON file
        modality_list (list): List of required modalities, default ['T1CE', 'FLAIR']
        segmentation_key (str): Key name for segmentation images
        target_key (str): Key name for target labels (usually points to dict containing time point labels, like 'responses')
        max_time_points (int): Maximum number of time points
        transform (callable, optional): Data augmentation transform
        lazy_loading (bool): Whether to use lazy loading
        mask_region_only (bool): Whether to consider only masked regions
        resample_size (tuple, optional): Target size for resampling
    """
    def __init__(
            self,
            json_path,
            modality_list=['T1CE', 'FLAIR'],
            segmentation_key='seg_ctv_c0.05_n24', # Assume this is unified for both datasets or needs configuration
            target_key='responses', # Use this key to get label dictionary
            max_time_points=8,
            transform=None,
            lazy_loading=True,
            mask_region_only=True, # Keep just in case, but check if actually used
            resample_size=None
        ):
        super(LongitudinalGliomaDataset, self).__init__(
            json_path,
            modality_list,
            segmentation_key,
            task_type='sequential', # Fixed to sequential
            max_time_points=max_time_points,
            transform=transform,
            target_key=target_key, # Pass target_key to base class
            lazy_loading=lazy_loading,
            mask_region_only=mask_region_only,
            resample_size=resample_size
        )

        # Label mapping (data has been unified to binary classification in preprocessing stage)
        self.label_mapping = {
            'progression': 1.0,
            'non-progression': 0.0,
        }

    def _prepare_sample(self, patient_data, segmentation_processed, modality_images_processed, valid_time_points, target=None, bbox=None):
        """
        Prepare glioma sample data.
        Assume input segmentation_processed and modality_images_processed are already resampled and cropped (if applicable) data.
        """
        # Determine valid and labeled time points
        time_points = sorted(valid_time_points)
        if target is not None and isinstance(target, dict):
            labeled_time_points = {int(tp) for tp in target.keys()}
            time_points = sorted([tp for tp in time_points if tp in labeled_time_points])
        else:
            if not isinstance(target, dict):
                 logger.warning(f"Patient {patient_data.get('id', 'Unknown')} missing valid target dict ('{self.target_key}'), cannot generate sample.")
                 return None

        time_points = time_points[:self.max_time_points]
        num_time_points = len(time_points)
        if num_time_points == 0:
            logger.warning(f"Patient {patient_data.get('id', 'Unknown')} has no valid, labeled time points.")
            return None

        # --- Cropping logic moved to _preload_data (for non-lazy) or _load_patient_data (for lazy) ---
        # segmentation_cropped = segmentation_processed
        # modality_images_cropped = modality_images_processed

        # Get shape of processed images to initialize correct tensors
        img_shape_processed = None
        # Get shape from processed segmentation map (more reliable as it always exists)
        if segmentation_processed is not None:
             img_shape_processed = segmentation_processed.shape # (D, H, W)
        # If segmentation map is abnormal, try to get from modality maps
        elif modality_images_processed:
             for tp_str, mods in modality_images_processed.items():
                  for mod_key, img_array in mods.items():
                       img_shape_processed = img_array.shape # (D, H, W)
                       if img_shape_processed: break
                  if img_shape_processed: break

        if img_shape_processed is None:
             logger.warning(f"Patient {patient_data.get('id', 'Unknown')} cannot determine valid processed image shape.")
             return None

        # Create modality input tensor: [T, C, D_proc, H_proc, W_proc]
        modality_tensor = torch.zeros((num_time_points, len(self.modality_list), *img_shape_processed))

        # Fill processed modality data
        for t_idx, tp in enumerate(time_points):
            tp_str = str(tp)
            if tp_str in modality_images_processed:
                for m_idx, modality in enumerate(self.modality_list):
                    if modality in modality_images_processed[tp_str]:
                        img_data_processed = modality_images_processed[tp_str][modality]
                        # Check if shape matches (should theoretically match, unless preprocessing/cropping logic is wrong)
                        if img_data_processed.shape == img_shape_processed:
                             modality_tensor[t_idx, m_idx] = convert_to_tensor(img_data_processed)
                        else:
                             logger.warning(f"Patient {patient_data.get('id', 'Unknown')}, TP {tp_str}, Mod {modality}: shape mismatch ({img_data_processed.shape} vs {img_shape_processed}), skip filling.")

        # Create processed segmentation mask tensor [1, D_proc, H_proc, W_proc]
        segmentation_tensor = convert_to_tensor(segmentation_processed).unsqueeze(0)

        # --- Target labels and mask creation remain unchanged ---
        progression_labels = []
        valid_steps_mask = torch.zeros(self.max_time_points, dtype=torch.float32)
        for tp_idx, tp in enumerate(time_points):
            tp_str = str(tp)
            label_text = target.get(tp_str, 'stable') # Default label
            if isinstance(label_text, str):
                prog_label = self.label_mapping.get(label_text, 0.0)
                if label_text not in self.label_mapping:
                     logger.warning(f"Unknown label: {label_text} ... use default 0.0")
            else:
                prog_label = float(label_text)
            progression_labels.append(prog_label)
            if tp_idx < self.max_time_points:
                 valid_steps_mask[tp_idx] = 1.0
        while len(progression_labels) < self.max_time_points:
            progression_labels.append(0.0)
        target_tensor = torch.tensor(progression_labels, dtype=torch.float32)

        # --- Create final sample dictionary ---
        sample = {
            'modality_inputs': modality_tensor.unsqueeze(0),  # [1, T, C, D_proc, H_proc, W_proc]
            'segmentation': segmentation_tensor, # [1, D_proc, H_proc, W_proc]
            'target': target_tensor,
            'mask': valid_steps_mask,
            'patient_id': patient_data.get('id', 'Unknown')
        }

        return sample


def create_dataset(dataset_type, dataset_config):
    """
    Create dataset instance
    
    Args:
        dataset_type (str): Dataset type, 'glioma'
        dataset_config (dict): Dataset configuration
    
    Returns:
        Dataset: Dataset instance
    """
    if dataset_type.lower() == 'glioma':
        return LongitudinalGliomaDataset(**dataset_config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_data_loaders(dataset_configs, batch_size=4, num_workers=4):
    """
    Create data loaders
    
    Args:
        dataset_configs (dict): Dataset configuration dict, format {'train': {'type': 'glioma', 'config': {...}}, ...}
        batch_size (int): Batch size
        num_workers (int): Number of data loading threads
    
    Returns:
        dict: Data loader dictionary
    """
    from torch.utils.data import DataLoader
    from dataset.utils import collate_longitudinal_batch
    
    loaders = {}
    
    for split, config in dataset_configs.items():
        dataset = create_dataset(config['type'], config['config'])
        
        # Set up DataLoader
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=collate_longitudinal_batch,
            pin_memory=True
        )
    
    return loaders

# --- Add specialized Dataset for get_all_vision.py ---
class VisualizationGliomaDataset(LongitudinalGliomaDataset):
    """
    Specialized glioma longitudinal dataset for get_all_vision.py visualization.
    Inherits from LongitudinalGliomaDataset, but __getitem__ adds 'original_paths' to the returned dictionary.
    Forces lazy_loading=False.
    """
    def __init__(self, json_path, **kwargs):
        # Force lazy_loading=False to preload all data
        kwargs['lazy_loading'] = False
        # Remove parameters no longer needed (if exists)
        kwargs.pop('return_original_info', None) 

        super().__init__(json_path=json_path, **kwargs)
        logger.info(f"VisualizationGliomaDataset initialized from {json_path}. Preloaded {len(self.preloaded_data)} patients.")

    def __getitem__(self, idx):
        """
        Get sample, return a dictionary containing tensor data and original path information.
        """
        patient_id = self.valid_patient_ids[idx]

        # --- Get processed tensors from preloaded data ---
        if patient_id not in self.preloaded_data:
             logger.error(f"[VisDataset] Patient {patient_id} not found in preloaded data.")
             return self.__getitem__((idx + 1) % len(self)) # Simple fallback

        preloaded = self.preloaded_data[patient_id]
        patient_metadata = self.data[patient_id] # Get original JSON entry

        # --- Prepare Tensor data dictionary ---
        try:
            # Call base class _prepare_sample to get dictionary containing tensor data
            tensor_sample = super()._prepare_sample(
                patient_metadata,
                preloaded['segmentation_cropped'],
                preloaded['modality_images_cropped'],
                preloaded['valid_time_points'],
                preloaded['target'],
                preloaded['bbox']
            )
            if tensor_sample is None:
                logger.error(f"[VisDataset] _prepare_sample returned None for patient {patient_id}.")
                if len(self) > 1:
                    return self.__getitem__((idx + 1) % len(self))
                else:
                    raise ValueError(f"Cannot prepare sample for only patient {patient_id}")

        except Exception as e:
            logger.error(f"[VisDataset] Error calling super()._prepare_sample for {patient_id}: {e}", exc_info=True)
            if len(self) > 1:
                 return self.__getitem__((idx + 1) % len(self))
            else:
                 raise e

        # --- Add Original Paths information to dictionary ---
        original_paths = {}
        # 1. Get original segmentation path
        original_paths['seg'] = patient_metadata.get(self.segmentation_key)
        if not original_paths['seg']:
            logger.warning(f"[VisDataset] Missing original segmentation path for key '{self.segmentation_key}' in patient {patient_id}")

        # 2. Get original T1CE paths
        t1ce_paths_dict = {}
        timepoint_keys = [k for k in patient_metadata.keys() if k.isdigit()]
        for tp_key in sorted(timepoint_keys, key=int):
            tp_int = int(tp_key)
            if isinstance(patient_metadata[tp_key], dict) and 'T1CE' in patient_metadata[tp_key]:
                t1ce_paths_dict[tp_int] = patient_metadata[tp_key]['T1CE']
        if t1ce_paths_dict:
             max_tp_index = max(t1ce_paths_dict.keys())
             original_paths['t1ce'] = [t1ce_paths_dict.get(i) for i in range(max_tp_index + 1)]
        else:
             original_paths['t1ce'] = []
        if not original_paths['t1ce']:
             logger.warning(f"[VisDataset] No T1CE paths found for patient {patient_id}")
        
        # Add original_paths to tensor_sample dictionary
        tensor_sample['original_paths'] = original_paths
        
        # Ensure patient_id is also present (usually _prepare_sample will add it)
        if 'patient_id' not in tensor_sample:
            tensor_sample['patient_id'] = patient_id

        # Return merged dictionary
        return tensor_sample
