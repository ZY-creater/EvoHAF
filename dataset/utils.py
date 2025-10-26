#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset Utility Functions (utils.py)

This file contains utility functions for processing longitudinal medical imaging datasets:
1. Data loading functions
2. Image reading and preprocessing
3. Segmentation image processing
4. Data format conversion

Supports multi-modal glioma imaging data (e.g., T1CE, FLAIR)
"""

import os
import json
import numpy as np
import SimpleITK as sitk
import torch
import logging
from pathlib import Path
import torch.nn.functional as F
import platform

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    """Get project root directory (opensource directory)"""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__))))

def load_json_data(file_path):
    """
    Load JSON data
    
    Args:
        file_path (str): JSON file path
    
    Returns:
        dict: Loaded JSON data
    """
    try:
        # Check if absolute path
        if os.path.isabs(file_path):
            full_path = file_path
        else:
            # If relative path, make it relative to project root
            full_path = os.path.join(get_project_root(), file_path)
            
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON data: {full_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to read JSON file: {str(e)}")
        return None

def save_json_data(data, file_path):
    """
    Save data as JSON file
    
    Args:
        data (dict): Data to save
        file_path (str): Output file path
    """
    try:
        # Check if absolute path
        if os.path.isabs(file_path):
            full_path = file_path
        else:
            # If relative path, make it relative to project root
            full_path = os.path.join(get_project_root(), file_path)
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved data to: {full_path}")
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")

def format_path_for_os(path_str):
    """Format path separators for the current operating system."""
    if not path_str or not isinstance(path_str, str):
        return path_str
    if platform.system() == "Windows":
        return path_str.replace('/', '\\')
    else: # Linux or macOS
        return path_str.replace('\\', '/')

def load_nifti_image(file_path, normalize=True, resample_size=None):
    """
    Load NIfTI format image and perform preprocessing
    
    Args:
        file_path (str): Image file path (from JSON)
        normalize (bool): Whether to perform normalization
        resample_size (tuple, optional): Target size for resampling, e.g., (128, 128, 128)
        
    Returns:
        numpy.ndarray: Processed image array
    """
    try:
        # Format path separators
        file_path = format_path_for_os(file_path)
        
        # Check if absolute path
        if os.path.isabs(file_path):
            full_path = file_path
        else:
            # If relative path, make it relative to project root
            full_path = os.path.join(get_project_root(), file_path)
            
        if not os.path.exists(full_path):
            logger.error(f"Image file does not exist: {full_path}")
            return None
            
        # Read image
        image = sitk.ReadImage(full_path)
        
        # Resample if size specified
        if resample_size is not None:
            # Call resampling function
            image = resample_image(image, new_shape=resample_size)
            
        # Convert SimpleITK image to NumPy array
        image_array = sitk.GetArrayFromImage(image)
        
        # Normalize
        if normalize:
            image_array = normalize_image_01(image_array)
        
        return image_array
    except Exception as e:
        logger.error(f"Failed to load image ({file_path}): {str(e)}")
        return None

def normalize_image_01(image_array):
    """
    Normalize image to 0-1 range
    
    Args:
        image_array (numpy.ndarray): Input image array
    
    Returns:
        numpy.ndarray: Normalized image array
    """
    # Handle all-zero or constant images
    if np.max(image_array) == np.min(image_array):
        return np.zeros_like(image_array, dtype=np.float32)
    
    # Linear normalization to [0, 1] range
    normalized = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    return normalized.astype(np.float32)

def resample_image(image, new_shape=(128, 128, 128)):
    """
    Resample image to specified size
    
    Args:
        image (sitk.Image): Input image
        new_shape (tuple): Target size
        
    Returns:
        sitk.Image: Resampled image
    """
    # Set resampling parameters
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_shape)
    
    # Calculate new pixel spacing
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_spacing = [original_spacing[i] * (original_size[i] / new_shape[i]) for i in range(3)]
    resampler.SetOutputSpacing(new_spacing)
    
    # Set other parameters
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    
    # Execute resampling
    resampled_image = resampler.Execute(image)
    
    return resampled_image

def load_segmentation(file_path, resample_size=None):
    """
    Load segmentation image
    
    Args:
        file_path (str): Segmentation file path (from JSON)
        resample_size (tuple, optional): Target size for resampling, e.g., (128, 128, 128)
        
    Returns:
        numpy.ndarray: Segmentation image array
    """
    try:
        # Format path separators
        file_path = format_path_for_os(file_path)
        
        # Check if absolute path
        if os.path.isabs(file_path):
            full_path = file_path
        else:
            # If relative path, make it relative to project root
            full_path = os.path.join(get_project_root(), file_path)
            
        if not os.path.exists(full_path):
            logger.error(f"Segmentation file does not exist: {full_path}")
            return None
            
        # Read segmentation image
        segmentation = sitk.ReadImage(full_path)
        
        # Resample if size specified
        if resample_size is not None:
            # Use nearest neighbor interpolation for segmentation
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(resample_size)
            
            # Calculate new pixel spacing
            original_spacing = segmentation.GetSpacing()
            original_size = segmentation.GetSize()
            new_spacing = [original_spacing[i] * (original_size[i] / resample_size[i]) for i in range(3)]
            resampler.SetOutputSpacing(new_spacing)
            
            resampler.SetOutputOrigin(segmentation.GetOrigin())
            resampler.SetOutputDirection(segmentation.GetDirection())
            # Use nearest neighbor to preserve label values
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            
            segmentation = resampler.Execute(segmentation)
        
        segmentation_array = sitk.GetArrayFromImage(segmentation)
        
        return segmentation_array
    except Exception as e:
        logger.error(f"Failed to load segmentation ({file_path}): {str(e)}")
        return None

def get_patient_modalities(patient_data, modality_list, time_points=None):
    """
    Get patient modality data for specified time points
    
    Args:
        patient_data (dict): Patient data
        modality_list (list): List of required modalities
        time_points (list, optional): List of time points, None means get all
        
    Returns:
        dict: Path dictionary organized by time point and modality
    """
    result = {}
    
    # Get all time points
    available_time_points = sorted([int(tp) for tp in patient_data.keys() if tp.isdigit()])
    
    # Get only specified time points if provided
    if time_points is not None:
        time_points = [str(tp) for tp in time_points]
    else:
        time_points = [str(tp) for tp in available_time_points]
    
    # Get modality data for each time point
    for tp in time_points:
        if tp in patient_data:
            result[tp] = {}
            for modality in modality_list:
                if modality in patient_data[tp]:
                    result[tp][modality] = patient_data[tp][modality]
    
    return result

def get_modality_images(patient_data, modality_paths, normalize=True, resample_size=None):
    """
    Get and load patient modality images
    
    Args:
        patient_data (dict): Patient data
        modality_paths (dict): Path dictionary organized by time point and modality
        normalize (bool): Whether to normalize
        resample_size (tuple, optional): Resampling size
        
    Returns:
        dict: Dictionary of loaded image arrays organized by time point and modality
    """
    images = {}
    for tp, mods in modality_paths.items():
        images[tp] = {}
        for mod, path in mods.items():
            # Call loading function
            img_array = load_nifti_image(path, normalize=normalize, resample_size=resample_size)
            if img_array is not None:
                images[tp][mod] = img_array
            else:
                logger.warning(f"Unable to load modality {mod} for time point {tp}: {path}")
    return images

def convert_to_tensor(image_array):
    """
    Convert NumPy array to PyTorch tensor
    
    Args:
        image_array (numpy.ndarray): Input image array
        
    Returns:
        torch.Tensor: Converted tensor
    """
    # Ensure data type is float32
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    
    # Convert to tensor
    tensor = torch.from_numpy(image_array)
    
    return tensor

def collate_longitudinal_batch(batch):
    """
    Custom batch collation function for longitudinal data.
    - Pads samples with different spatial sizes after cropping to unify spatial dimensions.
    - Pads sequences with different time lengths.
    - Handles other fields (segmentation, target, mask).

    Args:
        batch (list): List of batch data, each element is a dict containing 'modality_inputs', 'segmentation', 'target', 'mask', 'patient_id'.
                      'modality_inputs' expected shape: [1, T, C, D_crop, H_crop, W_crop].
                      'segmentation' expected shape: [1, D_crop, H_crop, W_crop].

    Returns:
        dict: Collated batch data containing:
              'modality_inputs': [B, max_T, C, max_D, max_H, max_W]
              'segmentation': [B, 1, max_D, max_H, max_W]
              'target': [B, max_T_target] 
              'mask': [B, max_T_mask] 
              'patient_id': list[str]
    """
    # Filter out invalid samples (e.g., _prepare_sample returns None)
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    elem = batch[0]
    elem_keys = elem.keys()

    # --- First pass: determine maximum dimensions --- 
    modality_time_dims = [sample['modality_inputs'].shape[1] for sample in batch]
    target_time_dims = [sample['target'].shape[0] for sample in batch if 'target' in sample and isinstance(sample['target'], torch.Tensor)]
    mask_time_dims = [sample['mask'].shape[0] for sample in batch if 'mask' in sample and isinstance(sample['mask'], torch.Tensor)]

    # Get cropped spatial dimensions D, H, W for each sample
    # modality_inputs: [1, T, C, D, H, W]
    # segmentation: [1, D, H, W]
    modality_spatial_dims = [sample['modality_inputs'].shape[3:] for sample in batch]
    segmentation_spatial_dims = [sample['segmentation'].shape[1:] for sample in batch]
    
    # Find maximum dimensions across all samples
    max_modality_T = max(modality_time_dims) if modality_time_dims else 0
    max_target_T = max(target_time_dims) if target_time_dims else 0
    max_mask_T = max(mask_time_dims) if mask_time_dims else 0

    # Calculate maximum D, H, W (considering modality and segmentation)
    all_Ds = [dims[0] for dims in modality_spatial_dims] + [dims[0] for dims in segmentation_spatial_dims]
    all_Hs = [dims[1] for dims in modality_spatial_dims] + [dims[1] for dims in segmentation_spatial_dims]
    all_Ws = [dims[2] for dims in modality_spatial_dims] + [dims[2] for dims in segmentation_spatial_dims]
    
    max_D = max(all_Ds) if all_Ds else 0
    max_H = max(all_Hs) if all_Hs else 0
    max_W = max(all_Ws) if all_Ws else 0

    num_modalities = elem['modality_inputs'].shape[2]

    # --- Initialize batch tensors --- 
    padded_modalities = torch.zeros((len(batch), max_modality_T, num_modalities, max_D, max_H, max_W), dtype=elem['modality_inputs'].dtype)
    padded_segmentations = torch.zeros((len(batch), 1, max_D, max_H, max_W), dtype=elem['segmentation'].dtype)
    padded_targets = torch.zeros((len(batch), max_target_T), dtype=elem['target'].dtype) if max_target_T > 0 else None
    padded_masks = torch.zeros((len(batch), max_mask_T), dtype=elem['mask'].dtype) if max_mask_T > 0 else None
    patient_ids = []

    # --- Second pass: pad and copy data --- 
    for i, sample in enumerate(batch):
        # 1. Modality Inputs
        mod_input = sample['modality_inputs'][0] # Shape: [T, C, D, H, W]
        t_mod, _, d, h, w = mod_input.shape
        
        # Calculate spatial padding (W, H, D order, corresponding to last three dimensions in F.pad)
        pad_w = max_W - w
        pad_h = max_H - h
        pad_d = max_D - d
        # (left, right, top, bottom, front, back)
        spatial_padding = (pad_w // 2, pad_w - pad_w // 2, 
                           pad_h // 2, pad_h - pad_h // 2,
                           pad_d // 2, pad_d - pad_d // 2)
        
        mod_input_padded = F.pad(mod_input, spatial_padding, mode='constant', value=0)
        padded_modalities[i, :t_mod, ...] = mod_input_padded

        # 2. Segmentation
        seg_input = sample['segmentation'] # Shape: [1, D, H, W]
        _, d, h, w = seg_input.shape
        pad_w = max_W - w
        pad_h = max_H - h
        pad_d = max_D - d
        spatial_padding_seg = (pad_w // 2, pad_w - pad_w // 2, 
                               pad_h // 2, pad_h - pad_h // 2,
                               pad_d // 2, pad_d - pad_d // 2)
        
        seg_input_padded = F.pad(seg_input, spatial_padding_seg, mode='constant', value=0)
        padded_segmentations[i, ...] = seg_input_padded

        # 3. Target
        if 'target' in sample and isinstance(sample['target'], torch.Tensor) and padded_targets is not None:
            t_target = sample['target'].shape[0]
            padded_targets[i, :t_target] = sample['target']

        # 4. Mask
        if 'mask' in sample and isinstance(sample['mask'], torch.Tensor) and padded_masks is not None:
            t_mask = sample['mask'].shape[0]
            padded_masks[i, :t_mask] = sample['mask']

        # 5. Patient ID
        if 'patient_id' in sample:
            patient_ids.append(sample['patient_id'])

    # --- Build final batch dictionary --- 
    final_batch = {
        'modality_inputs': padded_modalities,
        'segmentation': padded_segmentations,
        'patient_id': patient_ids
    }
    if padded_targets is not None:
        final_batch['target'] = padded_targets
    if padded_masks is not None:
        final_batch['mask'] = padded_masks

    # Check and include other non-tensor or simply stackable fields
    for key in elem_keys:
        if key not in final_batch and key in elem:
             try:
                 # Try default stacking
                 final_batch[key] = torch.stack([s[key] for s in batch if key in s])
             except Exception as e:
                 # If failed, collect as list
                 final_batch[key] = [s[key] for s in batch if key in s]

    return final_batch

def get_valid_time_points(patient_data, modality_list):
    """
    Get valid time points containing all specified modalities
    
    Args:
        patient_data (dict): Patient data
        modality_list (list): List of required modalities
        
    Returns:
        list: List of valid time points
    """
    valid_time_points = []
    
    # Get all time points
    time_points = sorted([int(tp) for tp in patient_data.keys() if tp.isdigit()])
    
    # Check if each time point contains all required modalities
    for tp in time_points:
        tp_str = str(tp)
        if tp_str in patient_data:
            has_all_modalities = True
            for modality in modality_list:
                if modality not in patient_data[tp_str]:
                    has_all_modalities = False
                    break
            if has_all_modalities:
                valid_time_points.append(tp)
    
    return valid_time_points

def find_bounding_box(segmentation_mask, margin=(10, 20, 20)):
    """
    Calculate bounding box of non-zero region in segmentation mask.

    Args:
        segmentation_mask (np.ndarray): Segmentation mask array (D, H, W).
        margin (tuple): Margin to add in D, H, W dimensions.

    Returns:
        tuple: Crop coordinates (min_d, max_d, min_h, max_h, min_w, max_w), or None if mask is empty.
    """
    coords = np.argwhere(segmentation_mask > 0)
    if coords.size == 0:
        return None  # No non-zero region

    min_d, min_h, min_w = coords.min(axis=0)
    max_d, max_h, max_w = coords.max(axis=0)

    # Apply margin and ensure not exceeding original image bounds
    D, H, W = segmentation_mask.shape
    min_d = max(0, min_d - margin[0])
    max_d = min(D, max_d + 1 + margin[0])  # +1 because slicing excludes upper bound
    min_h = max(0, min_h - margin[1])
    max_h = min(H, max_h + 1 + margin[1])
    min_w = max(0, min_w - margin[2])
    max_w = min(W, max_w + 1 + margin[2])

    return min_d, max_d, min_h, max_h, min_w, max_w

def crop_volume(volume, bbox):
    """
    Crop 3D or 4D volume according to bounding box.

    Args:
        volume (np.ndarray): Input volume (D, H, W) or (C, D, H, W).
        bbox (tuple): Crop coordinates (min_d, max_d, min_h, max_h, min_w, max_w).

    Returns:
        np.ndarray: Cropped volume.
    """
    if bbox is None:
        return volume  # If no bounding box, return original volume

    min_d, max_d, min_h, max_h, min_w, max_w = bbox
    if volume.ndim == 3:  # (D, H, W)
        return volume[min_d:max_d, min_h:max_h, min_w:max_w]
    elif volume.ndim >= 4:  # (..., D, H, W) supports (C, D, H, W), (T, C, D, H, W), etc.
        # Assume spatial dimensions are the last three
        slices = [slice(None)] * (volume.ndim - 3) + [slice(min_d, max_d), slice(min_h, max_h), slice(min_w, max_w)]
        return volume[tuple(slices)]
    else:
         logger.warning(f"Cropping {volume.ndim}D volume not supported, returning original volume.")
         return volume
