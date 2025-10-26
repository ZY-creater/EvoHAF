import os
import yaml
import logging
import random
import numpy as np
import torch
import time
from datetime import datetime
import json # 需要导入 json 模块
from pathlib import Path
from typing import Union # Add this import

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # 以下设置有助于提高可复现性，但可能影响性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set CUDA seeds and enabled deterministic mode (seed={seed}).")
    else:
        logger.info(f"Set CPU seeds (seed={seed}).")

def load_config(config_path: str) -> Union[dict, None]:
    """从 YAML 文件加载配置字典"""
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.debug(f"从 {config_path} 加载配置成功。")
        return config
    except Exception as e:
        logger.error(f"加载配置文件 {config_path} 时出错: {e}", exc_info=True)
        return None

def setup_logging(log_dir, name=None):
    """
    设置日志记录
    
    Args:
        log_dir: 日志目录
        name: 日志名称
    
    Returns:
        logger: 日志记录器
    """
    # 获取当前时间
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 设置日志名称
    if name:
        log_name = f"{name}_{current_time}.log"
    else:
        log_name = f"train_{current_time}.log"
    
    # 设置日志路径
    log_path = os.path.join(log_dir, log_name)
    
    # 创建日志记录器
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_device(config):
    """
    获取设备
    
    Args:
        config: 配置字典
    
    Returns:
        device: 设备
    """
    if torch.cuda.is_available() and config['device']['cuda']:
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
    else:
        device = torch.device('cpu')
    return device

class Timer:
    """计时器类"""
    
    def __init__(self):
        """初始化计时器"""
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
    
    def get_elapsed_time(self):
        """
        获取经过的时间（秒）
        
        Returns:
            elapsed_time: 经过的时间（秒）
        """
        if self.start_time is None:
            raise ValueError("计时器未启动")
        
        if self.end_time is None:
            # 如果未停止，使用当前时间
            return time.time() - self.start_time
        else:
            return self.end_time - self.start_time
    
    def get_elapsed_time_str(self):
        """
        获取格式化的经过时间
        
        Returns:
            elapsed_time_str: 格式化的经过时间
        """
        elapsed_time = self.get_elapsed_time()
        
        # 转换为时分秒格式
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def count_parameters(model):
    """
    计算模型参数数量
    
    Args:
        model: 模型
    
    Returns:
        total_params: 参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_dict_to_yaml(dict_obj, save_path):
    """
    将字典保存为YAML文件
    
    Args:
        dict_obj: 字典对象
        save_path: 保存路径
    """
    with open(save_path, 'w') as f:
        yaml.dump(dict_obj, f, default_flow_style=False)

def load_dict_from_yaml(load_path):
    """
    从YAML文件加载字典
    
    Args:
        load_path: 加载路径
    
    Returns:
        dict_obj: 字典对象
    """
    with open(load_path, 'r') as f:
        dict_obj = yaml.safe_load(f)
    return dict_obj

def prepare_directories(output_dir: str):
    """创建输出目录及其子目录"""
    try:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        # 创建子目录 (例如 checkpoints, results)
        (path / 'checkpoints').mkdir(exist_ok=True)
        (path / 'results').mkdir(exist_ok=True)
        logger.info(f"确保目录存在: {output_dir} 及其子目录")
    except Exception as e:
        logger.error(f"创建目录 {output_dir} 时出错: {e}", exc_info=True)

def save_config(config: dict, config_path: str):
    """将配置字典保存到 YAML 文件"""
    try:
        prepare_directories(os.path.dirname(config_path))
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"配置已保存到: {config_path}")
    except Exception as e:
        logger.error(f"保存配置到 {config_path} 时出错: {e}", exc_info=True)

def get_project_root() -> Path:
    """获取项目的根目录 (假设此文件在 opensource/utils 下)"""
    return Path(__file__).parent.parent

# --- 添加 NpEncoder 类定义 ---
class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist() # Convert arrays to lists
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None # Or handle appropriately
        return super(NpEncoder, self).default(obj) 