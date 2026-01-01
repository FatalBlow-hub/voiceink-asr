# -*- coding: utf-8 -*-
"""
FunASR 模型加载器 - 处理 PyInstaller 兼容性问题
解决 FunASR AutoModel 在打包后的动态加载失败问题
"""

import os
from typing import Dict, Tuple, Optional
from ..utils.logger import log


def load_funasr_model(model_dir: str, device: str = "cpu") -> Optional[object]:
    """
    安全加载 FunASR 模型，处理 PyInstaller 兼容性问题
    
    Args:
        model_dir: 模型目录路径
        device: 推理设备 (cpu/cuda)
    
    Returns:
        加载的模型对象，或 None 如果加载失败
    """
    try:
        # 方案 1: 尝试使用 AutoModel（正常情况）
        log("info", f"尝试使用 AutoModel 加载模型: {model_dir}")
        from funasr import AutoModel
        
        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            disable_update=True,
            device=device,
        )
        
        # 验证模型是否正确加载
        if model is None:
            raise RuntimeError("AutoModel 返回 None")
        
        log("info", "✓ AutoModel 加载成功")
        return model
        
    except Exception as e:
        log("warning", f"AutoModel 加载失败: {e}")
        
        # 方案 2: 尝试直接导入模型类（PyInstaller 兼容）
        try:
            log("info", "尝试直接导入模型类...")
            return _load_model_direct(model_dir, device)
        except Exception as e2:
            log("error", f"直接加载也失败: {e2}")
            return None


def _load_model_direct(model_dir: str, device: str) -> Optional[object]:
    """
    直接加载模型，不使用 AutoModel（PyInstaller 兼容）
    """
    # 检查模型目录
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    
    # 检查模型类型
    config_file = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"config.yaml 不存在: {config_file}")
    
    # 读取配置判断模型类型
    import yaml
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_type = config.get('model_type', 'unknown')
    log("info", f"检测到模型类型: {model_type}")
    
    # 根据模型类型直接导入
    if 'sensevoice' in model_type.lower():
        from funasr.models.sensevoice.model import Sensevoice
        log("info", "加载 SenseVoice 模型...")
        model = Sensevoice(model_dir, device=device)
        return model
    
    elif 'paraformer' in model_type.lower():
        from funasr.models.paraformer.model import Paraformer
        log("info", "加载 Paraformer 模型...")
        model = Paraformer(model_dir, device=device)
        return model
    
    else:
        # 回退到 AutoModel
        log("warning", f"未知模型类型: {model_type}，尝试 AutoModel")
        from funasr import AutoModel
        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            disable_update=True,
            device=device,
        )
        return model

