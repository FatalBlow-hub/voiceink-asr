# -*- coding: utf-8 -*-
"""FunASR-Nano 实现。"""

import sys
from typing import Dict, Tuple

import numpy as np

from ..utils.logger import log


def init_funasr_nano(model_dir: str, device: str) -> Tuple[Dict, Dict]:
    """初始化 FunASR-Nano 模型。
    
    需要模型目录中包含 model.py 文件。
    """
    # 将模型目录加入 sys.path 以便导入 model.py
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from model import FunASRNano

    log("info", f"加载 FunASR-Nano 模型: {model_dir}")

    model, model_kwargs = FunASRNano.from_pretrained(
        model=model_dir,
        device="cpu",  # 当前固定 CPU
    )
    model.eval()

    wrapper = {
        "type": "funasr-nano",
        "model": model,
        "model_kwargs": model_kwargs,
    }

    info = {
        "model_id": "funasr-nano",
        "device": "cpu",
    }

    return wrapper, info


def transcribe_funasr_nano(model_wrapper: Dict, audio_np: np.ndarray) -> str:
    """FunASR-Nano 转写（仅返回原始文本）。"""
    import torch

    model = model_wrapper["model"]
    model_kwargs = model_wrapper["model_kwargs"]

    # 支持 numpy 数组输入
    with torch.no_grad():
        result = model.inference(
            data_in=[audio_np],
            **model_kwargs,
        )

    text = ""
    if result and len(result) > 0:
        if isinstance(result[0], list) and len(result[0]) > 0:
            if isinstance(result[0][0], dict):
                text = result[0][0].get("text", "")
            else:
                text = str(result[0][0])
        elif isinstance(result[0], dict):
            text = result[0].get("text", "")
        elif isinstance(result[0], str):
            text = result[0]
        else:
            text = str(result[0])

    return text
