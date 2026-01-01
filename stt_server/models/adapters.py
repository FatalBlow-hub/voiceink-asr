# -*- coding: utf-8 -*-
"""ASR 模型适配器。

将现有模型函数封装为统一的 ASRModel 接口。
"""

import os
import re
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch

from .base import ASRModel, ASRResult, ASRSegment, ModelCapabilities, MODEL_CAPABILITIES
from ..utils.logger import log


class SenseVoiceONNXAdapter(ASRModel):
    """SenseVoice ONNX 模型适配器。
    
    支持 ModelScope 官方 iic/SenseVoiceSmall-onnx 模型。
    """

    def __init__(self, model_wrapper: Dict[str, Any]):
        self._wrapper = model_wrapper
        self._model = model_wrapper["model"]
        self._frontend = model_wrapper["frontend"]

    @property
    def model_id(self) -> str:
        return "sensevoice-onnx"

    @property
    def capabilities(self) -> ModelCapabilities:
        return MODEL_CAPABILITIES["sensevoice-onnx"]

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> ASRResult:
        """转写单段音频（不含 VAD/PUNC）。"""
        # 直接从音频数据提取特征（不需要临时文件）
        feats = self._frontend.get_features(audio)
        feats = feats[np.newaxis, :, :]  # 添加 batch 维度
        
        # 执行推理
        text = self._model(feats, language=language)
        
        # 移除特殊标记（如果有）
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()

        return ASRResult(text=text)


class SenseVoicePyTorchAdapter(ASRModel):
    """SenseVoice PyTorch 模型适配器。"""

    def __init__(self, model_wrapper: Dict[str, Any]):
        self._wrapper = model_wrapper
        self._model = model_wrapper["model"]

    @property
    def model_id(self) -> str:
        return "sensevoice-pytorch"

    @property
    def capabilities(self) -> ModelCapabilities:
        return MODEL_CAPABILITIES["sensevoice-pytorch"]

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> ASRResult:
        """转写单段音频（不含 VAD/PUNC）。"""
        try:
            # FunASR AutoModel 使用 generate 方法
            result = self._model.generate(
                input=audio,
                language=language,
                use_itn=True,  # 启用逆文本正则化
            )
            # generate 返回 [{"text": "...", ...}] 格式
            if result and len(result) > 0:
                text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            else:
                text = ""
        except Exception as e:
            log("error", f"SenseVoice PyTorch 转写失败: {e}")
            text = ""

        # 移除 SenseVoice 特殊标记: <|zh|><|NEUTRAL|><|Speech|><|withitn|> 等
        if text:
            text = re.sub(r"<\|[^|]+\|>", "", text).strip()

        return ASRResult(text=text)


class ParaformerAdapter(ASRModel):
    """Paraformer 模型适配器。"""

    def __init__(self, model_wrapper: Dict[str, Any], models_dir: str = ""):
        self._wrapper = model_wrapper
        self._model = model_wrapper["model"]
        self._models_dir = models_dir

    @property
    def model_id(self) -> str:
        return "paraformer"

    @property
    def capabilities(self) -> ModelCapabilities:
        return MODEL_CAPABILITIES["paraformer"]

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> ASRResult:
        """转写单段音频（不含 VAD/PUNC）。"""
        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, audio, 16000)
            result = self._model.generate(input=temp_path)

            text = ""
            if result and len(result) > 0:
                item = result[0]
                if isinstance(item, dict):
                    text = item.get("text", "")
                else:
                    text = str(item)
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

        return ASRResult(text=text)


class FunASRNanoAdapter(ASRModel):
    """FunASR-Nano 模型适配器（自带 VAD/PUNC）。"""

    def __init__(self, model_wrapper: Dict[str, Any]):
        self._wrapper = model_wrapper
        self._model = model_wrapper["model"]
        self._model_kwargs = model_wrapper.get("model_kwargs", {})

    @property
    def model_id(self) -> str:
        return "funasr-nano"

    @property
    def capabilities(self) -> ModelCapabilities:
        return MODEL_CAPABILITIES["funasr-nano"]

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> ASRResult:
        """转写单段音频（自带 VAD/PUNC）。"""
        try:
            # FunASR-Nano 使用 inference 方法，需要 torch tensor
            audio_tensor = torch.from_numpy(audio)
            
            with torch.no_grad():
                result = self._model.inference(
                    data_in=[audio_tensor],
                    **self._model_kwargs,
                )

            text = ""
            if result and len(result) > 0:
                # 结果可能是嵌套列表
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
        except Exception as e:
            log("error", f"FunASR-Nano 转写失败: {e}")
            text = ""

        return ASRResult(text=text)


def create_model_adapter(
    model_type: str,
    model_wrapper: Dict[str, Any],
    models_dir: str = "",
) -> ASRModel:
    """根据模型类型创建对应的适配器。
    
    参数:
        model_type: 模型类型标识
        model_wrapper: 已初始化的模型包装器
        models_dir: 模型目录（用于 Paraformer 时间戳模型等）
    
    返回:
        ASRModel 实例
    """
    adapters = {
        "sensevoice-onnx": lambda: SenseVoiceONNXAdapter(model_wrapper),
        "sensevoice-pytorch": lambda: SenseVoicePyTorchAdapter(model_wrapper),
        "paraformer": lambda: ParaformerAdapter(model_wrapper, models_dir),
        "funasr-nano": lambda: FunASRNanoAdapter(model_wrapper),
    }

    factory = adapters.get(model_type)
    if factory is None:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return factory()
