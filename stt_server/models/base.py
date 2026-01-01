# -*- coding: utf-8 -*-
"""模型基类与类型定义。

统一的 ASR 模型抽象层，支持：
- 模型能力声明（是否自带 VAD/PUNC）
- 统一的转写接口
- 可扩展的模型注册机制
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ============== 数据类定义 ==============

@dataclass
class ModelCapabilities:
    """模型能力声明。
    
    用于 Pipeline 判断是否需要调用外部 VAD/PUNC。
    新增模型时只需声明其能力，Pipeline 会自动选择正确的工作流。
    """
    has_builtin_vad: bool = False
    has_builtin_punc: bool = False
    has_native_timestamps: bool = False


@dataclass
class ASRSegment:
    """单个转写分段。"""
    start: float
    end: float
    text: str
    speaker: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"start": self.start, "end": self.end, "text": self.text}
        if self.speaker is not None:
            d["speaker"] = self.speaker
        return d


@dataclass
class ASRResult:
    """ASR 模型转写结果（不含 VAD/PUNC 后处理）。"""
    text: str
    segments: List[ASRSegment] = field(default_factory=list)
    raw_output: Optional[Any] = None


@dataclass
class TranscribeResult:
    """Pipeline 最终输出结果（含完整后处理）。"""
    text: str
    segments: List[ASRSegment] = field(default_factory=list)
    duration_ms: int = 0
    latency_ms: int = 0
    has_timestamps: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "transcribe_result",
            "text": self.text,
            "duration_ms": self.duration_ms,
            "latency_ms": self.latency_ms,
            "has_timestamps": self.has_timestamps,
            "segments": [s.to_dict() for s in self.segments] if self.has_timestamps else [],
        }


@dataclass
class TranscribeOptions:
    """转写选项（每次请求可独立配置）。"""
    language: str = "auto"
    use_vad: bool = True
    use_punc: bool = True
    enable_filler_filter: bool = True
    return_timestamps: bool = False
    enable_diarization: bool = False


# ============== 抽象基类 ==============

class ASRModel(ABC):
    """ASR 模型统一接口。"""

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str = "auto") -> ASRResult:
        """转写音频片段（单段，不含 VAD/PUNC）。"""
        pass


# ============== 模型注册表 ==============

MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "sensevoice-onnx": ModelCapabilities(
        has_builtin_vad=False,
        has_builtin_punc=False,
        has_native_timestamps=False,
    ),
    "sensevoice-pytorch": ModelCapabilities(
        has_builtin_vad=False,
        has_builtin_punc=False,
        has_native_timestamps=False,
    ),
    "paraformer": ModelCapabilities(
        has_builtin_vad=False,
        has_builtin_punc=False,
        has_native_timestamps=False,
    ),
    "funasr-nano": ModelCapabilities(
        has_builtin_vad=True,
        has_builtin_punc=True,
        has_native_timestamps=False,
    ),
}


def get_model_capabilities(model_type: str) -> ModelCapabilities:
    """获取模型能力声明。"""
    return MODEL_CAPABILITIES.get(model_type, ModelCapabilities())


# ============== 类型别名（兼容旧代码） ==============

Segments = List[Dict[str, Any]]
