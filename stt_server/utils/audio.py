# -*- coding: utf-8 -*-
"""音频工具。

约定：
- 输入音频为 PCM 16kHz 16bit little-endian（Rust 侧传入 raw bytes，再 Base64）。
- 输出为 float32 numpy，范围归一化到 [-1, 1]。
"""

import base64
from typing import Tuple

import numpy as np


def decode_base64_pcm16(audio_b64: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int, float]:
    """解码 Base64 的 PCM16 音频为 float32 numpy。

    Returns:
        (audio_np, duration_ms, duration_secs)
    """
    audio_bytes = base64.b64decode(audio_b64)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio_np = audio_np / 32768.0

    duration_ms = int(len(audio_np) / (sample_rate / 1000.0))
    duration_secs = duration_ms / 1000.0
    return audio_np, duration_ms, duration_secs
