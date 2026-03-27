# -*- coding: utf-8 -*-
"""Faster-Whisper 实现。

特点：CTranslate2 量化版 Whisper，多语言支持完整，CPU/GPU 均可用。
推荐模型来源：ModelScope angelala00/faster-whisper-small
"""

import os
import tempfile
import time
import wave
from typing import Dict, List, Tuple

import numpy as np

from ..utils.logger import log


def init_faster_whisper(model_dir: str, device: str) -> Tuple[Dict, Dict]:
    """初始化 Faster-Whisper 模型。"""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper 未安装，请运行: pip install faster-whisper"
        )

    normalized_device, device_index = _normalize_device(device)
    compute_type = "float16" if normalized_device == "cuda" else "int8"

    log("info", f"加载 Faster-Whisper 模型: {model_dir}, device={normalized_device}, compute_type={compute_type}")
    start = time.time()

    init_kwargs = {"device": normalized_device, "compute_type": compute_type}
    if device_index is not None:
        init_kwargs["device_index"] = device_index

    model = WhisperModel(model_dir, **init_kwargs)

    elapsed = int((time.time() - start) * 1000)
    log("info", f"✓ Faster-Whisper 加载完成，耗时 {elapsed}ms")

    wrapper = {"type": "faster-whisper", "model": model, "device": normalized_device}
    info = {"model_id": "faster-whisper", "device": normalized_device}
    return wrapper, info


def transcribe_faster_whisper(
    model_wrapper: Dict,
    audio_np: np.ndarray,
    language: str = "auto",
) -> str:
    """Faster-Whisper 转写，返回合并文本。"""
    text, _ = transcribe_faster_whisper_with_segments(model_wrapper, audio_np, language)
    return text


def transcribe_faster_whisper_with_segments(
    model_wrapper: Dict,
    audio_np: np.ndarray,
    language: str = "auto",
) -> Tuple[str, List[Dict]]:
    """Faster-Whisper 转写，返回文本和分段信息。"""
    model = model_wrapper["model"]
    normalized_language = _normalize_language(language)

    temp_path = _write_temp_wav(audio_np)
    try:
        kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
        if normalized_language is not None:
            kwargs["language"] = normalized_language

        segments_iter, info = model.transcribe(temp_path, **kwargs)
        segments = list(segments_iter)

        if not any(getattr(s, "text", "").strip() for s in segments):
            log("info", "Faster-Whisper VAD 过滤后无文本，回退关闭 VAD 重试")
            kwargs["vad_filter"] = False
            segments_iter, info = model.transcribe(temp_path, **kwargs)
            segments = list(segments_iter)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    detected_lang = getattr(info, "language", None) or language
    text = "".join(getattr(s, "text", "") for s in segments).strip()
    result_segments = [
        {
            "start": getattr(s, "start", 0.0),
            "end": getattr(s, "end", 0.0),
            "text": getattr(s, "text", "").strip(),
        }
        for s in segments
        if getattr(s, "text", "").strip()
    ]

    log("info", f"Faster-Whisper 识别完成: language={detected_lang}, 文本长度={len(text)}")
    return text, result_segments


# ── 私有工具函数 ──────────────────────────────────────────────────────────────

def _normalize_language(language: str):
    """将通用语言代码转换为 Whisper 支持的格式。"""
    normalized = (language or "").strip().lower()
    if not normalized or normalized == "auto":
        return None
    if normalized == "yue":
        return "zh"
    return normalized


def _normalize_device(device: str):
    """解析 device 字符串，返回 (device_str, device_index_or_None)。"""
    normalized = (device or "cpu").strip().lower()
    if normalized.startswith("cuda"):
        if ":" in normalized:
            _, _, index = normalized.partition(":")
            if index.isdigit():
                return "cuda", int(index)
        return "cuda", None
    return "cpu", None


def _write_temp_wav(audio_np: np.ndarray, sample_rate: int = 16000) -> str:
    """将 float32 numpy 音频写入临时 16-bit PCM WAV 文件。"""
    pcm = (audio_np * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(
        prefix="voiceink-whisper-", suffix=".wav", delete=False
    ) as f:
        temp_path = f.name

    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return temp_path
