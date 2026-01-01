# -*- coding: utf-8 -*-
"""SenseVoice PyTorch 实现（FunASR AutoModel）。"""

import time
from typing import Dict, List, Tuple

import numpy as np

from ..processors.text_processor import remove_fillers, remove_repeated_chars
from ..utils.logger import log


def init_sensevoice_pytorch(model_dir: str, device: str) -> Tuple[Dict, Dict]:
    """初始化 SenseVoice PyTorch 模型（完整版）。"""
    from .funasr_loader import load_funasr_model

    log("info", f"加载 SenseVoice PyTorch 模型, 设备: {device}")

    # 使用兼容 PyInstaller 的加载器
    model = load_funasr_model(model_dir, device)

    if model is None:
        raise RuntimeError("无法加载 SenseVoice PyTorch 模型")

    wrapper = {
        "type": "sensevoice-pytorch",
        "model": model,
    }

    info = {
        "model_id": "sensevoice-pytorch",
        "device": device,
    }

    return wrapper, info


def transcribe_sensevoice_pytorch_with_segments(
    model_wrapper: Dict,
    audio_np: np.ndarray,
    language: str = "auto",
    use_vad: bool = True,
    vad_model=None,
    use_punc: bool = True,
    punc_model=None,
    enable_filler_filter: bool = True,
) -> Tuple[str, List[Dict]]:
    """SenseVoice PyTorch 转写，返回文本和分段信息（完整版）。"""
    model = model_wrapper["model"]
    duration_secs = len(audio_np) / 16000.0

    if use_vad and vad_model is not None:
        vad_start = time.time()
        speech_segments = vad_model.get_speech_timestamps(
            audio_np,
            threshold=0.5,
            min_speech_duration_ms=500,
            min_silence_duration_ms=1000,
            speech_pad_ms=50,
        )
        vad_time = int((time.time() - vad_start) * 1000)
        log("info", f"VAD 完成: 检测到 {len(speech_segments)} 个语音段, 耗时 {vad_time}ms")

        if len(speech_segments) == 0:
            log("info", "VAD 未检测到语音，返回空结果")
            return "", []

        audio_segments = []
        segment_times = []
        for seg in speech_segments:
            start, end = seg["start"], seg["end"]
            audio_segments.append(audio_np[start:end])
            segment_times.append((start / 16000.0, end / 16000.0))
            log("info", f"  语音段: {start/16000:.2f}s - {end/16000:.2f}s ({(end-start)/16000:.2f}s)")

    else:
        audio_segments = [audio_np]
        segment_times = [(0, duration_secs)]

    texts: List[str] = []
    segments: List[Dict] = []

    for i, (segment, (seg_start, seg_end)) in enumerate(zip(audio_segments, segment_times)):
        try:
            # FunASR AutoModel 使用 generate 方法
            result = model.generate(
                input=segment,
                language=language,
                use_itn=True,  # 启用逆文本正则化
            )
            # generate 返回 [{"text": "...", ...}] 格式
            if result and len(result) > 0:
                text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            else:
                text = ""
        except Exception as e:
            log("error", f"段{i+1} ASR 失败: {e}")
            text = ""

        if text:
            text = remove_repeated_chars(text)
            if enable_filler_filter:
                text = remove_fillers(text)

            if text:
                texts.append(text)
                segments.append({"start": seg_start, "end": seg_end, "text": text})
                if len(audio_segments) > 1:
                    log("info", f"  段{i+1} ASR: '{text[:30]}...'" if len(text) > 30 else f"  段{i+1} ASR: '{text}'")

    merged_text = "".join(texts)

    if use_punc and punc_model is not None and merged_text:
        punc_start = time.time()
        merged_text = punc_model(merged_text)
        punc_time = int((time.time() - punc_start) * 1000)
        log("info", f"PUNC 完成: 耗时 {punc_time}ms")

    return merged_text, segments


def transcribe_sensevoice_pytorch(
    model_wrapper: Dict,
    audio_np: np.ndarray,
    language: str = "auto",
    use_vad: bool = True,
    vad_model=None,
    use_punc: bool = True,
    punc_model=None,
    enable_filler_filter: bool = True,
) -> str:
    """SenseVoice PyTorch 转写（仅返回文本）。"""
    text, _ = transcribe_sensevoice_pytorch_with_segments(
        model_wrapper=model_wrapper,
        audio_np=audio_np,
        language=language,
        use_vad=use_vad,
        vad_model=vad_model,
        use_punc=use_punc,
        punc_model=punc_model,
        enable_filler_filter=enable_filler_filter,
    )
    return text
