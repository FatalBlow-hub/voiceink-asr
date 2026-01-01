# -*- coding: utf-8 -*-
"""Paraformer 实现（含时间戳预测与 VAD 分块）。"""

import gc
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..processors.text_processor import remove_fillers, remove_repeated_chars, split_by_punctuation
from ..utils.logger import log


_timestamp_model = None
_timestamp_model_loaded = False


def init_paraformer(model_dir: str, device: str) -> Tuple[Dict, Dict]:
    """初始化 Paraformer 模型（当前实现固定 CPU）。"""
    from .funasr_loader import load_funasr_model

    log("info", "加载 Paraformer 模型, 设备: cpu")

    # 使用兼容 PyInstaller 的加载器
    model = load_funasr_model(model_dir, "cpu")

    if model is None:
        raise RuntimeError("无法加载 Paraformer 模型")

    wrapper = {
        "type": "paraformer",
        "model": model,
    }

    info = {
        "model_id": "paraformer",
        "device": "cpu",
    }

    return wrapper, info


def _ensure_timestamp_model(models_dir: str) -> Optional[object]:
    """延迟加载时间戳预测模型 (fa-zh)。"""
    global _timestamp_model, _timestamp_model_loaded

    if _timestamp_model_loaded:
        return _timestamp_model

    try:
        from funasr import AutoModel

        ts_model_path = os.path.join(models_dir, "speech_timestamp_prediction-v1-16k-offline")
        if not os.path.exists(ts_model_path):
            log("warn", f"时间戳模型不存在: {ts_model_path}，将跳过时间戳预测")
            _timestamp_model = None
            _timestamp_model_loaded = True
            return None

        log("info", f"正在加载时间戳预测模型: {ts_model_path}")
        load_start = time.time()

        _timestamp_model = AutoModel(
            model=ts_model_path,
            device="cpu",
            disable_update=True,
            trust_remote_code=True,
        )

        load_time = int((time.time() - load_start) * 1000)
        log("info", f"时间戳模型加载完成，耗时 {load_time}ms")

        _timestamp_model_loaded = True
        return _timestamp_model

    except Exception as e:
        log("error", f"加载时间戳模型失败: {e}")
        _timestamp_model = None
        _timestamp_model_loaded = True
        return None


def transcribe_paraformer_with_timestamps(
    model_wrapper: Dict,
    audio_np: np.ndarray,
    vad_model=None,
    punc_model=None,
    enable_filler_filter: bool = True,
    models_dir: str = "",
    enable_timestamp: bool = True,
) -> Tuple[str, List[Dict]]:
    """Paraformer 转写，返回文本和时间戳分段。

    参数:
    - enable_timestamp: 是否启用时间戳预测模型生成细粒度时间戳。
      为 True 时加载时间戳预测模型，按标点分句生成准确时间戳；
      为 False 时仅使用 VAD 分段的粗粒度时间戳。

    约定：
    - 有 VAD 时，始终走 VAD 分段路径（过滤静音 + 分块，避免 OOM）。
    - 无 VAD 时仅允许短音频回退处理。
    """
    import tempfile

    import soundfile as sf

    model = model_wrapper["model"]
    duration_secs = len(audio_np) / 16000.0

    # 优先走 VAD
    if vad_model is not None:
        log("info", f"Paraformer: 使用 VAD 处理 ({duration_secs:.1f}s 音频), 时间戳预测={enable_timestamp}")
        return _transcribe_paraformer_with_vad(
            model=model,
            audio_np=audio_np,
            vad_model=vad_model,
            punc_model=punc_model,
            enable_filler_filter=enable_filler_filter,
            models_dir=models_dir,
            enable_timestamp=enable_timestamp,
        )

    log("warn", "VAD 不可用，直接处理音频")
    MAX_NO_VAD_DURATION = 60
    if duration_secs > MAX_NO_VAD_DURATION:
        log("error", f"音频超过 {MAX_NO_VAD_DURATION}s 且 VAD 不可用，拒绝处理")
        return "", []

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_audio_path = f.name

    try:
        sf.write(temp_audio_path, audio_np, 16000)

        result = model.generate(input=temp_audio_path)

        text = ""
        if result and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = str(item)

        log("info", f"Paraformer ASR 结果: '{text[:100]}...'" if len(text) > 100 else f"Paraformer ASR 结果: '{text}'")

        # 后处理：去重复 + 语气词过滤
        if text:
            text = remove_repeated_chars(text)
            if enable_filler_filter:
                text = remove_fillers(text)

        # 先加标点（分句依赖标点）
        text_with_punc = text
        if text and punc_model is not None:
            try:
                text_with_punc = punc_model(text)
                log(
                    "info",
                    f"标点恢复后: '{text_with_punc[:100]}...'" if len(text_with_punc) > 100 else f"标点恢复后: '{text_with_punc}'",
                )
            except Exception as e:
                log("warn", f"标点恢复失败: {e}")
                text_with_punc = text

        segments: List[Dict] = []

        # 时间戳预测（仅在 enable_timestamp 且无 VAD 回退路径时执行）
        if text_with_punc and enable_timestamp:
            ts_model = _ensure_timestamp_model(models_dir) if models_dir else None
            if ts_model is not None:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
                        temp_text_path = f.name
                        f.write(text_with_punc)

                    ts_start = time.time()
                    ts_result = ts_model.generate(
                        input=(temp_audio_path, temp_text_path),
                        data_type=("sound", "text"),
                    )
                    ts_time = int((time.time() - ts_start) * 1000)
                    log("info", f"时间戳预测完成，耗时 {ts_time}ms")

                    try:
                        os.unlink(temp_text_path)
                    except Exception:
                        pass

                    if ts_result and len(ts_result) > 0:
                        ts_item = ts_result[0]
                        if isinstance(ts_item, dict):
                            timestamps = ts_item.get("timestamp", [])
                            if timestamps and len(timestamps) > 0:
                                log("info", f"获取到 {len(timestamps)} 个字级时间戳")
                                segments = split_by_punctuation(text_with_punc, timestamps)
                                log("info", f"按标点分句: {len(segments)} 个分段")

                except Exception as e:
                    log("error", f"时间戳预测失败: {e}")
            else:
                log("info", "时间戳模型未加载，跳过时间戳预测")

        if not segments and text_with_punc:
            segments = [{"start": 0, "end": duration_secs, "text": text_with_punc}]

        return text_with_punc, segments

    finally:
        try:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        except Exception:
            pass


def _transcribe_paraformer_with_vad(
    model,
    audio_np: np.ndarray,
    vad_model,
    punc_model=None,
    enable_filler_filter: bool = True,
    models_dir: str = "",
    enable_timestamp: bool = True,
) -> Tuple[str, List[Dict]]:
    """Paraformer 转写（使用 VAD + 可选时间戳预测 + 标点分句）。
    
    参数:
    - enable_timestamp: 是否启用时间戳预测模型生成细粒度时间戳。
    """
    import tempfile

    import soundfile as sf

    duration_secs = len(audio_np) / 16000.0
    sample_rate = 16000

    vad_start = time.time()
    speech_segments = vad_model.get_speech_timestamps(
        audio_np,
        threshold=0.5,
        min_speech_duration_ms=500,
        min_silence_duration_ms=1000,
        speech_pad_ms=50,
    )
    vad_time = int((time.time() - vad_start) * 1000)
    log("info", f"Paraformer VAD 完成: {len(speech_segments)} 个语音段, 耗时 {vad_time}ms")

    if not speech_segments:
        log("info", "VAD 未检测到语音")
        return "", []

    # 合并相邻语音段（避免切得太碎）
    merged_segments: List[Dict] = []
    current_seg = None
    MAX_SEGMENT_DURATION = 30 * sample_rate

    for seg in speech_segments:
        if current_seg is None:
            current_seg = {"start": seg["start"], "end": seg["end"]}
        else:
            gap = seg["start"] - current_seg["end"]
            new_duration = seg["end"] - current_seg["start"]
            if gap < 2 * sample_rate and new_duration < MAX_SEGMENT_DURATION:
                current_seg["end"] = seg["end"]
            else:
                merged_segments.append(current_seg)
                current_seg = {"start": seg["start"], "end": seg["end"]}

    if current_seg:
        merged_segments.append(current_seg)

    log("info", f"合并后: {len(merged_segments)} 个语音段")

    # 延迟加载时间戳模型（仅在启用时间戳预测时）
    ts_model = None
    if enable_timestamp:
        ts_model = _ensure_timestamp_model(models_dir) if models_dir else None
        if ts_model is None:
            log("info", "时间戳预测已启用但模型未加载，将使用 VAD 粗粒度分段")
    else:
        log("info", "时间戳预测已禁用，使用 VAD 粗粒度分段")

    all_texts: List[str] = []
    all_segments: List[Dict] = []

    for i, seg in enumerate(merged_segments):
        start_sample = seg["start"]
        end_sample = seg["end"]
        start_sec = start_sample / sample_rate
        end_sec = end_sample / sample_rate
        segment_audio = audio_np[start_sample:end_sample]

        log("info", f"  处理段 {i+1}/{len(merged_segments)}: {start_sec:.1f}s - {end_sec:.1f}s ({end_sec-start_sec:.1f}s)")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, segment_audio, sample_rate)
            result = model.generate(input=temp_path)

            text = ""
            if result and len(result) > 0:
                item = result[0]
                if isinstance(item, dict):
                    text = item.get("text", "")
                else:
                    text = str(item)

            if text:
                text = remove_repeated_chars(text)
                if enable_filler_filter:
                    text = remove_fillers(text)

                # 标点恢复
                text_with_punc = text
                if punc_model is not None and text:
                    try:
                        text_with_punc = punc_model(text)
                    except Exception:
                        text_with_punc = text

                if text_with_punc:
                    all_texts.append(text_with_punc)
                    
                    # 尝试使用时间戳预测 + 标点分句
                    segment_splits: List[Dict] = []
                    if ts_model is not None:
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
                                temp_text_path = f.name
                                f.write(text_with_punc)

                            ts_result = ts_model.generate(
                                input=(temp_path, temp_text_path),
                                data_type=("sound", "text"),
                            )

                            try:
                                os.unlink(temp_text_path)
                            except Exception:
                                pass

                            if ts_result and len(ts_result) > 0:
                                ts_item = ts_result[0]
                                if isinstance(ts_item, dict):
                                    timestamps = ts_item.get("timestamp", [])
                                    if timestamps and len(timestamps) > 0:
                                        # 按标点分句，并调整时间戳偏移
                                        raw_splits = split_by_punctuation(text_with_punc, timestamps)
                                        for s in raw_splits:
                                            segment_splits.append({
                                                "start": start_sec + s["start"],  # 加上 VAD 段的偏移
                                                "end": start_sec + s["end"],
                                                "text": s["text"],
                                            })
                                        log("info", f"    时间戳分句: {len(segment_splits)} 个子段")

                        except Exception as e:
                            log("warn", f"    时间戳预测失败: {e}")
                    
                    # 如果时间戳分句成功，使用细粒度分段；否则使用 VAD 段
                    if segment_splits:
                        all_segments.extend(segment_splits)
                    else:
                        all_segments.append({
                            "start": start_sec,
                            "end": end_sec,
                            "text": text_with_punc,
                        })
                    
                    log("info", f"    ASR: '{text_with_punc[:50]}...'" if len(text_with_punc) > 50 else f"    ASR: '{text_with_punc}'")

        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
            del segment_audio

        if (i + 1) % 5 == 0:
            gc.collect()
            log("info", f"  GC: 已处理 {i+1}/{len(merged_segments)} 个分段")

    gc.collect()

    merged_text = "".join(all_texts)
    log("info", f"Paraformer VAD 转写完成: {len(all_segments)} 个分段, 总长 {len(merged_text)} 字符")

    if not all_segments and merged_text:
        all_segments = [{"start": 0, "end": duration_secs, "text": merged_text}]

    return merged_text, all_segments
