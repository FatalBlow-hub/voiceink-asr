# -*- coding: utf-8 -*-
"""统一转写流水线。

根据模型能力自动选择工作流：
- 需要外部处理的模型：VAD → ASR → 后处理 → PUNC → 时间戳（可选）
- 自带完整功能的模型：直接调用 ASR → 后处理
"""

import gc
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .models.base import (
    ASRModel,
    ASRResult,
    ASRSegment,
    TranscribeOptions,
    TranscribeResult,
    get_model_capabilities,
)
from .processors.text_processor import remove_fillers, remove_repeated_chars, split_by_punctuation
from .utils.logger import log


# 时间戳模型（延迟加载）
_timestamp_model = None
_timestamp_model_loaded = False


@dataclass
class PipelineConfig:
    """Pipeline 配置。"""
    models_dir: str = ""
    vad_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.vad_params is None:
            self.vad_params = {
                "threshold": 0.5,
                "min_speech_duration_ms": 500,
                "min_silence_duration_ms": 1000,
                "speech_pad_ms": 50,
            }


class TranscriptionPipeline:
    """统一转写流水线。"""

    def __init__(
        self,
        asr_model: ASRModel,
        vad_model: Optional[Any] = None,
        punc_model: Optional[Callable[[str], str]] = None,
        diarization_model: Optional[Any] = None,
        config: Optional[PipelineConfig] = None,
    ):
        """初始化 Pipeline。
        
        参数:
            asr_model: ASR 模型适配器
            vad_model: VAD 模型（可选，自带 VAD 的模型不需要）
            punc_model: 标点恢复模型（可选，自带 PUNC 的模型不需要）
            diarization_model: 人声分离模型（可选）
            config: Pipeline 配置
        """
        self.asr_model = asr_model
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.diarization_model = diarization_model
        self.config = config or PipelineConfig()

        self._capabilities = asr_model.capabilities
        log("info", f"Pipeline 初始化: model={asr_model.model_id}, "
            f"builtin_vad={self._capabilities.has_builtin_vad}, "
            f"builtin_punc={self._capabilities.has_builtin_punc}")

    def transcribe(
        self,
        audio: np.ndarray,
        options: Optional[TranscribeOptions] = None,
    ) -> TranscribeResult:
        """执行转写。
        
        参数:
            audio: 音频数据（float32, 16kHz, mono）
            options: 转写选项
        
        返回:
            TranscribeResult
        """
        start_time = time.time()
        options = options or TranscribeOptions()

        duration_ms = int(len(audio) / 16.0)
        duration_secs = duration_ms / 1000.0

        log("info", f"Pipeline 开始转写: {duration_ms}ms, model={self.asr_model.model_id}")

        # 根据模型能力选择工作流
        if self._capabilities.has_builtin_vad and self._capabilities.has_builtin_punc:
            # 自带完整功能的模型（如 funasr-nano）
            text, segments = self._transcribe_builtin(audio, options)
        else:
            # 需要外部处理的模型
            text, segments = self._transcribe_with_pipeline(audio, options)

        # 人声分离（如果启用）
        if options.enable_diarization and self.diarization_model is not None:
            segments = self._apply_diarization(audio, segments)

        latency_ms = int((time.time() - start_time) * 1000)
        log("info", f"Pipeline 完成: '{text[:50]}...' (latency={latency_ms}ms)" if len(text) > 50 else f"Pipeline 完成: '{text}' (latency={latency_ms}ms)")

        return TranscribeResult(
            text=text.strip() if text else "",
            segments=segments,
            duration_ms=duration_ms,
            latency_ms=latency_ms,
            has_timestamps=options.return_timestamps or options.enable_diarization,
        )

    def _transcribe_builtin(
        self,
        audio: np.ndarray,
        options: TranscribeOptions,
    ) -> Tuple[str, List[ASRSegment]]:
        """自带完整功能的模型转写。"""
        log("info", "使用内置 VAD/PUNC 模式")

        result = self.asr_model.transcribe(audio, options.language)
        text = result.text

        # 后处理（去重复、语气词过滤）
        if text:
            text = remove_repeated_chars(text)
            if options.enable_filler_filter:
                text = remove_fillers(text)

        duration_secs = len(audio) / 16000.0
        segments = [ASRSegment(start=0, end=duration_secs, text=text)] if text else []

        return text, segments

    def _transcribe_with_pipeline(
        self,
        audio: np.ndarray,
        options: TranscribeOptions,
    ) -> Tuple[str, List[ASRSegment]]:
        """使用完整流水线转写。"""
        duration_secs = len(audio) / 16000.0

        # 1. VAD 分段
        if options.use_vad and self.vad_model is not None:
            speech_segments = self._vad_segment(audio)
            if not speech_segments:
                log("info", "VAD 未检测到语音")
                return "", []
        else:
            # 无 VAD，整段处理
            log("info", "VAD 未启用或不可用，整段处理")
            speech_segments = [{"start": 0, "end": len(audio)}]

        # 2. ASR 转写每个分段
        all_texts: List[str] = []
        all_segments: List[ASRSegment] = []

        for i, seg in enumerate(speech_segments):
            start_sample = seg["start"]
            end_sample = seg["end"]
            start_sec = start_sample / 16000.0
            end_sec = end_sample / 16000.0
            segment_audio = audio[start_sample:end_sample]

            if len(speech_segments) > 1:
                log("info", f"  处理段 {i+1}/{len(speech_segments)}: {start_sec:.1f}s - {end_sec:.1f}s")

            # ASR 转写
            result = self.asr_model.transcribe(segment_audio, options.language)
            text = result.text

            # 3. 后处理（去重复、语气词过滤）
            if text:
                text = remove_repeated_chars(text)
                if options.enable_filler_filter:
                    text = remove_fillers(text)

            if text:
                all_texts.append(text)
                all_segments.append(ASRSegment(start=start_sec, end=end_sec, text=text))

                if len(speech_segments) > 1:
                    log("info", f"    ASR: '{text[:30]}...'" if len(text) > 30 else f"    ASR: '{text}'")

            # 清理
            del segment_audio
            if (i + 1) % 10 == 0:
                gc.collect()
                log("info", f"  GC: 已处理 {i+1}/{len(speech_segments)} 个分段")

        gc.collect()

        # 合并文本
        merged_text = "".join(all_texts)

        # 4. PUNC 标点恢复
        if options.use_punc and self.punc_model is not None and merged_text:
            punc_start = time.time()
            try:
                merged_text = self.punc_model(merged_text)
                punc_time = int((time.time() - punc_start) * 1000)
                log("info", f"PUNC 完成: 耗时 {punc_time}ms")
            except Exception as e:
                log("warn", f"标点恢复失败: {e}")

        # 5. 时间戳预测（如果需要）
        if options.return_timestamps and merged_text:
            all_segments = self._predict_timestamps(audio, merged_text, all_segments)

        return merged_text, all_segments

    def _vad_segment(self, audio: np.ndarray) -> List[Dict[str, int]]:
        """VAD 分段。"""
        vad_start = time.time()
        params = self.config.vad_params

        speech_segments = self.vad_model.get_speech_timestamps(
            audio,
            threshold=params.get("threshold", 0.5),
            min_speech_duration_ms=params.get("min_speech_duration_ms", 500),
            min_silence_duration_ms=params.get("min_silence_duration_ms", 1000),
            speech_pad_ms=params.get("speech_pad_ms", 50),
        )

        vad_time = int((time.time() - vad_start) * 1000)
        log("info", f"VAD 完成: {len(speech_segments)} 个语音段, 耗时 {vad_time}ms")

        if not speech_segments:
            return []

        # 合并相邻语音段（避免切得太碎）
        merged = self._merge_segments(speech_segments)
        log("info", f"合并后: {len(merged)} 个语音段")

        return merged

    def _merge_segments(
        self,
        segments: List[Dict[str, int]],
        max_duration: int = 30 * 16000,
        max_gap: int = 2 * 16000,
    ) -> List[Dict[str, int]]:
        """合并相邻语音段。"""
        if not segments:
            return []

        merged: List[Dict[str, int]] = []
        current = None

        for seg in segments:
            if current is None:
                current = {"start": seg["start"], "end": seg["end"]}
            else:
                gap = seg["start"] - current["end"]
                new_duration = seg["end"] - current["start"]
                if gap < max_gap and new_duration < max_duration:
                    current["end"] = seg["end"]
                else:
                    merged.append(current)
                    current = {"start": seg["start"], "end": seg["end"]}

        if current:
            merged.append(current)

        return merged

    def _predict_timestamps(
        self,
        audio: np.ndarray,
        text: str,
        vad_segments: List[ASRSegment],
    ) -> List[ASRSegment]:
        """时间戳预测（使用外部时间戳模型）。"""
        ts_model = self._ensure_timestamp_model()
        if ts_model is None:
            log("info", "时间戳模型不可用，使用 VAD 粗粒度分段")
            return vad_segments

        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_audio_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            temp_text_path = f.name
            f.write(text)

        try:
            sf.write(temp_audio_path, audio, 16000)

            ts_start = time.time()
            ts_result = ts_model.generate(
                input=(temp_audio_path, temp_text_path),
                data_type=("sound", "text"),
            )
            ts_time = int((time.time() - ts_start) * 1000)
            log("info", f"时间戳预测完成，耗时 {ts_time}ms")

            if ts_result and len(ts_result) > 0:
                ts_item = ts_result[0]
                if isinstance(ts_item, dict):
                    timestamps = ts_item.get("timestamp", [])
                    if timestamps:
                        log("info", f"获取到 {len(timestamps)} 个字级时间戳")
                        raw_splits = split_by_punctuation(text, timestamps)
                        return [
                            ASRSegment(start=s["start"], end=s["end"], text=s["text"])
                            for s in raw_splits
                        ]

        except Exception as e:
            log("error", f"时间戳预测失败: {e}")
        finally:
            for path in [temp_audio_path, temp_text_path]:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception:
                    pass

        return vad_segments

    def _ensure_timestamp_model(self):
        """延迟加载时间戳预测模型。"""
        global _timestamp_model, _timestamp_model_loaded

        if _timestamp_model_loaded:
            return _timestamp_model

        if not self.config.models_dir:
            _timestamp_model_loaded = True
            return None

        try:
            from funasr import AutoModel

            ts_model_path = os.path.join(
                self.config.models_dir,
                "speech_timestamp_prediction-v1-16k-offline"
            )
            if not os.path.exists(ts_model_path):
                log("warn", f"时间戳模型不存在: {ts_model_path}")
                _timestamp_model = None
                _timestamp_model_loaded = True
                return None

            log("info", f"加载时间戳预测模型: {ts_model_path}")
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

    def _apply_diarization(
        self,
        audio: np.ndarray,
        segments: List[ASRSegment],
    ) -> List[ASRSegment]:
        """应用人声分离。"""
        if self.diarization_model is None:
            return segments

        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, audio, 16000)

            diar_start = time.time()
            raw_diar = self.diarization_model.diarize(temp_path)
            diar_time = int((time.time() - diar_start) * 1000)
            log("info", f"人声分离完成: {len(raw_diar)} 个分段, 耗时 {diar_time}ms")

            # 将说话人信息合并到 segments
            diar_segments = [
                {"start": float(item[1]), "end": float(item[2]), "speaker": int(item[3])}
                for item in raw_diar
            ]

            from .processors.diarization import merge_stt_and_diarization
            merged = merge_stt_and_diarization(
                [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
                diar_segments,
            )

            return [
                ASRSegment(
                    start=m["start"],
                    end=m["end"],
                    text=m["text"],
                    speaker=m.get("speaker"),
                )
                for m in merged
            ]

        except Exception as e:
            log("error", f"人声分离失败: {e}")
            return segments
        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
