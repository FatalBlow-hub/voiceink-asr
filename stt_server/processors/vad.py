# -*- coding: utf-8 -*-
"""VAD 处理器（Silero VAD）。"""

import gc
import os
from typing import Dict, List, Optional

import numpy as np

from ..utils.logger import log


class SileroVAD:
    """Silero VAD 封装，使用官方 silero-vad 包。

    重要：必须禁用 PyTorch 梯度，否则在多次调用或长音频处理时会内存泄漏。
    参考：https://github.com/snakers4/silero-vad/discussions/173
    """

    def __init__(self, model_path: str, sample_rate: int = 16000):
        """初始化 VAD 模型。

        Args:
            model_path: 模型路径（未使用，由 silero-vad 包自动管理）
            sample_rate: 采样率
        """
        from silero_vad import get_speech_timestamps, load_silero_vad
        import torch

        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

        self.sample_rate = sample_rate
        self._model = load_silero_vad(onnx=True)
        self._get_speech_timestamps = get_speech_timestamps
        self._torch = torch

        log("info", "SileroVAD 加载完成 (使用 silero-vad 包, 梯度已禁用)")

    def reset_states(self) -> None:
        """重置 VAD 模型内部状态。"""
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> List[Dict[str, int]]:
        """获取语音时间戳。

        对于长音频（>60s），分块处理以避免内存溢出。
        """
        audio_length = len(audio)
        duration_secs = audio_length / self.sample_rate

        self.reset_states()

        MAX_CHUNK_SECS = 60

        if duration_secs <= MAX_CHUNK_SECS:
            with self._torch.no_grad():
                audio_tensor = self._torch.tensor(audio.astype(np.float32))
                try:
                    timestamps = self._get_speech_timestamps(
                        audio_tensor,
                        self._model,
                        sampling_rate=self.sample_rate,
                        threshold=threshold,
                        min_speech_duration_ms=min_speech_duration_ms,
                        min_silence_duration_ms=min_silence_duration_ms,
                        speech_pad_ms=speech_pad_ms,
                    )
                finally:
                    del audio_tensor
                    self.reset_states()
            log("info", f"VAD 统计: 检测到 {len(timestamps)} 个语音段")
            return timestamps

        log("info", f"VAD: 长音频 ({duration_secs:.1f}s)，分块处理（每块 {MAX_CHUNK_SECS}s）...")

        chunk_samples = MAX_CHUNK_SECS * self.sample_rate
        overlap_samples = 2 * self.sample_rate

        all_timestamps: List[Dict[str, int]] = []
        offset = 0
        chunk_idx = 0

        while offset < audio_length:
            chunk_end = min(offset + chunk_samples, audio_length)
            chunk_audio = audio[offset:chunk_end]

            log(
                "info",
                f"  VAD 分块 {chunk_idx + 1}: {offset/self.sample_rate:.1f}s - {chunk_end/self.sample_rate:.1f}s",
            )

            with self._torch.no_grad():
                audio_tensor = self._torch.tensor(chunk_audio.astype(np.float32))
                try:
                    chunk_timestamps = self._get_speech_timestamps(
                        audio_tensor,
                        self._model,
                        sampling_rate=self.sample_rate,
                        threshold=threshold,
                        min_speech_duration_ms=min_speech_duration_ms,
                        min_silence_duration_ms=min_silence_duration_ms,
                        speech_pad_ms=speech_pad_ms,
                    )

                    for ts in chunk_timestamps:
                        ts["start"] += offset
                        ts["end"] += offset

                    all_timestamps.extend(chunk_timestamps)
                    log("info", f"    检测到 {len(chunk_timestamps)} 个语音段")
                finally:
                    del audio_tensor
                    del chunk_audio
                    self.reset_states()
                    gc.collect()

            offset = chunk_end - overlap_samples
            if offset >= audio_length - overlap_samples:
                break
            chunk_idx += 1

        all_timestamps = self._merge_overlapping_timestamps(all_timestamps)
        log("info", f"VAD 完成: 共 {len(all_timestamps)} 个语音段")
        return all_timestamps

    def _merge_overlapping_timestamps(self, timestamps: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """合并重叠的时间戳。"""
        if not timestamps:
            return []

        sorted_ts = sorted(timestamps, key=lambda x: x["start"])

        merged = [sorted_ts[0]]
        for ts in sorted_ts[1:]:
            last = merged[-1]
            if ts["start"] <= last["end"] + int(self.sample_rate * 0.5):
                last["end"] = max(last["end"], ts["end"])
            else:
                merged.append(ts)

        return merged


def init_vad(model_dir: str = "") -> Optional[SileroVAD]:
    """初始化 VAD 模型。
    
    注意：silero-vad 包会自动下载和管理模型，model_dir 参数仅用于兼容旧接口。
    """
    try:
        return SileroVAD(model_dir)
    except Exception as e:
        log("error", f"VAD 初始化失败: {e}")
        return None
