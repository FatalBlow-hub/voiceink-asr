# -*- coding: utf-8 -*-
"""人声分离（Diarization）处理器。"""

import time
from typing import Dict, List, Optional

from ..utils.logger import log


def load_diarization_model() -> Optional[object]:
    """加载 wespeaker 人声分离模型（失败返回 None）。"""
    try:
        import warnings

        warnings.filterwarnings("ignore")

        log("info", "正在加载 wespeaker 人声分离模型...")
        load_start = time.time()

        import wespeaker

        model = wespeaker.load_model("chinese")
        load_time = int((time.time() - load_start) * 1000)
        log("info", f"wespeaker 模型加载完成，耗时 {load_time}ms")
        return model

    except Exception as e:
        log("error", f"加载 wespeaker 失败: {e}")
        return None


def merge_stt_and_diarization(stt_segments: List[Dict], diar_segments: List[Dict]) -> List[Dict]:
    """合并 STT 分段与说话人分段。

    Args:
        stt_segments: [{"start": float, "end": float, "text": str}]
        diar_segments: [{"start": float, "end": float, "speaker": int}]

    Returns:
        [{"start": float, "end": float, "text": str, "speaker": int|None}]
    """
    merged: List[Dict] = []

    for seg in stt_segments:
        stt_start = seg.get("start", 0)
        stt_end = seg.get("end", 0)
        text = seg.get("text", "")

        best_speaker = None
        best_overlap = 0.0

        for diar_seg in diar_segments:
            diar_start = diar_seg["start"]
            diar_end = diar_seg["end"]

            overlap_start = max(stt_start, diar_start)
            overlap_end = min(stt_end, diar_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg["speaker"]

        merged.append({
            "start": stt_start,
            "end": stt_end,
            "text": text,
            "speaker": best_speaker,
        })

    return merged
