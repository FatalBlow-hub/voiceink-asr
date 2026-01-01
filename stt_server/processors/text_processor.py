# -*- coding: utf-8 -*-
"""文本后处理。

从 ASR 输出中做轻量、保守的清洗：
- 去重复（解决 SenseVoice 重复字问题）
- 语气词过滤（保守策略）
- Paraformer 时间戳分句：按标点进行分段
"""

import re
from typing import Dict, List


# ============================================================================
# ASR 输出去重复（解决 SenseVoice 重复字问题）
# ============================================================================

def remove_repeated_chars(text: str) -> str:
    """移除 ASR 输出中的重复字/词。

    处理模式：
    - 单字重复："测测试试" -> "测试"
    - 双字重复："一下一下" -> "一下"
    - 多字重复："出现出现" -> "出现"
    """
    if not text or len(text) < 2:
        return text

    result = text

    # 1. 移除连续重复的单字（中文字符）
    result = re.sub(r"([\u4e00-\u9fff])\1+", r"\1", result)

    # 2. 移除连续重复的双字词
    result = re.sub(r"([\u4e00-\u9fff]{2})\1+", r"\1", result)

    # 3. 移除连续重复的三字词
    result = re.sub(r"([\u4e00-\u9fff]{3})\1+", r"\1", result)

    # 4. 移除日文假名（SenseVoice 多语言问题）
    result = re.sub(r"[\u3040-\u309f\u30a0-\u30ff]+", "", result)

    return result


# ============================================================================
# 语气词过滤（保守策略）
# ============================================================================

FILLER_WORDS = {
    # 纯语气词（单字，安全删除）
    "嗯",
    "呃",
    "额",
    "噢",
    "嘿",
    # 口头禅（多字，谨慎删除）
    "就是说",
    "然后呢",
    "所以说",
    "怎么说呢",
}

# 正则模式：只过滤连续2个以上的纯语气词
FILLER_PATTERNS = [
    r"^[嗯呃额噢嘿]{2,}",  # 开头连续纯语气词
    r"[嗯呃额噢嘿]{2,}$",  # 结尾连续纯语气词
]


def remove_fillers(text: str) -> str:
    """移除语气词和口头禅（保守策略）。"""
    if not text:
        return text

    result = text

    # 1. 移除完整匹配的语气词（按长度降序，避免先删短的导致长的删不掉）
    for word in sorted(FILLER_WORDS, key=len, reverse=True):
        result = result.replace(word, "")

    # 2. 应用正则模式
    for pattern in FILLER_PATTERNS:
        result = re.sub(pattern, "", result)

    # 3. 清理连续标点和空格
    result = re.sub(r"\s+", "", result)
    result = re.sub(r"，+", "，", result)
    result = re.sub(r"。+", "。", result)
    result = result.strip("，。、")

    return result


# ============================================================================
# Paraformer 时间戳分句
# ============================================================================

# 句子结束标点符号
SENTENCE_END_PUNCTUATION = set("。！？。！？.!?")
# 句内停顿标点
PAUSE_PUNCTUATION = set("，、；：,;:")


def split_by_punctuation(text: str, timestamps: List[List[int]]) -> List[Dict]:
    """根据标点符号将词级时间戳分割为句子级分段。

    Args:
        text: 完整文本
        timestamps: 字/词级时间戳 [[start_ms, end_ms], ...]

    Returns:
        分段列表 [{"start": float, "end": float, "text": str}]
    """
    segments: List[Dict] = []

    current_text = ""
    current_start = None
    current_end = None

    for i, char in enumerate(text):
        if i >= len(timestamps):
            current_text += char
            continue

        ts = timestamps[i]
        if len(ts) >= 2:
            start_sec = ts[0] / 1000.0
            end_sec = ts[1] / 1000.0

            if current_start is None:
                current_start = start_sec
            current_end = end_sec
            current_text += char

            should_split = False
            segment_duration = current_end - current_start

            # 规则 1: 遇到句子结束标点
            if char in SENTENCE_END_PUNCTUATION:
                should_split = True
            # 规则 2: 超过 10 秒且遇到停顿标点
            elif segment_duration > 10.0 and char in PAUSE_PUNCTUATION:
                should_split = True
            # 规则 3: 超过 15 秒强制切分
            elif segment_duration > 15.0:
                should_split = True

            if should_split and current_text.strip():
                segments.append({
                    "start": current_start,
                    "end": current_end,
                    "text": current_text.strip(),
                })
                current_text = ""
                current_start = None
                current_end = None

    if current_text.strip() and current_start is not None:
        segments.append({
            "start": current_start,
            "end": current_end or (timestamps[-1][1] / 1000.0 if timestamps else 0),
            "text": current_text.strip(),
        })

    return segments
