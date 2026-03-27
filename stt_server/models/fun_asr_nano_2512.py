# -*- coding: utf-8 -*-
"""Fun-ASR-Nano-2512 适配器（2025 最新版）。

特点：FunAudioLLM/Fun-ASR-Nano-2512 模型，基于 LLM 的端到端语音识别，
      中英文效果优秀，自带 VAD/PUNC，模型体积约 2.15GB。
模型来源：ModelScope FunAudioLLM/Fun-ASR-Nano-2512
"""

import importlib.util
import os
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .funasr_bootstrap import prepare_fun_asr_nano_runtime
from ..utils.logger import log


def init_fun_asr_nano_2512(model_dir: str, device: str) -> Tuple[Dict, Dict]:
    """初始化 Fun-ASR-Nano-2512 模型。"""
    log("info", f"加载 Fun-ASR-Nano-2512 模型: {model_dir}, device={device}")
    start = time.time()

    prepare_fun_asr_nano_runtime()

    local_model_py = Path(model_dir) / "model.py"
    if not local_model_py.exists():
        raise RuntimeError(
            "未找到 Fun-ASR-Nano 运行时代码 model.py，"
            "请确认下载目录已包含完整模型文件。"
        )

    module_name = f"voiceink_fun_asr_nano_{abs(hash(str(local_model_py)))}"
    spec = importlib.util.spec_from_file_location(module_name, local_model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模型模块: {local_model_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # 为第三方加载链路提供 model 模块别名
    _previous_model_module = sys.modules.get("model")
    sys.modules["model"] = module
    model_dir_str = str(local_model_py.parent)
    path_inserted = False
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)
        path_inserted = True

    try:
        spec.loader.exec_module(module)

        if not hasattr(module, "FunASRNano"):
            raise RuntimeError("模型代码中未找到 FunASRNano 类")

        fun_asr_nano_cls = module.FunASRNano
        model_obj, model_kwargs = fun_asr_nano_cls.from_pretrained(
            model=model_dir,
            device=device,
        )
        _patch_llm_generate(getattr(model_obj, "llm", None))
        model_obj.eval()
    except Exception:
        # 恢复 model 模块别名
        if _previous_model_module is not None:
            sys.modules["model"] = _previous_model_module
        else:
            sys.modules.pop("model", None)
        if path_inserted:
            try:
                sys.path.remove(model_dir_str)
            except ValueError:
                pass
        raise

    elapsed = int((time.time() - start) * 1000)
    log("info", f"✓ Fun-ASR-Nano-2512 加载完成，耗时 {elapsed}ms")

    wrapper = {
        "type": "fun-asr-nano-2512",
        "model": model_obj,
        "model_kwargs": model_kwargs,
        "device": device,
        "_model_module": module,
        "_previous_model_module": _previous_model_module,
        "_path_inserted": model_dir_str if path_inserted else None,
    }
    info = {"model_id": "fun-asr-nano-2512", "device": device}
    return wrapper, info


def transcribe_fun_asr_nano_2512(
    model_wrapper: Dict,
    audio_np: np.ndarray,
    language: str = "auto",
) -> str:
    """Fun-ASR-Nano-2512 转写，返回文本。"""
    model = model_wrapper["model"]
    model_kwargs = model_wrapper.get("model_kwargs", {})

    temp_path = _write_temp_wav(audio_np)
    try:
        inference_kwargs = dict(model_kwargs)
        lang_hint = _normalize_language(language)
        if lang_hint is not None:
            inference_kwargs["language"] = lang_hint
        inference_kwargs.setdefault("itn", True)

        result = model.inference(data_in=[temp_path], **inference_kwargs)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    return _extract_text(result)


# ── 私有工具函数 ──────────────────────────────────────────────────────────────

def _normalize_language(language: str) -> Optional[str]:
    mapping = {
        "auto": None,
        "zh": "中文",
        "en": "英文",
        "ja": "日文",
        "ko": "韩文",
        "yue": "粤语",
    }
    return mapping.get((language or "").lower(), None)


def _extract_text(result) -> str:
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, list) and first:
            first = first[0]
        if isinstance(first, dict):
            return str(first.get("text", "")).strip()
        return str(first).strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result or "").strip()


def _write_temp_wav(audio_np: np.ndarray, sample_rate: int = 16000) -> str:
    pcm = (audio_np * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(
        prefix="voiceink-funasr-nano-", suffix=".wav", delete=False
    ) as f:
        temp_path = f.name

    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return temp_path


def _patch_llm_generate(llm) -> None:
    """为 Fun-ASR-Nano 的 generate 补齐 attention_mask / pad_token_id。"""
    if llm is None or getattr(llm, "_voiceink_generate_patched", False):
        return

    original_generate = llm.generate
    llm_config = getattr(llm, "config", None)

    def wrapped_generate(*args, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        shape = getattr(inputs_embeds, "shape", None)
        if kwargs.get("attention_mask") is None and shape is not None and len(shape) >= 2:
            import torch
            kwargs["attention_mask"] = torch.ones(
                (shape[0], shape[1]),
                dtype=torch.long,
                device=getattr(inputs_embeds, "device", None),
            )
        eos = getattr(llm_config, "eos_token_id", None)
        if kwargs.get("eos_token_id") is None and eos is not None:
            kwargs["eos_token_id"] = eos
        if kwargs.get("pad_token_id") is None:
            pad = getattr(llm_config, "pad_token_id", None)
            kwargs["pad_token_id"] = pad if pad is not None else eos
        return original_generate(*args, **kwargs)

    llm._voiceink_original_generate = original_generate
    llm.generate = wrapped_generate
    llm._voiceink_generate_patched = True
