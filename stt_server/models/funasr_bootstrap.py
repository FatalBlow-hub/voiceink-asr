# -*- coding: utf-8 -*-
"""FunASR 运行时引导工具。

在 PyInstaller / 冻结环境下，FunASR 的部分模块无法通过 inspect.getsourcelines
读取源代码，导致加载失败。这里通过猴子补丁修复该问题。
"""

import importlib
import inspect
import re


def _matches_allowed_prefix(module_name: str, prefixes: tuple) -> bool:
    return any(
        module_name == prefix.rstrip(".") or module_name.startswith(prefix)
        for prefix in prefixes
    )


def _import_required_modules(module_names: list, owner_name: str) -> None:
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception as error:
            raise RuntimeError(
                f"[{owner_name}] 预加载 FunASR 模块失败: {module_name}: {error}"
            ) from error


def _ensure_safe_inspect_getsourcelines(
    module_prefixes: tuple = ("funasr.",),
) -> None:
    existing_prefixes = tuple(
        getattr(inspect, "_voiceink_safe_getsourcelines_prefixes", tuple())
    )
    merged_prefixes = tuple(dict.fromkeys(existing_prefixes + tuple(module_prefixes)))

    if getattr(inspect, "_voiceink_safe_getsourcelines_patched", False):
        inspect._voiceink_safe_getsourcelines_prefixes = merged_prefixes
        return

    original_getsourcelines = inspect.getsourcelines

    def safe_getsourcelines(obj):
        try:
            return original_getsourcelines(obj)
        except (OSError, IOError, TypeError):
            module_name = getattr(obj, "__module__", "")
            allowed_prefixes = tuple(
                getattr(inspect, "_voiceink_safe_getsourcelines_prefixes", tuple())
            )
            if inspect.isclass(obj) and _matches_allowed_prefix(module_name, allowed_prefixes):
                return ([f"class {obj.__name__}:\n", "    pass\n"], 1)
            raise

    inspect.getsourcelines = safe_getsourcelines
    inspect._voiceink_safe_getsourcelines_patched = True
    inspect._voiceink_safe_getsourcelines_prefixes = merged_prefixes


def _ensure_registry_entry(table_name: str, key: str, owner_name: str) -> None:
    from funasr.register import tables

    registry = getattr(tables, table_name, None)
    if registry is None or key not in registry:
        raise RuntimeError(f"[{owner_name}] FunASR 注册缺失: {table_name}.{key}")


def _ensure_char_tokenizer_helpers() -> None:
    module = importlib.import_module("funasr.tokenizer.char_tokenizer")
    if hasattr(module, "load_seg_dict") and hasattr(module, "seg_tokenize"):
        return

    def load_seg_dict(seg_dict_file: str):
        seg_dict = {}
        with open(seg_dict_file, "r", encoding="utf8") as file:
            for line in file.readlines():
                parts = line.strip().split()
                if not parts:
                    continue
                seg_dict[parts[0]] = " ".join(parts[1:])
        return seg_dict

    def seg_tokenize(txt, seg_dict):
        pattern = re.compile(r"([\u4E00-\u9FA5A-Za-z0-9])")
        out_txt = ""
        for word in txt:
            word = word.lower()
            if word in seg_dict:
                out_txt += seg_dict[word] + " "
            elif pattern.match(word):
                for char in word:
                    out_txt += seg_dict.get(char, "<unk>") + " "
            else:
                out_txt += "<unk> "
        return out_txt.strip().split()

    module.load_seg_dict = load_seg_dict
    module.seg_tokenize = seg_tokenize


def prepare_paraformer_funasr_runtime() -> None:
    _ensure_safe_inspect_getsourcelines()
    _import_required_modules(
        [
            "funasr.frontends.wav_frontend",
            "funasr.tokenizer.char_tokenizer",
            "funasr.models.seaco_paraformer.model",
        ],
        "Paraformer",
    )
    _ensure_char_tokenizer_helpers()
    _ensure_registry_entry("frontend_classes", "WavFrontend", "Paraformer")
    _ensure_registry_entry("tokenizer_classes", "CharTokenizer", "Paraformer")
    _ensure_registry_entry("model_classes", "SeacoParaformer", "Paraformer")


def prepare_sensevoice_funasr_runtime() -> None:
    _ensure_safe_inspect_getsourcelines()
    _import_required_modules(
        [
            "funasr.frontends.wav_frontend",
            "funasr.tokenizer.sentencepiece_tokenizer",
            "funasr.models.sense_voice.model",
            "funasr.models.fsmn_vad_streaming.model",
        ],
        "SenseVoice",
    )
    _ensure_registry_entry("frontend_classes", "WavFrontend", "SenseVoice")
    _ensure_registry_entry(
        "tokenizer_classes", "SentencepiecesTokenizer", "SenseVoice"
    )
    _ensure_registry_entry("model_classes", "SenseVoiceSmall", "SenseVoice")


def prepare_fun_asr_nano_runtime() -> None:
    # 注意：前缀需与开源版模块路径一致
    _ensure_safe_inspect_getsourcelines(
        ("funasr.", "stt_server.models.fun_asr_nano_compat")
    )
