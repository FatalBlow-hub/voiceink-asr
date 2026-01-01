# -*- coding: utf-8 -*-
"""PUNC 处理器（CT-Punc ONNX）。"""

import json
import os
import re
from typing import List, Optional

import numpy as np
import yaml

from ..utils.logger import log


class CTPunc:
    """CT-Transformer 标点恢复模型 ONNX 封装。"""

    def __init__(self, model_dir: str, use_quant: bool = True):
        import onnxruntime as ort

        self.model_dir = model_dir

        # 加载配置 - 支持多种配置文件格式
        self.punc_list = self._load_punc_list(model_dir)

        # 兼容 tokens.json 和 tokens.txt 两种格式
        tokens_json_path = os.path.join(model_dir, "tokens.json")
        tokens_txt_path = os.path.join(model_dir, "tokens.txt")
        
        if os.path.exists(tokens_json_path):
            with open(tokens_json_path, "r", encoding="utf-8") as f:
                self.tokens = json.load(f)
        elif os.path.exists(tokens_txt_path):
            with open(tokens_txt_path, "r", encoding="utf-8") as f:
                self.tokens = [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"tokens 文件不存在: {model_dir}")
        
        self.token2id = {tok: i for i, tok in enumerate(self.tokens)}

        model_file = "model_quant.onnx" if use_quant else "model.onnx"
        model_path = os.path.join(model_dir, model_file)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

        log("info", f"CTPunc 加载完成: {model_path}")
        log("info", f"  标点列表: {self.punc_list}")

    def _load_punc_list(self, model_dir: str) -> List[str]:
        """加载标点列表，支持多种配置文件格式。"""
        # 方案 1: 从 punc.yaml 读取 (HuggingFace 格式)
        punc_yaml_path = os.path.join(model_dir, "punc.yaml")
        if os.path.exists(punc_yaml_path):
            with open(punc_yaml_path, "r", encoding="utf-8") as f:
                punc_config = yaml.safe_load(f)
            if punc_config and "punc_list" in punc_config:
                punc_list = punc_config["punc_list"]
                log("info", f"从 punc.yaml 加载标点列表")
                return punc_list
        
        # 方案 2: 从 config.yaml 的 model_conf.punc_list 读取 (ModelScope 格式)
        config_path = os.path.join(model_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if config:
                # 尝试从 model_conf.punc_list 读取
                if "model_conf" in config and "punc_list" in config["model_conf"]:
                    log("info", f"从 config.yaml model_conf.punc_list 加载标点列表")
                    return config["model_conf"]["punc_list"]
                # 尝试从顶层 punc_list 读取
                if "punc_list" in config:
                    log("info", f"从 config.yaml punc_list 加载标点列表")
                    return config["punc_list"]
        
        # 方案 3: 使用默认标点列表
        log("warn", f"未找到标点配置，使用默认列表")
        return ["<unk>", "_", "，", "。", "？"]

    def _tokenize(self, text: str) -> List[int]:
        ids: List[int] = []
        for char in text.lower():
            if char in self.token2id:
                ids.append(self.token2id[char])
            else:
                ids.append(self.token2id.get("<unk>", 0))
        return ids

    def __call__(self, text: str) -> str:
        if not text or not text.strip():
            return text

        clean_text = re.sub(r"[\s，。？、,\.\?!！]", "", text)
        if not clean_text:
            return text

        token_ids = self._tokenize(clean_text)
        if len(token_ids) == 0:
            return text

        input_ids = np.array([token_ids], dtype=np.int32)
        text_lengths = np.array([len(token_ids)], dtype=np.int32)

        try:
            outputs = self.session.run(
                None,
                {
                    "inputs": input_ids,
                    "text_lengths": text_lengths,
                },
            )
            logits = outputs[0][0]
            punc_ids = np.argmax(logits, axis=-1)
        except Exception as e:
            log("warn", f"CTPunc 推理失败: {e}，返回原文本")
            return text

        result: List[str] = []
        for i, char in enumerate(clean_text):
            result.append(char)
            if i < len(punc_ids):
                punc_id = int(punc_ids[i])
                if 2 <= punc_id < len(self.punc_list):
                    result.append(self.punc_list[punc_id])

        return "".join(result)


def init_punc(model_dir: str) -> Optional[CTPunc]:
    """初始化 PUNC 模型。"""
    # 检查模型文件是否存在
    model_quant = os.path.join(model_dir, "model_quant.onnx")
    model_fp32 = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_quant) and not os.path.exists(model_fp32):
        log("warn", f"PUNC 模型不存在: {model_dir}")
        return None

    try:
        return CTPunc(model_dir)
    except Exception as e:
        log("error", f"PUNC 初始化失败: {e}")
        import traceback
        log("error", traceback.format_exc())
        return None
