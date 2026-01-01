# -*- coding: utf-8 -*-
"""SenseVoice ONNX 实现。

支持 ModelScope 官方 iic/SenseVoiceSmall-onnx 模型格式，
使用 onnxruntime 直接加载，无需额外依赖。
"""

import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

from ..utils.logger import log


class SenseVoiceFrontend:
    """SenseVoice 音频前端处理器。
    
    负责将音频信号转换为模型输入特征。
    实现 LFR (Low Frame Rate) + CMVN 特征提取。
    """

    def __init__(self, cmvn_file: str):
        """初始化前端处理器。
        
        参数:
            cmvn_file: CMVN 文件路径 (am.mvn)
        """
        # 加载 CMVN 参数
        self.mean, self.istd = self._load_cmvn(cmvn_file)
        
        # LFR 参数
        self.lfr_m = 7  # 堆叠帧数
        self.lfr_n = 6  # 跳帧数
        
        # 特征参数
        self.n_mels = 80
        self.sample_rate = 16000
        self.frame_length = 25  # ms
        self.frame_shift = 10   # ms
        self.n_fft = 400  # 25ms * 16000 / 1000
        self.hop_length = 160  # 10ms * 16000 / 1000
        self.win_length = 400

    def _load_cmvn(self, cmvn_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载 CMVN 均值和逆标准差。"""
        means = []
        istds = []
        
        with open(cmvn_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('<AddShift>'):
                i += 1
                # 读取 <LearnRateCoef> 行
                while i < len(lines) and not lines[i].strip().startswith('<LearnRateCoef>'):
                    i += 1
                if i < len(lines):
                    parts = lines[i].strip().split()
                    # 提取数值部分（跳过 <LearnRateCoef> 0 标签）
                    start_idx = 2 if parts[0] == '<LearnRateCoef>' else 0
                    for j in range(start_idx, len(parts)):
                        if parts[j].startswith('[') or parts[j].endswith(']'):
                            continue
                        try:
                            means.append(float(parts[j]))
                        except ValueError:
                            pass
            elif line.startswith('<Rescale>'):
                i += 1
                # 读取 <LearnRateCoef> 行
                while i < len(lines) and not lines[i].strip().startswith('<LearnRateCoef>'):
                    i += 1
                if i < len(lines):
                    parts = lines[i].strip().split()
                    start_idx = 2 if parts[0] == '<LearnRateCoef>' else 0
                    for j in range(start_idx, len(parts)):
                        if parts[j].startswith('[') or parts[j].endswith(']'):
                            continue
                        try:
                            istds.append(float(parts[j]))
                        except ValueError:
                            pass
            i += 1
        
        # 如果解析失败，使用备用方案：直接加载数值
        if len(means) == 0 or len(istds) == 0:
            means, istds = self._load_cmvn_fallback(cmvn_file)
        
        mean = np.array(means, dtype=np.float32)
        istd = np.array(istds, dtype=np.float32)
        
        log("info", f"CMVN 加载完成: mean shape={mean.shape}, istd shape={istd.shape}")
        return mean, istd

    def _load_cmvn_fallback(self, cmvn_file: str) -> Tuple[List[float], List[float]]:
        """备用 CMVN 加载方法。"""
        means = []
        istds = []
        
        with open(cmvn_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有浮点数
        numbers = re.findall(r'-?\d+\.\d+(?:e[+-]?\d+)?', content)
        numbers = [float(n) for n in numbers]
        
        # 假设前半部分是 mean，后半部分是 istd
        half = len(numbers) // 2
        means = numbers[:half]
        istds = numbers[half:]
        
        return means, istds

    def compute_fbank(self, audio: np.ndarray) -> np.ndarray:
        """计算 Mel-filterbank 特征。
        
        参数:
            audio: 音频数据 (float32, 16kHz)
            
        返回:
            fbank 特征 [T, n_mels]
        """
        import librosa
        
        # 确保音频是 float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 计算 mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=8000,
            window='hamming',
            center=True,
            pad_mode='reflect',
        )
        
        # 转换为 log scale
        fbank = np.log(mel_spec.T + 1e-10)
        
        return fbank.astype(np.float32)

    def apply_lfr(self, fbank: np.ndarray) -> np.ndarray:
        """应用 LFR (Low Frame Rate) 变换。
        
        将多帧特征堆叠为单帧，降低帧率。
        
        参数:
            fbank: 输入特征 [T, D]
            
        返回:
            LFR 特征 [T', D * lfr_m]
        """
        T, D = fbank.shape
        T_lfr = (T + self.lfr_n - 1) // self.lfr_n
        
        # 填充到能被 lfr_n 整除
        pad_len = T_lfr * self.lfr_n - T
        if pad_len > 0:
            fbank = np.pad(fbank, ((0, pad_len), (0, 0)), mode='edge')
        
        # 堆叠帧
        lfr_feats = []
        for i in range(T_lfr):
            start = i * self.lfr_n
            frames = []
            for j in range(self.lfr_m):
                idx = min(start + j, fbank.shape[0] - 1)
                frames.append(fbank[idx])
            lfr_feats.append(np.concatenate(frames))
        
        return np.array(lfr_feats, dtype=np.float32)

    def apply_cmvn(self, feats: np.ndarray) -> np.ndarray:
        """应用 CMVN 归一化。"""
        # 确保维度匹配
        if feats.shape[-1] != len(self.mean):
            # 如果维度不匹配，可能是 LFR 导致的
            # 尝试截断或填充
            target_dim = len(self.mean)
            if feats.shape[-1] > target_dim:
                feats = feats[..., :target_dim]
            else:
                pad_width = [(0, 0)] * (len(feats.shape) - 1) + [(0, target_dim - feats.shape[-1])]
                feats = np.pad(feats, pad_width, mode='constant')
        
        return (feats + self.mean) * self.istd

    def get_features(self, audio: np.ndarray) -> np.ndarray:
        """提取完整的音频特征。
        
        参数:
            audio: 音频数据 (float32, 16kHz, mono)
            
        返回:
            特征 [T, D] 适合模型输入
        """
        # 1. 计算 Mel-filterbank
        fbank = self.compute_fbank(audio)
        
        # 2. 应用 LFR
        lfr_feats = self.apply_lfr(fbank)
        
        # 3. 应用 CMVN
        feats = self.apply_cmvn(lfr_feats)
        
        return feats


class SenseVoiceONNXModel:
    """SenseVoice ONNX 模型封装。
    
    直接使用 onnxruntime 加载 ModelScope 官方模型。
    """

    # 语言 ID 映射
    LANGUAGE_IDS = {
        "auto": 0,
        "zh": 0,
        "en": 3,
        "yue": 7,
        "ja": 11,
        "ko": 12,
    }

    def __init__(self, model_path: str, tokens_path: str, num_threads: int = 4):
        """初始化模型。
        
        参数:
            model_path: ONNX 模型文件路径
            tokens_path: tokens.json 文件路径
            num_threads: 推理线程数
        """
        # 加载 tokens
        with open(tokens_path, 'r', encoding='utf-8') as f:
            tokens_data = json.load(f)
        
        # 构建 ID 到 token 的映射
        # 支持两种格式：
        # 1. 列表格式: ["<unk>", "<s>", ...] - 索引即为 ID
        # 2. 字典格式: {"token": id, ...}
        if isinstance(tokens_data, list):
            # 列表格式 - ModelScope 官方格式
            self.id2token = {i: token for i, token in enumerate(tokens_data)}
        else:
            # 字典格式
            self.id2token = {int(v): k for k, v in tokens_data.items()}
        
        # 创建 ONNX 会话
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
            sess_options=opts,
        )
        
        # 获取输入输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        log("info", f"SenseVoice ONNX 模型加载完成")
        log("info", f"  输入: {self.input_names}")
        log("info", f"  输出: {self.output_names}")
        log("info", f"  词表大小: {len(self.id2token)}")

    def decode_tokens(self, token_ids: np.ndarray) -> str:
        """将 token ID 序列解码为文本。"""
        tokens = []
        for tid in token_ids:
            tid = int(tid)
            if tid in self.id2token:
                token = self.id2token[tid]
                # 跳过特殊 token
                if token.startswith('<') and token.endswith('>'):
                    continue
                tokens.append(token)
        
        # 合并 tokens
        text = ''.join(tokens)
        return text

    def __call__(
        self,
        feats: np.ndarray,
        language: str = "auto",
    ) -> str:
        """执行推理。
        
        参数:
            feats: 特征 [B, T, D]
            language: 语言代码
            
        返回:
            识别文本
        """
        batch_size = feats.shape[0]
        seq_len = feats.shape[1]
        
        # 语言 ID: 0=zh, 3=en, 7=yue, 11=ja, 12=ko
        lang_id = self.LANGUAGE_IDS.get(language, 0)
        
        # 构建输入
        # speech: [batch_size, feats_length, 560]
        # speech_lengths: [batch_size] - 序列长度
        # language: [batch_size] - 语言 ID
        # textnorm: [batch_size] - 文本规范化模式 (15=withitn, 14=woitn)
        inputs = {
            "speech": feats.astype(np.float32),
            "speech_lengths": np.array([seq_len] * batch_size, dtype=np.int32),
            "language": np.array([lang_id] * batch_size, dtype=np.int32),
            "textnorm": np.array([15] * batch_size, dtype=np.int32),  # 15 = with ITN
        }
        
        # 执行推理
        try:
            outputs = self.session.run(self.output_names, inputs)
        except Exception as e:
            log("error", f"ONNX 推理失败: {e}")
            log("error", f"  输入形状: {[(k, v.shape) for k, v in inputs.items()]}")
            raise
        
        # 解析输出 - ctc_logits [B, T, V]
        logits = outputs[0]
        
        # 取 argmax 获取 token IDs
        if len(logits.shape) == 3:
            # [B, T, V] -> [B, T]
            token_ids = np.argmax(logits, axis=-1)
        else:
            token_ids = logits
        
        # CTC 解码：去除重复和 blank token
        text = self.ctc_decode(token_ids[0])
        
        return text

    def ctc_decode(self, token_ids: np.ndarray) -> str:
        """简单的 CTC 解码，去除重复和 blank。"""
        result = []
        prev_id = -1
        blank_id = 0  # 通常 blank 是 ID 0
        
        for tid in token_ids:
            tid = int(tid)
            # 跳过 blank 和重复
            if tid != blank_id and tid != prev_id:
                if tid in self.id2token:
                    token = self.id2token[tid]
                    # 跳过特殊 token
                    if not (token.startswith('<') and token.endswith('>')):
                        result.append(token)
            prev_id = tid
        
        return ''.join(result)


def init_sensevoice_onnx(model_dir: str, use_int8: bool = True) -> Tuple[Dict, Dict]:
    """初始化 SenseVoice ONNX 模型 (CPU 版)。
    
    支持 ModelScope 官方 iic/SenseVoiceSmall-onnx 模型格式。
    
    参数:
        model_dir: 模型目录路径
        use_int8: 是否使用 INT8 量化模型
        
    返回:
        (model_wrapper, model_info) 元组
    """
    # 确定模型文件
    if use_int8:
        model_file = os.path.join(model_dir, "model_quant.onnx")
        if not os.path.exists(model_file):
            log("warn", f"量化模型不存在: {model_file}，尝试 FP32")
            model_file = os.path.join(model_dir, "model.onnx")
    else:
        model_file = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(model_file):
            log("warn", f"FP32 模型不存在: {model_file}，尝试量化模型")
            model_file = os.path.join(model_dir, "model_quant.onnx")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    
    log("info", f"加载 SenseVoice ONNX 模型: {model_file}")
    
    # 加载 tokens
    tokens_file = os.path.join(model_dir, "tokens.json")
    if not os.path.exists(tokens_file):
        raise FileNotFoundError(f"tokens 文件不存在: {tokens_file}")
    
    # 加载 CMVN
    cmvn_file = os.path.join(model_dir, "am.mvn")
    if not os.path.exists(cmvn_file):
        raise FileNotFoundError(f"CMVN 文件不存在: {cmvn_file}")
    
    # 创建前端处理器
    frontend = SenseVoiceFrontend(cmvn_file)
    
    # 创建模型
    model = SenseVoiceONNXModel(model_file, tokens_file)
    
    wrapper = {
        "type": "sensevoice-onnx",
        "model": model,
        "frontend": frontend,
    }
    
    info = {
        "model_id": "sensevoice-onnx",
        "device": "cpu",
        "use_int8": use_int8,
    }
    
    return wrapper, info
