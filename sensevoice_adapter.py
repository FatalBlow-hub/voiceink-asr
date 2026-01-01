#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SenseVoice ONNX 适配器
提供与 stt_server.py 兼容的接口
"""

import os
import time
import numpy as np
from typing import Optional, Dict, Any

class SenseVoiceONNX:
    """SenseVoice ONNX 模型封装"""
    
    def __init__(self, model_dir: str, use_int8: bool = True):
        """
        初始化模型
        
        Args:
            model_dir: 模型目录路径
            use_int8: 是否使用 INT8 量化版本（更小更快）
        """
        self.model_dir = model_dir
        self.model = None
        self.frontend = None
        self.languages = None
        self.use_int8 = use_int8
        
    def load(self) -> Dict[str, Any]:
        """加载模型"""
        start_time = time.time()
        
        from sensevoice.sense_voice import SenseVoiceInferenceSession, languages, WavFrontend
        
        self.languages = languages
        
        # 选择模型文件
        if self.use_int8:
            encoder_file = os.path.join(self.model_dir, 'sense-voice-encoder-int8.onnx')
            if not os.path.exists(encoder_file):
                encoder_file = os.path.join(self.model_dir, 'sense-voice-encoder.onnx')
        else:
            encoder_file = os.path.join(self.model_dir, 'sense-voice-encoder.onnx')
        
        # 加载前端
        self.frontend = WavFrontend(
            cmvn_file=os.path.join(self.model_dir, 'am.mvn')
        )
        
        # 加载模型
        self.model = SenseVoiceInferenceSession(
            embedding_model_file=os.path.join(self.model_dir, 'embedding.npy'),
            encoder_model_file=encoder_file,
            bpe_model_file=os.path.join(self.model_dir, 'chn_jpn_yue_eng_ko_spectok.bpe.model')
        )
        
        load_time = time.time() - start_time
        
        return {
            "model_id": "sensevoice-onnx",
            "device": "cpu",
            "load_time_s": load_time,
            "use_int8": self.use_int8,
        }
    
    def transcribe(self, audio_data: np.ndarray, language: str = "auto") -> Dict[str, Any]:
        """
        转写音频
        
        Args:
            audio_data: 音频数据，numpy 数组，16kHz 采样率
            language: 语言代码 (auto, zh, en, yue, ja, ko)
            
        Returns:
            转写结果字典
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        start_time = time.time()
        
        # 获取语言 ID
        lang_id = self.languages.get(language, self.languages['auto'])
        
        # 提取特征
        # 如果输入是原始波形，需要先转换
        if len(audio_data.shape) == 1:
            # 保存临时文件（WavFrontend 需要文件路径）
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            try:
                sf.write(temp_path, audio_data, 16000)
                feats = self.frontend.get_features(temp_path)
            finally:
                os.unlink(temp_path)
        else:
            feats = audio_data
        
        # 添加 batch 维度
        if len(feats.shape) == 2:
            feats = feats[np.newaxis, :, :]
        
        # 推理
        text = self.model(feats, language=lang_id, use_itn=False)
        
        # 后处理：移除特殊标签
        text = self._postprocess(text)
        
        latency_ms = int((time.time() - start_time) * 1000)
        duration_ms = int(len(audio_data) / 16.0) if len(audio_data.shape) == 1 else 0
        
        return {
            "text": text,
            "duration_ms": duration_ms,
            "latency_ms": latency_ms,
        }
    
    def _postprocess(self, text: str) -> str:
        """后处理：移除特殊标签"""
        import re
        # 移除 <|...|> 格式的标签
        text = re.sub(r'<\|[^|]+\|>', '', text)
        return text.strip()
    
    def transcribe_file(self, file_path: str, language: str = "auto") -> Dict[str, Any]:
        """转写音频文件"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        start_time = time.time()
        
        # 获取语言 ID
        lang_id = self.languages.get(language, self.languages['auto'])
        
        # 提取特征
        feats = self.frontend.get_features(file_path)
        feats = feats[np.newaxis, :, :]
        
        # 推理
        text = self.model(feats, language=lang_id, use_itn=False)
        text = self._postprocess(text)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # 获取音频时长
        import soundfile as sf
        audio, sr = sf.read(file_path)
        duration_ms = int(len(audio) / sr * 1000)
        
        return {
            "text": text,
            "duration_ms": duration_ms,
            "latency_ms": latency_ms,
        }


# 测试代码
if __name__ == "__main__":
    import sys
    
    model_dir = r"E:\ShunyuAI_Tauri\models\sensevoice-onnx"
    test_audio = os.path.join(model_dir, "asr_example_zh.wav")
    
    print("=== SenseVoice ONNX 适配器测试 ===")
    
    # 创建模型
    model = SenseVoiceONNX(model_dir, use_int8=True)
    
    # 加载
    print("加载模型...")
    info = model.load()
    print(f"加载完成: {info}")
    
    # 转写文件
    print(f"\n转写文件: {test_audio}")
    result = model.transcribe_file(test_audio)
    print(f"结果: {result}")
