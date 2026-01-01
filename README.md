# VoiceInk / 声墨 - 本地语音识别服务

把声音变成文字，基于 FunASR/SenseVoice 的本地语音识别服务。通过 JSON-RPC over stdio 协议与客户端通信，方便集成到各种前端应用。

## 特性

- 🎯 **多模型支持**: SenseVoice ONNX/PyTorch、Paraformer、FunASR-Nano
- 🔒 **完全本地**: 无需联网，数据不出本地
- 🎨 **VAD 支持**: Silero VAD 语音活动检测，智能断句
- ✨ **标点恢复**: CT-Punc 自动添加标点符号
- 🚀 **易于集成**: JSON-RPC 协议，支持任意语言的前端调用

## 快速开始

### 1. 创建环境

```bash
# 推荐使用 conda 创建虚拟环境
conda create -n voiceink python=3.10
conda activate voiceink

# 安装依赖（注意 PyTorch CPU 版本的额外索引）
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. 下载模型

使用 `model_downloader.py` 下载模型：

```bash
# 下载推荐的 SenseVoice ONNX 模型
python model_downloader.py --root_dir "./models" --items '{
  "items": [
    {"item": "asr", "model_id": "lovemefan/SenseVoiceSmall-onnx", "revision": "main", "source": "huggingface"}
  ]
}'
```

或使用配置文件批量下载（参考 `download_items.sample.json`）。

### 3. 测试运行

```bash
# 测试转写（需要准备一个 16kHz WAV 音频文件）
python test_transcribe.py --audio test.wav --model_dir ./models/SenseVoiceSmall-onnx
```

### 4. 启动服务

```bash
python stt_server_entry.py
```

服务启动后通过 stdin 发送 JSON-RPC 请求，从 stdout 接收响应。

## 支持的模型

### ASR 语音识别模型

| 模型类型 | model_type | 说明 | 推荐场景 |
|---------|------------|------|----------|
| SenseVoice ONNX | `sensevoice-onnx` | ONNX 量化版本，速度最快 | ⭐ CPU 环境推荐 |
| SenseVoice PyTorch | `sensevoice-pytorch` | PyTorch 版本，支持 GPU | GPU 环境 |
| Paraformer | `paraformer` | 阿里达摩院大模型，精度高 | 高精度需求 |
| FunASR-Nano | `funasr-nano` | 轻量级，自带 VAD/PUNC | 资源受限环境 |

### VAD 语音活动检测

使用 **Silero VAD** 进行语音活动检测，自动由 `silero-vad` 包管理，无需手动下载。

功能：
- 智能检测语音段落
- 过滤静音区间
- 支持长音频分块处理

### PUNC 标点恢复

使用 **CT-Punc** ONNX 模型进行标点恢复。

**下载地址**：

```bash
# 从 HuggingFace 下载
python model_downloader.py --root_dir "./models" --items '{
  "items": [
    {"item": "punc", "model_id": "lovemefan/punc_ct-transformer_zh-cn-common-vocab272727-onnx", "revision": "main", "source": "huggingface"}
  ]
}'
```

或从 ModelScope 下载：
- `damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`

## 模型下载参考

### 完整模型列表

| 模型类型 | 来源 | model_id |
|---------|------|----------|
| SenseVoice ONNX | HuggingFace | `lovemefan/SenseVoiceSmall-onnx` |
| SenseVoice PyTorch | ModelScope | `iic/SenseVoiceSmall` |
| Paraformer | ModelScope | `damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| FunASR-Nano | ModelScope | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| CT-Punc (ONNX) | HuggingFace | `lovemefan/punc_ct-transformer_zh-cn-common-vocab272727-onnx` |
| CT-Punc (PyTorch) | ModelScope | `damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` |

### 下载示例

```bash
# 推荐配置：SenseVoice ONNX + CT-Punc ONNX
python model_downloader.py --root_dir "./models" --items '{
  "items": [
    {"item": "asr", "model_id": "lovemefan/SenseVoiceSmall-onnx", "revision": "main", "source": "huggingface"},
    {"item": "punc", "model_id": "lovemefan/punc_ct-transformer_zh-cn-common-vocab272727-onnx", "revision": "main", "source": "huggingface"}
  ]
}'
```

## API 接口

服务使用 **JSON-RPC 2.0 over stdio** 协议：
- stdin: 接收 JSON-RPC 请求（每行一个）
- stdout: 输出 JSON-RPC 响应
- stderr: 输出日志

### 初始化模型

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "init",
  "params": {
    "type": "init",
    "model_dir": "/path/to/SenseVoiceSmall-onnx",
    "model_type": "sensevoice-onnx",
    "device": "cpu",
    "vad_dir": "",
    "punc_dir": "/path/to/punc_model",
    "enable_vad": true,
    "enable_punc": true
  }
}
```

### 转写音频

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "transcribe",
  "params": {
    "type": "transcribe",
    "audio_data": "<base64 编码的 16kHz PCM 音频>",
    "language": "auto"
  }
}
```

### 响应格式

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "type": "transcribe_result",
    "text": "识别结果文本",
    "duration_ms": 5000,
    "latency_ms": 500
  }
}
```

## 项目结构

```
voiceink-asr/
├── stt_server/
│   ├── main.py              # JSON-RPC 服务入口
│   ├── pipeline.py          # 统一转写流水线
│   ├── models/              # ASR 模型适配器
│   │   ├── base.py          # 抽象基类和能力声明
│   │   ├── adapters.py      # 模型适配器工厂
│   │   ├── sensevoice_onnx.py
│   │   ├── sensevoice_pytorch.py
│   │   ├── paraformer.py
│   │   └── funasr_nano.py
│   ├── processors/          # 音频处理器
│   │   ├── vad.py           # Silero VAD 语音活动检测
│   │   ├── punc.py          # CT-Punc 标点恢复
│   │   └── text_processor.py
│   └── utils/
├── model_downloader.py      # 模型下载工具
├── test_transcribe.py       # 测试脚本
├── stt_server_entry.py      # 服务启动入口
├── requirements.txt
└── README.md
```

## 配置

环境变量配置（可创建 `.env` 文件，参考 `.env.example`）：

```env
# GPU 配置
VOICEINK_USE_GPU=0              # 1=启用 GPU，0=使用 CPU
VOICEINK_GPU_DEVICE_ID=0        # GPU 设备 ID
```

## 性能参考

测试环境：53秒音频，CPU 模式

| 模型 | 转写耗时 | 实时率 |
|------|---------|--------|
| sensevoice-onnx | ~3.8s | 0.07x |
| sensevoice-pytorch | ~4.1s | 0.08x |
| paraformer | ~4.0s | 0.07x |

> 实时率 = 转写时间 / 音频时长，越小越快

## License

MIT License

## 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 阿里达摩院语音识别框架
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 阿里通义实验室语音模型
- [Silero VAD](https://github.com/snakers4/silero-vad) - 语音活动检测模型
