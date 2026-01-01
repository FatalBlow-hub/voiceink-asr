# VoiceInk / 声墨 - 本地语音识别服务

把声音变成文字，基于 FunASR/SenseVoice 的本地语音识别服务，支持实时流式转写。

## 特性

- 🎯 **多模型支持**: SenseVoice (ONNX/PyTorch)、Paraformer
- ⚡ **实时流式**: WebSocket 实时音频流转写
- 🔒 **完全本地**: 无需联网，数据不出本地
- 🎨 **VAD 支持**: 语音活动检测，智能断句
- ✨ **标点恢复**: 自动添加标点符号

## 快速开始

### 1. 安装依赖

```bash
# 推荐使用 conda 创建虚拟环境
conda create -n voiceink python=3.10
conda activate voiceink

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型

首次运行会自动下载模型，也可以手动下载：

```bash
python model_downloader.py
```

模型默认保存在 `~/.cache/modelscope/` 目录。

### 3. 启动服务

```bash
python -m stt_server.main --host 127.0.0.1 --port 6006
```

或直接运行：

```bash
python stt_server_entry.py
```

## API 接口

### WebSocket 实时转写

```
ws://127.0.0.1:6006/ws/transcribe
```

**请求格式**:
```json
{
  "type": "audio",
  "data": "<base64 编码的音频数据>",
  "sample_rate": 16000
}
```

**响应格式**:
```json
{
  "type": "result",
  "text": "识别结果文本",
  "is_final": true
}
```

### HTTP 接口

```
POST /transcribe
Content-Type: application/json

{
  "audio": "<base64 编码的音频>",
  "sample_rate": 16000
}
```

## 项目结构

```
voiceink-asr/
├── stt_server/
│   ├── main.py              # FastAPI 服务入口
│   ├── pipeline.py          # 识别流水线
│   ├── models/              # 模型适配器
│   │   ├── sensevoice_onnx.py
│   │   ├── sensevoice_pytorch.py
│   │   └── paraformer.py
│   ├── processors/          # 音频处理器
│   │   ├── vad.py           # 语音活动检测
│   │   ├── punc.py          # 标点恢复
│   │   └── text_processor.py
│   └── utils/
├── requirements.txt
└── README.md
```

## 配置

环境变量配置（可创建 `.env` 文件）：

```env
# 模型配置
MODEL_TYPE=sensevoice_onnx    # sensevoice_onnx / sensevoice_pytorch / paraformer
MODEL_PATH=                   # 自定义模型路径（可选）

# 服务配置
HOST=127.0.0.1
PORT=6006

# 处理配置
ENABLE_VAD=true
ENABLE_PUNC=true
```

## 性能建议

- **CPU**: 推荐使用 SenseVoice ONNX 版本，速度更快
- **GPU**: 如有 NVIDIA GPU，可使用 PyTorch 版本获得更好性能
- **内存**: 建议 8GB+ RAM

## License

MIT License

## 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 阿里达摩院语音识别框架
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 阿里通义实验室语音模型
