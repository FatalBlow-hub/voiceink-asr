# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-03-27

### Fixed

- **关键 Bug 修复**: `funasr` 在首次 `import` 时会向 `stdout` 打印版本信息（如 `funasr version: 1.2.9.`），导致 JSON-RPC over stdio 协议通道被污染，客户端解析响应时抛出 `JSONDecodeError`。
  - `stt_server_entry.py`: 在所有第三方模块导入前，将 `sys.stdout` 重定向至 `sys.stderr`，从源头阻断第三方库的 stdout 打印
  - `stt_server/main.py`: `send_response()` 改用 `sys.__stdout__`（Python 原始 stdout 引用）输出 JSON-RPC 响应，不受 `sys.stdout` 重定向影响
  - `stt_server/models/funasr_loader.py`: 在 `from funasr import AutoModel` 前后加入 stdout 保护（双重防御）

## [1.0.0] - 2026-03-26

### Added

- 基于 FunASR/SenseVoice 的本地语音识别服务
- JSON-RPC 2.0 over stdio 协议，适配任意语言前端集成
- 支持多种 ASR 模型：SenseVoice ONNX、SenseVoice PyTorch、Paraformer、FunASR-Nano
- Silero VAD 语音活动检测，支持长音频智能分块
- CT-Punc ONNX/PyTorch 标点恢复
- 并行模型加载（VAD + ASR + PUNC 并发初始化）
- GPU 加速支持（通过环境变量配置）
- `model_downloader.py`: 支持从 HuggingFace / ModelScope 下载模型
- `test_transcribe.py`: 完整的端到端测试脚本
