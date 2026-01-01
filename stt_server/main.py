# -*- coding: utf-8 -*-
"""STT 服务 - JSON-RPC over stdio。

协议：每行一个 JSON-RPC 2.0 请求/响应。
- stdout: 仅输出 JSON-RPC 响应（不可混入日志）
- stderr: 输出日志
"""

import base64
import gc
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 模型初始化函数（保留用于并行加载）
from .models.funasr_nano import init_funasr_nano
from .models.paraformer import init_paraformer
from .models.sensevoice_onnx import init_sensevoice_onnx
from .models.sensevoice_pytorch import init_sensevoice_pytorch

# 统一 Pipeline
from .models.adapters import create_model_adapter
from .models.base import TranscribeOptions
from .pipeline import TranscriptionPipeline, PipelineConfig

from .processors.diarization import load_diarization_model
from .processors.punc import init_punc
from .processors.vad import init_vad
from .utils.logger import log


# 全局 Pipeline 实例
_pipeline: Optional[TranscriptionPipeline] = None
_model_type: Optional[str] = None
_model_info: Dict[str, Any] = {}

# 人声分离模型实例（延迟加载）
_diarization_model = None
_diarization_loaded = False

# 配置开关
_enable_vad = True
_enable_punc = True
_enable_filler_filter = True


def send_response(id: int, result: Any = None, error: Dict = None) -> None:
    """发送 JSON-RPC 响应到 stdout。"""
    response = {"jsonrpc": "2.0", "id": id}
    if error:
        response["error"] = error
    else:
        response["result"] = result
    print(json.dumps(response, ensure_ascii=False), flush=True)


def handle_init(id: int, params: Dict) -> None:
    """处理初始化请求（并行加载模型优化）。"""
    global _pipeline, _model_type, _model_info
    global _enable_vad, _enable_punc, _enable_filler_filter

    model_dir = params.get("model_dir", "")
    model_type = params.get("model_type", "sensevoice-onnx")
    device = params.get("device", "cpu")
    use_int8 = params.get("use_int8", True)

    vad_dir = params.get("vad_dir", "")
    punc_dir = params.get("punc_dir", "")
    _enable_vad = params.get("enable_vad", True)
    _enable_punc = params.get("enable_punc", True)
    _enable_filler_filter = params.get("enable_filler_filter", True)

    log("info", f"初始化模型: type={model_type}, dir={model_dir}, device={device}")
    log("info", f"VAD: enable={_enable_vad}, dir={vad_dir}")
    log("info", f"PUNC: enable={_enable_punc}, dir={punc_dir}")
    log("info", f"语气词过滤: enable={_enable_filler_filter}")

    try:
        start_time = time.time()
        log("info", "开始并行加载模型 (VAD + ASR + PUNC)...")

        def load_asr():
            asr_start = time.time()
            if model_type == "sensevoice-onnx":
                result = init_sensevoice_onnx(model_dir, use_int8)
            elif model_type == "sensevoice-pytorch":
                result = init_sensevoice_pytorch(model_dir, device)
            elif model_type == "funasr-nano":
                result = init_funasr_nano(model_dir, device)
            elif model_type == "paraformer":
                result = init_paraformer(model_dir, device)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            asr_time = time.time() - asr_start
            log("info", f"  ASR 加载完成: {asr_time:.2f}s")
            return ("asr", result)

        def load_vad():
            if not _enable_vad or not vad_dir:
                return ("vad", None)
            vad_start = time.time()
            result = init_vad(vad_dir)
            vad_time = time.time() - vad_start
            log("info", f"  VAD 加载完成: {vad_time:.2f}s")
            return ("vad", result)

        def load_punc():
            if not _enable_punc or not punc_dir:
                return ("punc", None)
            punc_start = time.time()
            result = init_punc(punc_dir)
            punc_time = time.time() - punc_start
            log("info", f"  PUNC 加载完成: {punc_time:.2f}s")
            return ("punc", result)

        results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(load_asr),
                executor.submit(load_vad),
                executor.submit(load_punc),
            ]

            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result

        model_wrapper, _model_info = results["asr"]
        _model_type = model_type

        # 保存 models 目录路径（用于时间戳模型等）
        models_dir = os.path.dirname(model_dir)
        _model_info["models_dir"] = models_dir

        vad_model = results.get("vad")
        if _enable_vad and not vad_model:
            log("warn", "VAD 模型加载失败，禁用 VAD")
            _enable_vad = False
        elif not _enable_vad:
            log("info", "VAD 已禁用")

        punc_model = results.get("punc")
        if _enable_punc and not punc_model:
            log("warn", "PUNC 模型加载失败，禁用 PUNC")
            _enable_punc = False
        elif not _enable_punc:
            log("info", "PUNC 已禁用")

        # 创建模型适配器和 Pipeline
        asr_adapter = create_model_adapter(model_type, model_wrapper, models_dir)
        pipeline_config = PipelineConfig(models_dir=models_dir)
        _pipeline = TranscriptionPipeline(
            asr_model=asr_adapter,
            vad_model=vad_model if _enable_vad else None,
            punc_model=punc_model if _enable_punc else None,
            config=pipeline_config,
        )

        load_time = time.time() - start_time
        _model_info["load_time_s"] = load_time
        _model_info["vad_enabled"] = _enable_vad and vad_model is not None
        _model_info["punc_enabled"] = _enable_punc and punc_model is not None

        log("info", f"模型并行加载完成: {model_type}, 总耗时: {load_time:.2f}s")
        log("info", f"  VAD: {_model_info['vad_enabled']}, PUNC: {_model_info['punc_enabled']}")

        actual_device = _model_info.get("device", device)

        send_response(
            id,
            result={
                "type": "init_ok",
                "model_id": _model_info.get("model_id", model_dir),
                "model_type": model_type,
                "device": actual_device,
                "streaming_supported": False,
                "vad_enabled": _model_info["vad_enabled"],
                "punc_enabled": _model_info["punc_enabled"],
            },
        )

    except Exception as e:
        log("error", f"模型初始化失败: {e}")
        log("error", traceback.format_exc())
        send_response(
            id,
            error={
                "code": -32000,
                "message": f"模型加载失败: {str(e)}",
            },
        )


def _ensure_diarization_loaded() -> None:
    global _diarization_model, _diarization_loaded
    if _diarization_loaded:
        return

    _diarization_model = load_diarization_model()
    _diarization_loaded = True


def handle_transcribe(id: int, params: Dict) -> None:
    """处理转写请求（使用统一 Pipeline）。"""
    global _pipeline, _diarization_model

    if _pipeline is None:
        send_response(
            id,
            result={
                "type": "error",
                "code": "ModelNotInitialized",
                "message": "模型未初始化",
            },
        )
        return

    try:
        # 解析音频数据
        audio_b64 = params.get("audio_data", "")
        log("info", f"收到 Base64 数据长度: {len(audio_b64)}")
        audio_bytes = base64.b64decode(audio_b64)
        log("info", f"解码后字节数: {len(audio_bytes)}")

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        raw_max = float(np.max(np.abs(audio_np))) if len(audio_np) else 0.0
        log(
            "info",
            f"音频诊断: 原始 max={raw_max:.1f}, min={float(np.min(audio_np)) if len(audio_np) else 0.0:.1f}, max={float(np.max(audio_np)) if len(audio_np) else 0.0:.1f}",
        )

        audio_np = audio_np / 32768.0

        norm_max = float(np.max(np.abs(audio_np))) if len(audio_np) else 0.0
        log("info", f"音频诊断: 归一化后 max={norm_max:.4f}, mean={float(np.mean(np.abs(audio_np))) if len(audio_np) else 0.0:.4f}")

        # 构建转写选项
        options = TranscribeOptions(
            language=params.get("language", "auto"),
            use_vad=params.get("use_vad", _enable_vad),
            use_punc=params.get("use_punc", _enable_punc),
            enable_filler_filter=_enable_filler_filter,
            return_timestamps=params.get("return_timestamps", False),
            enable_diarization=params.get("enable_diarization", False),
        )

        log("info", f"开始转写: {len(audio_bytes)} bytes, model={_model_type}")
        log("info", f"  VAD={options.use_vad}, PUNC={options.use_punc}, Diarization={options.enable_diarization}")

        # 延迟加载人声分离模型
        if options.enable_diarization and _pipeline.diarization_model is None:
            _ensure_diarization_loaded()
            _pipeline.diarization_model = _diarization_model

        # 调用 Pipeline 执行转写
        result = _pipeline.transcribe(audio_np, options)

        # 构建响应
        response_data = result.to_dict()

        # 清理
        del audio_np
        del audio_bytes
        del audio_b64
        gc.collect()

        send_response(id, result=response_data)

    except Exception as e:
        log("error", f"转写失败: {e}")
        log("error", traceback.format_exc())
        send_response(
            id,
            result={
                "type": "error",
                "code": "TranscribeFailed",
                "message": str(e),
            },
        )


def handle_ping(id: int, params: Dict) -> None:
    send_response(id, result={"type": "pong"})


def handle_shutdown(id: int, params: Dict) -> None:
    global _pipeline, _model_type
    log("info", "收到关闭请求，正在退出...")

    _pipeline = None
    _model_type = None

    send_response(id, result={"type": "shutdown_ok"})
    sys.exit(0)


def handle_request(request: Dict) -> None:
    id = request.get("id", 0)
    method = request.get("method", "")
    params = request.get("params", {})

    req_type = params.get("type", method)

    handlers = {
        "init": handle_init,
        "transcribe": handle_transcribe,
        "ping": handle_ping,
        "shutdown": handle_shutdown,
    }

    handler = handlers.get(req_type)
    if handler:
        handler(id, params)
    else:
        send_response(
            id,
            error={
                "code": -32601,
                "message": f"未知方法: {req_type}",
            },
        )


def main() -> None:
    log("info", "STT 服务启动，等待请求...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            handle_request(request)
        except json.JSONDecodeError as e:
            log("error", f"JSON 解析错误: {e}")
            send_response(
                0,
                error={
                    "code": -32700,
                    "message": "JSON 解析错误",
                },
            )
        except Exception as e:
            log("error", f"处理请求时发生错误: {e}")
            log("error", traceback.format_exc())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("info", "收到中断信号，退出")
    except Exception as e:
        log("error", f"服务异常退出: {e}")
        sys.exit(1)
