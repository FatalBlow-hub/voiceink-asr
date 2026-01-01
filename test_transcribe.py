#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试脚本 - 演示如何调用 STT 服务。

用法：
    python test_transcribe.py --audio <音频文件路径> --model_dir <模型目录>

示例：
    python test_transcribe.py --audio test.wav --model_dir ./models/SenseVoiceSmall-onnx
"""

import argparse
import base64
import json
import subprocess
import sys
import os
import time


def send_request(proc, request: dict, timeout: int = 300) -> dict:
    """发送 JSON-RPC 请求并接收响应。
    
    Args:
        proc: 子进程
        request: JSON-RPC 请求
        timeout: 超时时间（秒），默认 300 秒（PyTorch 模型加载较慢）
    """
    import select
    import threading
    
    line = json.dumps(request, ensure_ascii=False) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()
    
    # 使用线程读取，支持超时
    result = {"response": None, "error": None}
    
    def read_response():
        try:
            result["response"] = proc.stdout.readline()
        except Exception as e:
            result["error"] = e
    
    thread = threading.Thread(target=read_response)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        raise RuntimeError(f"请求超时（{timeout}秒）")
    
    if result["error"]:
        raise result["error"]
    
    response_line = result["response"]
    if not response_line:
        # 尝试读取 stderr 获取错误信息
        try:
            stderr_output = proc.stderr.read()
            if stderr_output:
                raise RuntimeError(f"服务无响应，stderr: {stderr_output[:500]}")
        except:
            pass
        raise RuntimeError("服务无响应")
    
    return json.loads(response_line)


def main():
    parser = argparse.ArgumentParser(description="STT 服务测试脚本")
    parser.add_argument("--audio", required=True, help="音频文件路径（16kHz WAV）")
    parser.add_argument("--model_dir", required=True, help="ASR 模型目录")
    parser.add_argument("--model_type", default="sensevoice-onnx", 
                        choices=["sensevoice-onnx", "sensevoice-pytorch", "paraformer", "funasr-nano"],
                        help="模型类型（默认: sensevoice-onnx）")
    parser.add_argument("--vad_dir", default="", help="VAD 模型目录（可选）")
    parser.add_argument("--punc_dir", default="", help="PUNC 模型目录（可选）")
    args = parser.parse_args()

    # 检查音频文件
    if not os.path.exists(args.audio):
        print(f"错误: 音频文件不存在: {args.audio}")
        sys.exit(1)

    # 检查模型目录
    if not os.path.exists(args.model_dir):
        print(f"错误: 模型目录不存在: {args.model_dir}")
        sys.exit(1)

    print("=" * 60)
    print("VoiceInk STT 服务测试")
    print("=" * 60)

    # 启动服务进程
    print("\n[1/4] 启动 STT 服务...")
    proc = subprocess.Popen(
        [sys.executable, "stt_server_entry.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    try:
        # 等待服务启动
        time.sleep(2)

        # 发送 init 请求
        print(f"\n[2/4] 初始化模型: {args.model_type}")
        print(f"      模型目录: {args.model_dir}")
        
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "init",
            "params": {
                "type": "init",
                "model_dir": os.path.abspath(args.model_dir),
                "model_type": args.model_type,
                "device": "cpu",
                "vad_dir": os.path.abspath(args.vad_dir) if args.vad_dir else "",
                "punc_dir": os.path.abspath(args.punc_dir) if args.punc_dir else "",
                "enable_vad": bool(args.vad_dir),
                "enable_punc": bool(args.punc_dir),
            }
        }
        
        init_response = send_request(proc, init_request)
        
        if "error" in init_response:
            print(f"初始化失败: {init_response['error']}")
            sys.exit(1)
        
        result = init_response.get("result", {})
        print(f"      初始化成功!")
        print(f"      - 模型ID: {result.get('model_id', 'N/A')}")
        print(f"      - 设备: {result.get('device', 'N/A')}")
        print(f"      - VAD: {result.get('vad_enabled', False)}")
        print(f"      - PUNC: {result.get('punc_enabled', False)}")

        # 读取音频文件
        print(f"\n[3/4] 读取音频文件: {args.audio}")
        import wave
        
        with wave.open(args.audio, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)
        
        duration_secs = n_frames / sample_rate
        print(f"      采样率: {sample_rate} Hz")
        print(f"      声道数: {n_channels}")
        print(f"      时长: {duration_secs:.2f} 秒")

        if sample_rate != 16000:
            print(f"警告: 音频采样率不是 16kHz，可能影响识别效果")

        # 转为 base64
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # 发送转写请求
        print(f"\n[4/4] 开始转写...")
        start_time = time.time()
        
        transcribe_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "transcribe",
            "params": {
                "type": "transcribe",
                "audio_data": audio_b64,
                "language": "auto",
            }
        }
        
        transcribe_response = send_request(proc, transcribe_request)
        elapsed = time.time() - start_time

        if "error" in transcribe_response:
            print(f"转写失败: {transcribe_response['error']}")
            sys.exit(1)

        result = transcribe_response.get("result", {})
        
        print("\n" + "=" * 60)
        print("转写结果")
        print("=" * 60)
        print(f"\n文本: {result.get('text', '(空)')}")
        print(f"\n统计:")
        print(f"  - 音频时长: {result.get('duration_ms', 0)} ms")
        print(f"  - 识别延迟: {result.get('latency_ms', 0)} ms")
        print(f"  - 总耗时: {elapsed*1000:.0f} ms")
        print(f"  - 实时率: {elapsed/duration_secs:.2f}x")

        # 关闭服务
        shutdown_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "shutdown",
            "params": {"type": "shutdown"}
        }
        send_request(proc, shutdown_request)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        proc.terminate()
        proc.wait(timeout=5)

    print("\n测试完成!")


if __name__ == "__main__":
    main()
