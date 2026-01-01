#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model_downloader.py

模型下载脚本（用于验证：能否完整下载模型文件）

设计目标：
- 不写死 HTTP 直链
- 通过 ModelScope SDK（优先）或 FunASR AutoModel（兜底）触发下载
- 可指定用户选择的根目录 root_dir（用于后续“按目录大小估算进度”）
- 以 JSON Lines 方式输出进度，便于 Rust/Tauri 读取并转发给前端

输出示例（每行一个 JSON）：
- {"event":"start","item":"asr","model_id":"damo/...","revision":"v2.0.4"}
- {"event":"stage","item":"asr","stage":"downloading"}
- {"event":"done","item":"asr","path":"..."}
- {"event":"error","item":"asr","error":"..."}

用法示例：
python model_downloader.py \
--root_dir "./models" \
  --items '{"items":[{"item":"asr","model_id":"damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch","revision":"v2.0.4"}]}'

说明：
- 第一版只输出“阶段型进度”。后续可在下载过程中通过目录大小增长估算百分比。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DownloadItem:
    # 逻辑名称：asr / vad / punc / sensevoice-onnx 等，用于前端显示
    item: str
    # 模型 ID，例如 damo/xxx (ModelScope) 或 lovemefan/sense-voice-onnx (Huggingface)
    model_id: str
    # 模型版本/分支，例如 v2.0.4 / main
    revision: str
    # 下载来源：auto / modelscope / huggingface / funasr
    source: str = "auto"
    # 可选：期望大小（字节），用于后续估算进度
    expected_size_bytes: Optional[int] = None


def _print_json(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False))
    sys.stdout.flush()


def _safe_dir_name(model_id: str) -> str:
    # 将 damo/xxx 变成 xxx，避免目录层级过深
    # 同时替换掉不适合作为目录名的字符
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    for ch in ["\\", "/", ":", "*", "?", '"', "<", ">", "|", " "]:
        name = name.replace(ch, "_")
    return name


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _get_dir_size(path: str) -> int:
    """获取目录总大小（字节）"""
    total = 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return total


class ProgressMonitor:
    """后台线程监控下载进度，每 5% 输出一次"""
    
    def __init__(self, item: str, local_dir: str, expected_size: Optional[int], interval: float = 1.0):
        self.item = item
        self.local_dir = local_dir
        self.expected_size = expected_size
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_percent = 0
    
    def start(self) -> None:
        if self.expected_size and self.expected_size > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
    
    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            current_size = _get_dir_size(self.local_dir)
            percent = min(99, int(current_size * 100 / self.expected_size))  # type: ignore
            # 每 5% 输出一次，更快的反馈
            if percent >= self._last_percent + 5:
                self._last_percent = (percent // 5) * 5
                _print_json({
                    "event": "progress",
                    "item": self.item,
                    "percent": self._last_percent,
                    "current_bytes": current_size,
                    "expected_bytes": self.expected_size,
                })


def try_download_with_modelscope(item: DownloadItem, root_dir: str) -> str:
    """优先使用 ModelScope 的 snapshot_download。

    好处：
    - 可明确指定 local_dir（即用户指定的 root_dir 下的目录），便于后续按目录大小估算进度。
    """
    try:
        # 兼容不同版本的 modelscope
        from modelscope.hub.snapshot_download import snapshot_download  # type: ignore
    except Exception as e:
        raise RuntimeError(f"ModelScope SDK 不可用: {e}")

    local_dir = os.path.join(root_dir, _safe_dir_name(item.model_id))
    _ensure_dir(local_dir)

    # stage: downloading
    _print_json({
        "event": "stage",
        "item": item.item,
        "stage": "downloading",
        "model_id": item.model_id,
        "revision": item.revision,
    })

    # 启动进度监控
    monitor = ProgressMonitor(item.item, local_dir, item.expected_size_bytes)
    monitor.start()

    try:
        # 注意：不同版本参数略有差异，这里按“由强到弱”尝试：
        # 1) cache_dir + local_dir + local_dir_use_symlinks
        # 2) cache_dir + local_dir
        # 3) 仅 cache_dir
        try:
            path = snapshot_download(
                model_id=item.model_id,
                revision=item.revision,
                cache_dir=root_dir,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        except TypeError:
            try:
                path = snapshot_download(
                    model_id=item.model_id,
                    revision=item.revision,
                    cache_dir=root_dir,
                    local_dir=local_dir,
                )
            except TypeError:
                path = snapshot_download(
                    model_id=item.model_id,
                    revision=item.revision,
                    cache_dir=root_dir,
                )
    finally:
        monitor.stop()

    # snapshot_download 返回的可能是 cache path 或 local_dir
    return str(path)


def try_download_with_huggingface(item: DownloadItem, root_dir: str) -> str:
    """使用 huggingface_hub 的 snapshot_download 下载模型。
    
    自动使用 HuggingFace 镜像（hf-mirror.com）以解决国内访问问题。
    """
    # 设置 HuggingFace 镜像，解决国内访问问题
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Huggingface Hub 不可用: {e}")

    local_dir = os.path.join(root_dir, _safe_dir_name(item.model_id))
    _ensure_dir(local_dir)

    _print_json({
        "event": "stage",
        "item": item.item,
        "stage": "downloading",
        "model_id": item.model_id,
        "revision": item.revision,
        "mirror": "hf-mirror.com",
    })

    # 启动进度监控
    monitor = ProgressMonitor(item.item, local_dir, item.expected_size_bytes)
    monitor.start()

    try:
        # 兼容不同版本的 huggingface_hub 参数
        try:
            path = snapshot_download(
                repo_id=item.model_id,
                revision=(item.revision or None),
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        except TypeError:
            try:
                path = snapshot_download(
                    repo_id=item.model_id,
                    revision=(item.revision or None),
                    local_dir=local_dir,
                )
            except TypeError:
                path = snapshot_download(
                    repo_id=item.model_id,
                    revision=(item.revision or None),
                    cache_dir=root_dir,
                )
    finally:
        monitor.stop()

    return str(path)


def try_download_with_funasr(item: DownloadItem, root_dir: str) -> str:
    """兜底：使用 FunASR AutoModel 触发下载。

    说明：
    - AutoModel 会走其内部依赖（可能仍由 ModelScope 下载）
    - 为了尽量让下载落到用户 root_dir，这里设置 MODELSCOPE_CACHE
    - 由于 AutoModel 细节较多，这里只做“能下载成功”的验证
    """
    # 尽量将缓存根目录指向用户指定目录
    os.environ.setdefault("MODELSCOPE_CACHE", root_dir)

    try:
        from funasr import AutoModel  # type: ignore
    except Exception as e:
        raise RuntimeError(f"FunASR 不可用: {e}")

    _print_json({
        "event": "stage",
        "item": item.item,
        "stage": "downloading",
        "model_id": item.model_id,
        "revision": item.revision,
    })

    # 触发下载
    AutoModel(model=item.model_id, model_revision=item.revision)

    # AutoModel 不一定返回落盘路径，这里返回 root_dir 作为“下载根目录”
    return root_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True, help="用户选择的模型根目录")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--items",
        help="JSON 字符串，格式：{\"items\":[{\"item\":...,\"model_id\":...,\"revision\":...}]}"
    )
    g.add_argument(
        "--items_file",
        help="items 的 JSON 文件路径（推荐，避免命令行转义问题）",
    )

    p.add_argument(
        "--prefer",
        choices=["auto", "modelscope", "huggingface", "funasr"],
        default="auto",
        help="优先使用的下载后端：auto / modelscope / huggingface / funasr（默认 auto）",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = os.path.abspath(args.root_dir)
    _ensure_dir(root_dir)

    try:
        if args.items_file:
            with open(args.items_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = json.loads(args.items)

        items_raw = raw.get("items")
        if not isinstance(items_raw, list) or not items_raw:
            raise ValueError("items 为空")

        items: List[DownloadItem] = []
        for x in items_raw:
            if not isinstance(x, dict):
                continue
            items.append(
                DownloadItem(
                    item=str(x.get("item") or "model"),
                    model_id=str(x.get("model_id") or ""),
                    revision=str(x.get("revision") or "main"),
                    source=str(x.get("source") or "auto"),
                    expected_size_bytes=(int(x["expected_size_bytes"]) if "expected_size_bytes" in x and x["expected_size_bytes"] is not None else None),
                )
            )

        # 简单校验
        for it in items:
            if not it.model_id:
                raise ValueError("存在空的 model_id")

    except Exception as e:
        _print_json({"event": "error", "stage": "parse_args", "error": str(e)})
        return 2

    _print_json({
        "event": "init",
        "root_dir": root_dir,
        "count": len(items),
        "prefer": args.prefer,
    })

    results: Dict[str, Any] = {}
    for idx, it in enumerate(items, start=1):
        _print_json({
            "event": "start",
            "index": idx,
            "count": len(items),
            "item": it.item,
            "model_id": it.model_id,
            "revision": it.revision,
            "source": it.source,
        })

        t0 = time.time()
        try:
            prefer = (args.prefer or "auto").lower()
            source = (it.source or "auto").lower()
            if source == "huggingface":
                order = ["huggingface", "modelscope", "funasr"]
            elif source == "modelscope":
                order = ["modelscope", "huggingface", "funasr"]
            elif source == "funasr":
                order = ["funasr", "modelscope", "huggingface"]
            else:
                if prefer == "huggingface":
                    order = ["huggingface", "modelscope", "funasr"]
                elif prefer == "funasr":
                    order = ["funasr", "modelscope", "huggingface"]
                else:
                    order = ["modelscope", "huggingface", "funasr"]

            # 如果明确指定了 source，则只用该后端，不回退
            if source in ("huggingface", "modelscope", "funasr"):
                order = [source]

            last_err: Optional[Exception] = None
            for backend in order:
                _print_json({
                    "event": "backend_attempt",
                    "item": it.item,
                    "backend": backend,
                })
                try:
                    if backend == "modelscope":
                        path = try_download_with_modelscope(it, root_dir)
                    elif backend == "huggingface":
                        path = try_download_with_huggingface(it, root_dir)
                    else:
                        path = try_download_with_funasr(it, root_dir)
                    break
                except Exception as e:
                    last_err = e
                    _print_json({
                        "event": "backend_failed",
                        "item": it.item,
                        "backend": backend,
                        "error": str(e),
                    })
                    continue
            else:
                raise RuntimeError(str(last_err) if last_err else "all backends failed")

            results[it.item] = {"success": True, "path": path, "seconds": round(time.time() - t0, 3)}
            _print_json({
                "event": "done",
                "item": it.item,
                "model_id": it.model_id,
                "path": path,
                "seconds": round(time.time() - t0, 3),
            })

        except Exception as e:
            results[it.item] = {"success": False, "error": str(e), "seconds": round(time.time() - t0, 3)}
            _print_json({
                "event": "error",
                "item": it.item,
                "model_id": it.model_id,
                "error": str(e),
                "seconds": round(time.time() - t0, 3),
            })
            # 不再中断后续下载，继续处理其它 items
            continue

    ok = all(v.get("success") for v in results.values()) and len(results) == len(items)
    _print_json({
        "event": "final",
        "success": ok,
        "results": results,
    })

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
