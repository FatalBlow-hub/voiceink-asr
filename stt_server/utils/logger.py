# -*- coding: utf-8 -*-
"""日志工具。

注意：JSON-RPC 响应必须输出到 stdout（由 main.py 负责）。
这里的日志统一输出到 stderr，避免干扰协议。
"""

import sys


def log(level: str, message: str) -> None:
    """输出日志到 stderr。"""
    print(f"[{level.upper()}] {message}", file=sys.stderr, flush=True)
