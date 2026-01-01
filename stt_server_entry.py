#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STT 服务入口脚本。

用途：作为 `python -u stt_server_entry.py` 的启动入口，转交给包内实现：`stt_server.main`。

注意：
- 入口文件名刻意不叫 stt_server.py，以避免与 stt_server/ 包同名导致 import 解析歧义。
"""

import os
import sys

# 有些嵌入式/定制 Python 环境会重写 sys.path（例如设置 PYTHONHOME/PYTHONPATH），
# 可能不会自动把脚本目录加入模块搜索路径。
# 这里显式加入，确保能 import 到同目录下的 stt_server/ 包。
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from stt_server.main import main


if __name__ == "__main__":
    main()
