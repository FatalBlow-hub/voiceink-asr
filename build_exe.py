# -*- coding: utf-8 -*-
"""
STT Server 打包脚本
使用 PyInstaller 将 Python 后端打包成单个 EXE
"""

import subprocess
import sys
import os

# 切换到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# UPX 路径（用于压缩 EXE）
upx_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "tools", "upx-5.0.1-win64")

# PyInstaller Hook 路径
hook_dir = script_dir  # Hook 文件在当前目录

# 查找 funasr 包路径
try:
    import funasr
    funasr_dir = os.path.dirname(funasr.__file__)
except ImportError:
    funasr_dir = None
    print("警告: 未找到 funasr 包")

# PyInstaller 命令
cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",           # 打包成单个 EXE
    "--name", "stt_server", # 输出文件名
    "--console",           # 控制台程序（需要 stdin/stdout 通信）
    "--noconfirm",         # 覆盖已有输出
    "--clean",             # 清理临时文件
    
    # UPX 压缩（禁用 - 可能导致运行时问题）
    # "--upx-dir", upx_dir,
    "--noupx",  # 禁用 UPX 压缩

    # Hook 路径（用于正确处理 FunASR 的动态加载）
    "--additional-hooks-dir", hook_dir,

    # 隐式导入（PyInstaller 可能检测不到的模块）
    "--hidden-import", "silero_vad",
    "--hidden-import", "funasr",
    "--hidden-import", "funasr.models",
    "--hidden-import", "torch",
    "--hidden-import", "torchaudio",
    "--hidden-import", "onnxruntime",
    "--hidden-import", "soundfile",
    "--hidden-import", "librosa",
    "--hidden-import", "numpy",
    "--hidden-import", "scipy",
    "--hidden-import", "yaml",
    "--hidden-import", "modelscope",
    "--hidden-import", "huggingface_hub",
    
    # 排除不需要的模块（减小体积）
    # -- 图形界面相关 --
    "--exclude-module", "matplotlib",
    "--exclude-module", "PIL",
    "--exclude-module", "tkinter",
    "--exclude-module", "PySide6",
    "--exclude-module", "PyQt5",
    "--exclude-module", "cv2",
    # -- 开发工具 --
    "--exclude-module", "IPython",
    "--exclude-module", "jupyter",
    "--exclude-module", "notebook",
    "--exclude-module", "tensorboardx",
    # -- 未使用的 ASR 模型 --
    "--exclude-module", "openai_whisper",
    "--exclude-module", "whisper",
    # -- 未使用的功能 --
    "--exclude-module", "accelerate",
    "--exclude-module", "peft",
    "--exclude-module", "s3prl",
    # -- 聚类算法（未使用）--
    "--exclude-module", "umap",
    "--exclude-module", "hdbscan",
    "--exclude-module", "pynndescent",
    
    # 入口脚本
    "stt_server_entry.py",
]

# 添加 funasr 数据文件
if funasr_dir:
    version_txt = os.path.join(funasr_dir, "version.txt")
    if os.path.exists(version_txt):
        # --add-data "source;dest" (Windows 使用分号)
        cmd.insert(-1, "--add-data")
        cmd.insert(-1, f"{version_txt};funasr")
        print(f"添加数据文件: {version_txt}")

print("=" * 60)
print("开始打包 STT Server...")
print("=" * 60)
print(f"命令: {' '.join(cmd)}")
print()

# 执行打包
result = subprocess.run(cmd)

if result.returncode == 0:
    print()
    print("=" * 60)
    print("打包完成！")
    print(f"输出文件: {os.path.join(script_dir, 'dist', 'stt_server.exe')}")
    
    # 显示文件大小
    exe_path = os.path.join(script_dir, "dist", "stt_server.exe")
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"文件大小: {size_mb:.1f} MB")
    print("=" * 60)
else:
    print()
    print("打包失败！")
    sys.exit(1)
