# -*- coding: utf-8 -*-
"""
PyInstaller Hook for FunASR
用于正确处理 FunASR 的动态模型加载机制
"""

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    get_module_file_attribute,
)

# 收集所有 FunASR 子模块
hiddenimports = collect_submodules('funasr')

# 添加关键的隐式导入
hiddenimports.extend([
    'funasr.auto',
    'funasr.auto.auto_model',
    'funasr.models',
    'funasr.models.sensevoice',
    'funasr.models.sensevoice.model',
    'funasr.models.paraformer',
    'funasr.models.paraformer.model',
    'funasr.models.paraformer.model_origin',
    'funasr.models.fsmn_vad',
    'funasr.models.fsmn_vad.model',
    'funasr.models.ct_transformer',
    'funasr.models.ct_transformer.model',
    'funasr.frontends',
    'funasr.frontends.fbank',
    'funasr.frontends.wav_frontend',
    'funasr.metrics',
    'funasr.metrics.metrics',
    'funasr.utils',
    'funasr.utils.load_utils',
    'funasr.utils.postprocess_utils',
])

# 收集数据文件
datas = collect_data_files('funasr')

# 添加关键的数据文件
try:
    funasr_path = get_module_file_attribute('funasr')
    if funasr_path:
        import os
        funasr_dir = os.path.dirname(funasr_path)
        # 添加 version.txt
        version_file = os.path.join(funasr_dir, 'version.txt')
        if os.path.exists(version_file):
            datas.append((version_file, 'funasr'))
except Exception:
    pass

