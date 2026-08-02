@echo off
setlocal
:: 获取当前文件夹路径
set ROOT=%~dp0
set BIN_PATH=%ROOT%bin
:: 指向你的 Python 环境路径（conda: videoLingo20260312）
set PY_PATH=D:\Users\xuyuwei\miniconda3\envs\videoLingo20260312
set VL_PATH=%ROOT%VideoLingo

:: 注入环境变量，让系统优先找 bin 里的 ffmpeg
set PATH=%BIN_PATH%;%PY_PATH%;%PY_PATH%\Scripts;%PATH%

:: 强制设置模型存放位置，防止它跑去 C 盘
set HF_HOME=%ROOT%models
set XDG_CACHE_HOME=%ROOT%models

cd /d %VL_PATH%
echo VideoLingo is starting...
"%PY_PATH%\python.exe" -m batch.utils.batch_subtitle
pause