@echo off
cd /D "%~dp0"
cd ..

call conda activate videoLingo20260312

@rem 运行批处理脚本
call python batch\utils\batch_processor.py

:end
pause
