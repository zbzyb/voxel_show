
@echo off
echo server start...

REM 设置环境变量
set PYTHONPATH=%~dp0
set UV_PYTHON_PREFERENCE=only-managed
set PYTHONUNBUFFERED=1

REM 切换到项目目录（已在PYTHONPATH中设置）

REM 启动应用
echo use uvpython start...
"%~dp0uvpython\uv.exe" run app.py

pause