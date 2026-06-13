@echo off
setlocal
cd /d "%~dp0\.."

set MODEL_PATH=models\trained\multitaxon\best_efficientnet.pt
set MODEL_TYPE=efficientnet
set MODEL_DEVICE=cpu
set APP_HOST=0.0.0.0
set APP_PORT=8000

python -m uvicorn src.api.main:app --host %APP_HOST% --port %APP_PORT% --reload
