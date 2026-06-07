@echo off
REM =============================================================================
REM scripts/setup_dev.bat -- Bioacustica Fauna
REM Setup automatizado para Windows (PowerShell/CMD)
REM
REM Uso:
REM   scripts\setup_dev.bat
REM   scripts\setup_dev.bat --no-torch-gpu   (fuerza CPU-only)
REM   scripts\setup_dev.bat --skip-db         (omite inicializacion DB)
REM =============================================================================

setlocal EnableDelayedExpansion

REM -- Configuracion -------------------------------------------------------
set PROJECT_NAME=bioacustica-fauna
set VENV_DIR=.venv
set PYTHON_MIN=3.10
set TORCH_CPU_URL=https://download.pytorch.org/whl/cpu

REM -- Colores ANSI (requiere Windows 10+) ---------------------------------
set GREEN=[92m
set YELLOW=[93m
set RED=[91m
set RESET=[0m
set BOLD=[1m

REM -- Argumentos ----------------------------------------------------------
set NO_GPU=0
set SKIP_DB=0
set SKIP_TORCH=0

:parse_args
if "%~1"=="" goto :main
if "%~1"=="--no-torch-gpu" set NO_GPU=1
if "%~1"=="--skip-db" set SKIP_DB=1
if "%~1"=="--skip-torch" set SKIP_TORCH=1
shift
goto :parse_args

:main
echo.
echo %BOLD%============================================================%RESET%
echo %BOLD%  Bioacustica Fauna -- Setup de Desarrollo (Windows)%RESET%
echo %BOLD%============================================================%RESET%
echo.

REM =========================================================================
REM PASO 1: Verificar Python
REM =========================================================================
echo %GREEN%[1/7] Verificando Python...%RESET%

where python >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: Python no encontrado en PATH.%RESET%
    echo Descarga Python 3.11 desde https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VERSION=%%v
echo   Python version: %PY_VERSION%

REM =========================================================================
REM PASO 2: Crear entorno virtual
REM =========================================================================
echo %GREEN%[2/7] Configurando entorno virtual...%RESET%

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo   Entorno virtual existente detectado en %VENV_DIR%\
) else (
    echo   Creando entorno virtual en %VENV_DIR%\...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo %RED%ERROR: No se pudo crear el entorno virtual.%RESET%
        exit /b 1
    )
)

call %VENV_DIR%\Scripts\activate.bat
echo   Entorno activado: %VIRTUAL_ENV%

REM Actualizar pip
python -m pip install --upgrade pip --quiet
echo   pip actualizado.

REM =========================================================================
REM PASO 3: Instalar PyTorch
REM =========================================================================
echo %GREEN%[3/7] Instalando PyTorch...%RESET%

if "%SKIP_TORCH%"=="1" (
    echo   %YELLOW%SKIP: --skip-torch especificado.%RESET%
    goto :step4
)

REM Verificar si ya esta instalado
python -c "import torch; print('  PyTorch ya instalado: v' + torch.__version__)" 2>nul
if not errorlevel 1 goto :step4

if "%NO_GPU%"=="1" (
    echo   Instalando PyTorch CPU-only...
    pip install torch torchvision torchaudio --index-url %TORCH_CPU_URL% --quiet
) else (
    REM Detectar CUDA
    where nvcc >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=5" %%v in ('nvcc --version ^| findstr "release"') do set CUDA_VER=%%v
        echo   CUDA detectado: !CUDA_VER!
        echo   Instalando PyTorch con soporte GPU (CUDA 11.8)...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
    ) else (
        echo   %YELLOW%CUDA no detectado. Instalando PyTorch CPU-only...%RESET%
        pip install torch torchvision torchaudio --index-url %TORCH_CPU_URL% --quiet
    )
)

python -c "import torch; print('  PyTorch instalado: v' + torch.__version__)"
if errorlevel 1 (
    echo %RED%ERROR: Fallo la instalacion de PyTorch.%RESET%
    exit /b 1
)

:step4
REM =========================================================================
REM PASO 4: Instalar dependencias del proyecto
REM =========================================================================
echo %GREEN%[4/7] Instalando dependencias del proyecto...%RESET%

pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo %RED%ERROR: Fallo la instalacion de requirements.txt%RESET%
    exit /b 1
)

REM Herramientas de desarrollo
pip install ruff black pytest pytest-cov --quiet
echo   Dependencias instaladas correctamente.

REM =========================================================================
REM PASO 5: Configurar variables de entorno
REM =========================================================================
echo %GREEN%[5/7] Configurando variables de entorno...%RESET%

if exist ".env" (
    echo   .env ya existe -- omitiendo copia.
) else (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo   .env creado desde .env.example
        echo   %YELLOW%IMPORTANTE: Edita .env con tus credenciales antes de continuar.%RESET%
    ) else (
        echo   %YELLOW%ADVERTENCIA: .env.example no encontrado. Crea .env manualmente.%RESET%
    )
)

REM =========================================================================
REM PASO 6: Inicializar base de datos (opcional)
REM =========================================================================
echo %GREEN%[6/7] Base de datos...%RESET%

if "%SKIP_DB%"=="1" (
    echo   %YELLOW%SKIP: --skip-db especificado.%RESET%
    goto :step7
)

where psql >nul 2>&1
if errorlevel 1 (
    echo   %YELLOW%psql no encontrado en PATH -- omitiendo inicializacion de DB.%RESET%
    echo   Instala PostgreSQL 15 y ejecuta manualmente:
    echo     psql -U postgres -f database\schema.sql
    goto :step7
)

echo   PostgreSQL detectado. Inicializando schema...
psql -U postgres -c "CREATE DATABASE bioacoustics;" 2>nul
psql -U postgres -c "CREATE USER biouser WITH PASSWORD 'biopassword';" 2>nul
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE bioacoustics TO biouser;" 2>nul
psql -U biouser -d bioacoustics -f database\schema.sql
if errorlevel 1 (
    echo   %YELLOW%ADVERTENCIA: Error al aplicar schema.sql. Puede que ya exista.%RESET%
) else (
    echo   Schema aplicado correctamente.
)

:step7
REM =========================================================================
REM PASO 7: Verificar instalacion
REM =========================================================================
echo %GREEN%[7/7] Verificando instalacion...%RESET%

python scripts\verify_install.py
if errorlevel 1 (
    echo %RED%ADVERTENCIA: Algunos checks fallaron. Revisa los errores arriba.%RESET%
) else (
    echo   %GREEN%Todos los checks pasaron.%RESET%
)

REM =========================================================================
REM RESUMEN
REM =========================================================================
echo.
echo %BOLD%============================================================%RESET%
echo %BOLD%  Setup completado%RESET%
echo %BOLD%============================================================%RESET%
echo.
echo  Para activar el entorno en nuevas sesiones:
echo    %YELLOW%.venv\Scripts\activate%RESET%
echo.
echo  Para correr tests:
echo    %YELLOW%python -m pytest tests\ -v%RESET%
echo.
echo  Para iniciar la API:
echo    %YELLOW%python -m uvicorn src.api.main:app --reload --port 8000%RESET%
echo.
echo  Documentacion:
echo    %YELLOW%INSTALL.md%RESET%     -- Guia de instalacion completa
echo    %YELLOW%README.md%RESET%      -- Documentacion del proyecto
echo    %YELLOW%docs\%RESET%          -- Documentacion tecnica
echo.

endlocal
