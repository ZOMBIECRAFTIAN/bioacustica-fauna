@echo off
setlocal
cd /d "%~dp0\.."

echo Bioacustica Fauna -- pipeline de maestria multitaxon
echo.

echo [1/5] Scientific readiness
python scripts\verify_scientific_readiness.py
if errorlevel 1 exit /b 1

echo.
echo [2/5] Download dataset piloto
echo Nota: Xeno-canto requiere XENO_CANTO_API_KEY para descargas completas.
python -m src.data.dataset_builder --profile mexico_multitaxon --output data/raw/multitaxon --max-per-class 150
if errorlevel 1 exit /b 1

echo.
echo [3/5] Extract adaptive spectrograms + manifest
python -m src.feature_extraction.batch_extractor --input data/raw/multitaxon --output data/spectrograms/multitaxon --mfcc-dir data/features/mfcc/multitaxon --preset adaptive --workers 1
if errorlevel 1 exit /b 1

echo.
echo [4/5] Refresh dataset_manifest.csv
python -m src.data.manifest --raw-dir data/raw/multitaxon --spectrogram-dir data/spectrograms/multitaxon
if errorlevel 1 exit /b 1

echo.
echo [5/5] Train short CPU pilot
python -m src.models.train --config configs/train_multitaxon.yaml --device cpu --epochs 3
if errorlevel 1 exit /b 1

echo.
echo Pipeline finished. Review results/maestria_multitaxon.
