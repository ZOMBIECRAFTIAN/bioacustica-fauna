@echo off
setlocal
cd /d "%~dp0\.."

echo [1/5] Black
black --check --diff src tests scripts
if errorlevel 1 exit /b 1

echo [2/5] Ruff
ruff check --no-cache src tests scripts --output-format=github
if errorlevel 1 exit /b 1

echo [3/5] Compile
python -m py_compile src\data\manifest.py src\feature_extraction\batch_extractor.py src\models\train.py src\models\cnn_baseline.py tests\test_manifest_and_splits.py scripts\verify_scientific_readiness.py
if errorlevel 1 exit /b 1

echo [4/5] Targeted tests
set TMPPYTEST=.tmp_pytest_%RANDOM%_%RANDOM%
pytest -q -o addopts="" -p no:cacheprovider --basetemp %TMPPYTEST% tests\test_manifest_and_splits.py
if errorlevel 1 exit /b 1

echo [5/5] Scientific readiness
python scripts\verify_scientific_readiness.py
if errorlevel 1 exit /b 1

echo All local checks passed.
