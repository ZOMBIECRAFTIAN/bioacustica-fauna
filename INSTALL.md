# Bioacustica Fauna - Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11 or 3.12 |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB | 20 GB |
| OS | Windows 10, Ubuntu 20.04, macOS 12 | Ubuntu 22.04 |
| GPU | none (CPU mode) | NVIDIA CUDA 11.8+ |

---

## Option A: Local Installation (venv)

### 1. Clone the repository

```bash
git clone https://github.com/ZOMBIECRAFTIAN/bioacustica-fauna.git
cd bioacustica-fauna
```

### 2. Create virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch (BEFORE requirements.txt)

**CPU only (all platforms):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install project dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Database
POSTGRES_DB=bioacustica_fauna
POSTGRES_USER=bioacustica_user
POSTGRES_PASSWORD=your_secure_password_here
DATABASE_URL=postgresql://bioacustica_user:your_secure_password_here@localhost:5432/bioacustica_fauna

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Models
MODEL_TYPE=cnn_baseline
MODEL_PATH=models/trained/mexico_birds/best_model.pt
MODEL_CHECKPOINT_PATH=/app/models/trained/mexico_birds/best_model.pt
MODEL_DEVICE=cpu
```

### 6. Initialize the database

Requires PostgreSQL 15 running locally.

```bash
# Create database
psql -U postgres -c "CREATE DATABASE bioacustica_fauna;"
psql -U postgres -c "CREATE USER bioacustica_user WITH PASSWORD 'your_password';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE bioacustica_fauna TO bioacustica_user;"

# Apply schema
psql -U bioacustica_user -d bioacustica_fauna -f database/schema.sql
psql -U bioacustica_user -d bioacustica_fauna -f database/seed.sql
```

### 7. Verify installation

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests pass (48+ passing).

### 8. Run the API

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API available at: `http://localhost:8000`
Docs (Swagger): `http://localhost:8000/docs`

---

## Option B: Docker (recommended for deployment)

Requires: Docker >= 24.0 and Docker Compose >= 2.20

### 1. Clone and configure

```bash
git clone https://github.com/ZOMBIECRAFTIAN/bioacustica-fauna.git
cd bioacustica-fauna
cp .env.example .env
# Edit .env with your passwords
```

### 2. Start all services

```bash
docker compose up -d
```

Services started:
- `db` — PostgreSQL 15 + PostGIS on port 5432
- `api` — FastAPI on port 8000
- `pgadmin` — Database admin UI on port 5050

### 3. Verify

```bash
# Check all containers are running
docker compose ps

# Check API health
curl http://localhost:8000/health

# View logs
docker compose logs api --follow
```

### 4. Stop

```bash
docker compose down          # stop, keep data volumes
docker compose down -v       # stop and delete all data
```

### Optional: Start with MLflow

```bash
docker compose --profile mlflow up -d
# MLflow UI: http://localhost:5000
```

---

## Option C: Development Setup

For active development with hot reload and test watching.

```bash
# After completing Option A steps 1-6:
pip install ruff black pytest pytest-cov pre-commit

# Install pre-commit hooks (auto-lint on git commit)
pre-commit install

# Run tests with coverage
python -m pytest tests/ -v --cov=src --cov-report=html
# Coverage report: htmlcov/index.html

# Start API with hot reload
python -m uvicorn src.api.main:app --reload --port 8000
```

---

## Windows-Specific Notes

### PyAudio (real-time recording)

PyAudio requires PortAudio. On Windows:

```powershell
# Option 1: direct pip (Python 3.13 compatible)
pip install pyaudio

# Option 2: if that fails, use pipwin (Python <= 3.11 only)
pip install pipwin
pipwin install pyaudio
```

Note: `pipwin` is incompatible with Python 3.12+. Use direct `pip install pyaudio` on Python 3.12/3.13.

### Anaconda conflict

If you have Anaconda installed, always use `python -m pytest` instead of bare `pytest` to ensure the venv interpreter is used:

```powershell
# Correct:
python -m pytest tests/ -v

# May use wrong Python if Anaconda is on PATH:
pytest tests/ -v
```

### Long path support

If you get path-length errors on Windows:
```
# Run as Administrator in PowerShell:
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```

---

## Verifying the Installation

Run this script to check all critical dependencies:

```bash
python -c "
import sys
print(f'Python: {sys.version}')
import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')
import librosa; print(f'librosa: {librosa.__version__}')
import soundfile; print(f'soundfile: {soundfile.__version__}')
import numpy; print(f'numpy: {numpy.__version__}')
import fastapi; print(f'fastapi: {fastapi.__version__}')
import sqlalchemy; print(f'sqlalchemy: {sqlalchemy.__version__}')
print('All critical dependencies OK')
"
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: torch` | PyTorch not installed | Run `pip install torch ... --index-url ...` |
| `librosa.feature.spectral_centroid() takes 0 positional arguments` | librosa >= 0.10 API change | Update to latest preprocessor.py |
| `psycopg2.OperationalError: could not connect` | PostgreSQL not running | `pg_ctlcluster 15 main start` (Linux) or start PostgreSQL service |
| `pipwin RuntimeError` on Python 3.12+ | js2py bytecode incompatibility | Use `pip install pyaudio` directly |
| `pytest` uses wrong Python | Anaconda PATH conflict | Use `python -m pytest` always |
| CI: `torch==2.2.2 not found` | Version not available for Python 3.13 | Remove version pin, use `pip install torch` |

---

## Updating

```bash
git pull origin main
pip install -r requirements.txt
# If database schema changed:
psql -U bioacustica_user -d bioacustica_fauna -f database/schema.sql
```

For Docker:
```bash
git pull origin main
docker compose build --no-cache api
docker compose up -d
```
