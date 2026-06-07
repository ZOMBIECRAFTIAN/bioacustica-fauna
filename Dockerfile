# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Bioacustica Fauna API
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build:
#   Stage 1 (builder): instala dependencias Python en un venv aislado.
#   Stage 2 (runtime): imagen slim con solo lo necesario para producción.
#
# Build:
#   docker build -t bioacustica-fauna-api:latest .
#
# Run standalone (sin docker-compose):
#   docker run -p 8000:8000 --env-file .env bioacustica-fauna-api:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Dependencias de sistema necesarias para compilar paquetes C/C++
# (librosa → numba/llvmlite, soundfile → libsndfile, psycopg2 → libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual explícito → más limpio que site-packages global
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Actualizar pip y setuptools antes de instalar dependencias
RUN pip install --upgrade pip setuptools wheel

# Copiar solo requirements para aprovechar layer cache de Docker
COPY requirements.txt /tmp/requirements.txt

# Instalar dependencias Python
# torch CPU-only para inferencia en contenedor sin GPU disponible.
# Si el host tiene GPU CUDA, cambiar el índice a https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r /tmp/requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Ian <brianferbaez@gmail.com>"
LABEL description="Bioacustica Fauna API -- FastAPI + PyTorch"
LABEL version="1.0.0"

# Librerías de sistema runtime (sin dev-headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar venv del builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Variables de entorno por defecto ─────────────────────────────────────────
# Todas sobreescribibles vía .env o docker-compose environment:
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=1 \
    # Base de datos
    DB_HOST=db \
    DB_PORT=5432 \
    DB_NAME=bioacustica_fauna \
    DB_USER=bioacustica_user \
    DB_PASSWORD=changeme \
    # MLflow (opcional)
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    # Modelo
    MODEL_CHECKPOINT_PATH=/app/models/trained/mexico_birds/best_model.pt \
    MODEL_TYPE=cnn_baseline \
    MODEL_BACKBONE="" \
    MODEL_DEVICE=cpu \
    MODEL_TOP_K=5 \
    # Logging
    LOG_LEVEL=INFO

# ── Directorio de trabajo ─────────────────────────────────────────────────────
WORKDIR /app

# Crear estructura de directorios que la app espera
RUN mkdir -p \
    /app/models \
    /app/models/trained/mexico_birds \
    /app/data/raw \
    /app/data/processed \
    /app/results/logs \
    /app/results/visualizations

# Copiar código fuente
# Se excluye lo que está en .dockerignore (tests, notebooks, datos crudos, etc.)
COPY src/         /app/src/
COPY configs/     /app/configs/
COPY database/    /app/database/

# Script de entrada: espera PostgreSQL y lanza uvicorn
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Puerto expuesto
EXPOSE 8000

# Healthcheck: FastAPI tiene /health endpoint
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:${APP_PORT}/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
