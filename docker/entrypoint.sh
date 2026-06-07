#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# docker/entrypoint.sh
# ─────────────────────────────────────────────────────────────────────────────
# Script de entrada del contenedor API.
# Responsabilidades:
#   1. Esperar a que PostgreSQL esté disponible (wait-for-it pattern).
#   2. Aplicar migraciones de schema si la BD está vacía.
#   3. Verificar existencia del checkpoint del modelo (warning si falta).
#   4. Lanzar el comando pasado como $@ (uvicorn por defecto desde CMD).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colores para logs ─────────────────────────────────────────────────────────
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLUE}[ENTRYPOINT]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }

# ── Variables con defaults ────────────────────────────────────────────────────
DB_HOST="${DB_HOST:-db}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-bioacustica_fauna}"
DB_USER="${DB_USER:-bioacustica_user}"
DB_PASSWORD="${DB_PASSWORD:-changeme}"
MAX_RETRIES=30
RETRY_INTERVAL=2

# ── 1. Esperar PostgreSQL ─────────────────────────────────────────────────────
log "Esperando PostgreSQL en ${DB_HOST}:${DB_PORT} ..."

retry=0
until python -c "
import socket, sys
try:
    s = socket.create_connection(('${DB_HOST}', ${DB_PORT}), timeout=1)
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
    retry=$((retry + 1))
    if [ "$retry" -ge "$MAX_RETRIES" ]; then
        err "PostgreSQL no disponible después de ${MAX_RETRIES} intentos. Abortando."
        exit 1
    fi
    log "  Intento ${retry}/${MAX_RETRIES} — reintentando en ${RETRY_INTERVAL}s ..."
    sleep "$RETRY_INTERVAL"
done

ok "PostgreSQL disponible en ${DB_HOST}:${DB_PORT}"

# ── 2. Verificar / aplicar schema ─────────────────────────────────────────────
log "Verificando schema de base de datos ..."

SCHEMA_EXISTS=$(python -c "
import sys
try:
    import psycopg2
    conn = psycopg2.connect(
        host='${DB_HOST}',
        port=${DB_PORT},
        dbname='${DB_NAME}',
        user='${DB_USER}',
        password='${DB_PASSWORD}',
    )
    cur = conn.cursor()
    cur.execute(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_name='species'\")
    row = cur.fetchone()
    conn.close()
    print('yes' if row[0] > 0 else 'no')
except Exception as e:
    print('error:' + str(e))
" 2>&1)

if [[ "$SCHEMA_EXISTS" == "yes" ]]; then
    ok "Schema ya existente — omitiendo migración."
elif [[ "$SCHEMA_EXISTS" == "no" ]]; then
    log "Schema no encontrado — aplicando database/schema.sql ..."
    if [ -f "/app/database/schema.sql" ]; then
        PGPASSWORD="$DB_PASSWORD" psql \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            -d "$DB_NAME" \
            -f "/app/database/schema.sql" \
            && ok "Schema aplicado correctamente." \
            || { err "Fallo al aplicar schema."; exit 1; }
    else
        warn "database/schema.sql no encontrado en /app/. Saltando migración."
    fi
else
    warn "No se pudo verificar schema: ${SCHEMA_EXISTS}. Continuando de todas formas."
fi

# ── 3. Verificar checkpoint del modelo ───────────────────────────────────────
MODEL_PATH="${MODEL_CHECKPOINT_PATH:-/app/models/trained/mexico_birds/best_efficientnet.pt}"
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    ok "Checkpoint del modelo encontrado: ${MODEL_PATH} (${SIZE})"
else
    warn "Checkpoint no encontrado: ${MODEL_PATH}"
    warn "La API arrancará en modo 'sin modelo cargado'."
    warn "Para cargar un modelo, monte el volumen model_weights o"
    warn "copie el archivo a /app/models/trained/mexico_birds/best_efficientnet.pt"
fi

# ── 4. Configuración de entorno ───────────────────────────────────────────────
log "Configuración:"
log "  APP_ENV   = ${APP_ENV:-production}"
log "  MODEL     = ${MODEL_TYPE:-efficientnet} / ${MODEL_BACKBONE:-efficientnet_b0}"
log "  DEVICE    = ${MODEL_DEVICE:-cpu}"
log "  LOG_LEVEL = ${LOG_LEVEL:-INFO}"

# ── 5. Lanzar aplicación ──────────────────────────────────────────────────────
log "Iniciando: $*"
exec "$@"
