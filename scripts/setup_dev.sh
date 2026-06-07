#!/usr/bin/env bash
# =============================================================================
# scripts/setup_dev.sh -- Bioacustica Fauna
# Setup automatizado para Linux / macOS
#
# Uso:
#   bash scripts/setup_dev.sh
#   bash scripts/setup_dev.sh --no-torch-gpu
#   bash scripts/setup_dev.sh --skip-db
# =============================================================================

set -euo pipefail

# -- Colores ------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

# -- Configuracion ------------------------------------------------------------
PROJECT_NAME="bioacustica-fauna"
VENV_DIR=".venv"
PYTHON_MIN="3.10"
TORCH_CPU_URL="https://download.pytorch.org/whl/cpu"

# -- Argumentos ---------------------------------------------------------------
NO_GPU=0
SKIP_DB=0
SKIP_TORCH=0

for arg in "$@"; do
    case $arg in
        --no-torch-gpu) NO_GPU=1 ;;
        --skip-db)      SKIP_DB=1 ;;
        --skip-torch)   SKIP_TORCH=1 ;;
    esac
done

# -- Funciones ----------------------------------------------------------------
log_step()  { echo -e "\n${GREEN}[$1/$TOTAL_STEPS] $2${RESET}"; }
log_ok()    { echo -e "  ${GREEN}OK:${RESET} $1"; }
log_warn()  { echo -e "  ${YELLOW}WARN:${RESET} $1"; }
log_error() { echo -e "  ${RED}ERROR:${RESET} $1"; exit 1; }
log_skip()  { echo -e "  ${YELLOW}SKIP:${RESET} $1"; }

TOTAL_STEPS=7

# =============================================================================
echo -e "\n${BOLD}============================================================${RESET}"
echo -e "${BOLD}  Bioacustica Fauna -- Setup de Desarrollo${RESET}"
echo -e "${BOLD}  OS: $(uname -s) $(uname -m)${RESET}"
echo -e "${BOLD}============================================================${RESET}"

# =============================================================================
# PASO 1: Verificar Python
# =============================================================================
log_step 1 "Verificando Python..."

# Detectar python3 o python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    log_error "Python no encontrado. Instala Python >= ${PYTHON_MIN}."
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

echo "  Python version: $PY_VERSION"

if [[ $PY_MAJOR -lt 3 ]] || [[ $PY_MAJOR -eq 3 && $PY_MINOR -lt 10 ]]; then
    log_error "Se requiere Python >= ${PYTHON_MIN}. Version actual: ${PY_VERSION}"
fi
log_ok "Python ${PY_VERSION}"

# =============================================================================
# PASO 2: Dependencias del sistema
# =============================================================================
log_step 2 "Verificando dependencias del sistema..."

OS=$(uname -s)

install_system_deps() {
    if [[ "$OS" == "Linux" ]]; then
        if command -v apt-get &>/dev/null; then
            echo "  Instalando via apt-get..."
            sudo apt-get update -qq
            sudo apt-get install -y --no-install-recommends \
                libsndfile1 libsndfile1-dev \
                portaudio19-dev \
                libpq-dev \
                ffmpeg \
                git 2>/dev/null || true
        elif command -v yum &>/dev/null; then
            sudo yum install -y libsndfile portaudio-devel postgresql-devel ffmpeg git 2>/dev/null || true
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            echo "  Instalando via Homebrew..."
            brew install libsndfile portaudio postgresql ffmpeg 2>/dev/null || true
        else
            log_warn "Homebrew no encontrado. Instala manualmente: libsndfile, portaudio, ffmpeg"
        fi
    fi
}

# Solo instalar si libsndfile no existe
if ! $PYTHON -c "import soundfile" &>/dev/null; then
    install_system_deps
else
    log_ok "libsndfile disponible"
fi

# =============================================================================
# PASO 3: Entorno virtual
# =============================================================================
log_step 3 "Configurando entorno virtual..."

if [[ -d "${VENV_DIR}/bin" ]] || [[ -d "${VENV_DIR}/Scripts" ]]; then
    log_warn "Entorno virtual existente en ${VENV_DIR}/ -- reutilizando"
else
    echo "  Creando entorno virtual en ${VENV_DIR}/..."
    $PYTHON -m venv "$VENV_DIR"
    log_ok "Entorno virtual creado"
fi

# Activar
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate" 2>/dev/null || source "${VENV_DIR}/Scripts/activate"
log_ok "Entorno activado: $VIRTUAL_ENV"

# Actualizar pip
pip install --upgrade pip --quiet
log_ok "pip actualizado"

# =============================================================================
# PASO 4: PyTorch
# =============================================================================
log_step 4 "Instalando PyTorch..."

if [[ $SKIP_TORCH -eq 1 ]]; then
    log_skip "--skip-torch especificado"
elif python -c "import torch" &>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    log_ok "PyTorch ya instalado: v${TORCH_VER}"
else
    if [[ $NO_GPU -eq 1 ]]; then
        echo "  Instalando PyTorch CPU-only..."
        pip install torch torchvision torchaudio --index-url "$TORCH_CPU_URL" --quiet
    elif command -v nvidia-smi &>/dev/null; then
        # Detectar version de CUDA
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        if [[ -n "$CUDA_VER" ]]; then
            CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
            CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
            echo "  CUDA ${CUDA_VER} detectado."
            if [[ $CUDA_MAJOR -ge 12 ]]; then
                TORCH_URL="https://download.pytorch.org/whl/cu121"
            else
                TORCH_URL="https://download.pytorch.org/whl/cu118"
            fi
            echo "  Instalando PyTorch con GPU (${TORCH_URL})..."
            pip install torch torchvision torchaudio --index-url "$TORCH_URL" --quiet
        else
            log_warn "nvidia-smi disponible pero no se pudo leer version CUDA. Usando CPU."
            pip install torch torchvision torchaudio --index-url "$TORCH_CPU_URL" --quiet
        fi
    else
        log_warn "GPU no detectada. Instalando PyTorch CPU-only."
        pip install torch torchvision torchaudio --index-url "$TORCH_CPU_URL" --quiet
    fi

    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    log_ok "PyTorch instalado: v${TORCH_VER}"
fi

# =============================================================================
# PASO 5: Dependencias del proyecto
# =============================================================================
log_step 5 "Instalando dependencias del proyecto..."

pip install -r requirements.txt --quiet
log_ok "requirements.txt instalado"

# Herramientas de desarrollo
pip install ruff black pytest pytest-cov --quiet
log_ok "Herramientas de desarrollo instaladas (ruff, black, pytest)"

# =============================================================================
# PASO 6: Variables de entorno
# =============================================================================
log_step 6 "Configurando variables de entorno..."

if [[ -f ".env" ]]; then
    log_ok ".env ya existe -- omitiendo copia"
elif [[ -f ".env.example" ]]; then
    cp .env.example .env
    log_ok ".env creado desde .env.example"
    log_warn "Edita .env con tus credenciales antes de continuar."
else
    log_warn ".env.example no encontrado. Crea .env manualmente."
fi

# =============================================================================
# PASO 7: Base de datos (opcional)
# =============================================================================
log_step 7 "Base de datos..."

if [[ $SKIP_DB -eq 1 ]]; then
    log_skip "--skip-db especificado"
elif command -v psql &>/dev/null; then
    echo "  PostgreSQL detectado. Inicializando schema..."
    psql -U postgres -c "CREATE DATABASE bioacoustics;" 2>/dev/null || true
    psql -U postgres -c "CREATE USER biouser WITH PASSWORD 'biopassword';" 2>/dev/null || true
    psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE bioacoustics TO biouser;" 2>/dev/null || true
    psql -U biouser -d bioacoustics -f database/schema.sql 2>/dev/null \
        && log_ok "Schema aplicado" \
        || log_warn "Error al aplicar schema.sql (puede que ya exista)"
else
    log_warn "psql no encontrado -- omitiendo inicializacion de DB."
    echo "  Instala PostgreSQL 15 y ejecuta:"
    echo "    psql -U postgres -f database/schema.sql"
fi

# =============================================================================
# Verificacion final
# =============================================================================
echo -e "\n${BOLD}Verificando instalacion...${RESET}"
python scripts/verify_install.py && log_ok "Todos los checks pasaron" \
    || log_warn "Algunos checks fallaron. Revisa los errores arriba."

# =============================================================================
echo -e "\n${BOLD}============================================================${RESET}"
echo -e "${BOLD}  Setup completado${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo -e "
  Para activar el entorno en nuevas sesiones:
    ${YELLOW}source .venv/bin/activate${RESET}        (Linux/macOS)
    ${YELLOW}.venv\\Scripts\\activate${RESET}          (Windows)

  Para correr tests:
    ${YELLOW}python -m pytest tests/ -v${RESET}

  Para iniciar la API:
    ${YELLOW}python -m uvicorn src.api.main:app --reload --port 8000${RESET}

  Swagger UI: http://localhost:8000/docs

  Documentacion:
    ${YELLOW}INSTALL.md${RESET}   -- Guia de instalacion completa
    ${YELLOW}README.md${RESET}    -- Documentacion del proyecto
"
