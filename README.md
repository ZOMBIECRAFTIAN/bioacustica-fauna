# BioAcoustics Fauna Identification System

Sistema de identificación automática de fauna mediante **bioacústica e inteligencia artificial**. Detecta e identifica mamíferos, anfibios, reptiles e insectos a partir de grabaciones de campo usando aprendizaje profundo.

---

## Descripción

Este proyecto implementa un pipeline completo de **Monitoreo Acústico Pasivo (PAM)** con IA:

- Adquisición y preprocesamiento de audio multicanal
- Extracción de características espectrales (Mel, MFCC, ZCR, Chroma)
- Clasificación con tres arquitecturas de deep learning (CNN personalizada, EfficientNet, PANNs-CNN14)
- API REST para inferencia en tiempo real
- Monitor acústico pasivo continuo (captura + VAD + inferencia)
- Almacenamiento en PostgreSQL con esquema taxonómico completo
- Exportación a ONNX / TorchScript para deployment en edge

---

## Grupos taxonómicos soportados

| Grupo | Rango frecuencial | Sample rate | Observaciones |
|---|---|---|---|
| Chiroptera (murciélagos) | 20 – 200 kHz | 192 kHz | Pulsos ultrasónicos de ecolocalización |
| Amphibia (anuros) | 100 – 8 000 Hz | 22 050 Hz | Cantos de apareamiento |
| Insecta (ortópteros, cicadas) | 200 – 100 kHz | 44 100 Hz | Estridulación |
| Mammalia (vocal) | 20 – 20 000 Hz | 22 050 Hz | Ruidos, vocalizaciones |
| Reptilia | 100 – 5 000 Hz | 22 050 Hz | Crocodilianos, geckos |

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│  CAPA 1 — Adquisición                                           │
│  AudioMoth / SM4BAT / Zoom H5  →  .wav / .flac                  │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  CAPA 2 — Preprocesamiento  (src/audio_processing/preprocessor) │
│  Bandpass → Noise Reduction → Normalize → VAD → Segment         │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  CAPA 3 — Extracción de Características                         │
│  Mel Spectrogram · MFCC+Δ+ΔΔ · ZCR · Spectral Centroid · Chroma│
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  CAPA 4 — Clasificación                                         │
│  BioAcousticCNN · EfficientNet-B0/B4 · PANNs-CNN14             │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌──────────────────────────┬──────────────────────────────────────┐
│  CAPA 5 — Almacenamiento │  CAPA 6 — API REST                  │
│  PostgreSQL 15 + PostGIS │  FastAPI + asyncpg                  │
│  pgAdmin 4               │  /classify · /detections · /species │
└──────────────────────────┴──────────────────────────────────────┘
```

---

## Estructura del proyecto

```
.
├── src/
│   ├── audio_processing/
│   │   └── preprocessor.py          # Pipeline de audio con presets por taxón
│   ├── data/
│   │   ├── dataset_builder.py       # Descarga de Xeno-canto, iNaturalist, GBIF
│   │   └── augmentation.py          # Augmentación bioacústica avanzada
│   ├── models/
│   │   ├── cnn_baseline.py          # BioAcousticCNN con residual blocks
│   │   ├── efficientnet_classifier.py  # EfficientNet + fine-tuning progresivo
│   │   ├── panns_classifier.py      # PANNs-CNN14 preentrenado en AudioSet
│   │   └── train.py                 # Pipeline de entrenamiento unificado
│   ├── feature_extraction/
│   │   └── batch_extractor.py       # Extracción batch paralela de features
│   ├── evaluation/
│   │   └── evaluator.py             # Métricas, figuras, integración MLflow
│   ├── monitoring/
│   │   └── acoustic_monitor.py      # Monitor PAM en tiempo real
│   └── api/
│       └── main.py                  # FastAPI REST API
├── database/
│   ├── schema.sql                   # Esquema PostgreSQL 15 con taxonomía
│   └── seed.sql                     # Datos semilla
├── configs/
│   └── train_config.yaml            # Configuración de entrenamiento
├── scripts/
│   └── export_model.py              # Exportación ONNX / TorchScript
├── notebooks/
│   └── 01_eda.ipynb                 # Análisis exploratorio de datos
├── tests/
│   ├── test_preprocessor.py         # 35 tests unitarios (pytest)
│   └── test_models.py               # 50 tests de modelos
├── docker/
│   ├── entrypoint.sh                # Script de arranque del contenedor
│   └── pgadmin_servers.json         # BD preconfigurada en pgAdmin
├── docs/
│   ├── marco_teorico/               # Documento académico (.docx)
│   └── arquitectura_sistema.md      # Diagrama de arquitectura
├── Dockerfile                       # Multi-stage: builder + runtime
├── docker-compose.yml               # API + PostgreSQL + pgAdmin + MLflow
├── .env.example                     # Variables de entorno (template)
├── requirements.txt
└── .github/workflows/ci.yml         # GitHub Actions CI pipeline
```

---

## Instalación

### Requisitos del sistema

- Python 3.10 – 3.11
- libsndfile (`sudo apt install libsndfile1`)
- PostgreSQL 15 (o Docker)

### Instalación local

```bash
# 1. Clonar repositorio
git clone https://github.com/usuario/bioacoustics-fauna-id.git
cd bioacoustics-fauna-id

# 2. Entorno virtual
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. PyTorch CPU (cambiar URL para CUDA)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 4. Dependencias del proyecto
pip install -r requirements.txt
```

### Instalación con Docker (recomendada)

```bash
# Copiar y editar variables de entorno
cp .env.example .env

# Iniciar todos los servicios
docker-compose up -d

# Verificar estado
docker-compose ps
docker-compose logs -f api
```

| Servicio | URL |
|---|---|
| API REST (Swagger) | http://localhost:8000/docs |
| pgAdmin 4 | http://localhost:5050 |
| MLflow (perfil opcional) | http://localhost:5000 |

---

## Uso

### 1. Construcción del dataset

```bash
# Descargar especies por defecto (17 especies, todas las fuentes)
python -m src.data.dataset_builder \
    --output-dir data/raw \
    --workers 4

# Extracción batch de features
python -m src.feature_extraction.batch_extractor \
    --input data/raw \
    --output data/processed \
    --preset mammals \
    --workers 4
```

### 2. Entrenamiento

```bash
# Con config YAML (recomendado)
python -m src.models.train \
    --config configs/train_config.yaml

# Con argumentos CLI
python -m src.models.train \
    --config configs/train_config.yaml \
    --model efficientnet \
    --backbone efficientnet_b0 \
    --epochs 50 \
    --device cuda
```

### 3. Evaluación

```bash
python -m src.evaluation.evaluator \
    --checkpoint models/checkpoint_best.pth \
    --data-dir data/processed/test \
    --output-dir results/evaluation \
    --model-type efficientnet \
    --mlflow
```

### 4. Clasificar un archivo de audio

```bash
# Via API REST
curl -X POST http://localhost:8000/classify \
    -F "file=@grabacion.wav" \
    -F "top_k=5"

# Via monitor (archivo)
python -m src.monitoring.acoustic_monitor file \
    --checkpoint models/best.pth \
    --class-names '["bat","frog","cricket"]' \
    --input grabacion.wav \
    --output resultados.json
```

### 5. Monitor en tiempo real

```bash
# Listar dispositivos de audio disponibles
python -m src.monitoring.acoustic_monitor devices

# Iniciar captura continua
python -m src.monitoring.acoustic_monitor run \
    --checkpoint models/best.pth \
    --class-names class_names.txt \
    --device-index 0 \
    --sample-rate 22050 \
    --site-id <UUID-del-sitio> \
    --model-id <UUID-del-modelo>
```

### 6. Exportar modelo para deployment

```bash
# ONNX + TorchScript con benchmarking
python scripts/export_model.py \
    --checkpoint models/checkpoint_best.pth \
    --model-type efficientnet \
    --n-classes 17 \
    --output-dir models/exported \
    --formats torchscript onnx \
    --benchmark

# Cuantización INT8
python scripts/export_model.py \
    --checkpoint models/checkpoint_best.pth \
    --model-type efficientnet \
    --n-classes 17 \
    --formats onnx_int8 \
    --calibration-data data/processed/mammals
```

---

## Tests

```bash
# Tests de preprocesamiento
pytest tests/test_preprocessor.py -v

# Tests de modelos
pytest tests/test_models.py -v

# Suite completa con cobertura
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
```

---

## Modelos disponibles

| Modelo | Parámetros | Descripción |
|---|---|---|
| `BioAcousticCNN` | ~2M | CNN residual con atención temporal. Entrenamiento desde cero. |
| `EfficientNetBioAcoustic` | 5–19M | EfficientNet-B0/B4 con fine-tuning progresivo en 3 fases. |
| `PANNSCNN14BioAcoustic` | ~80M | CNN14 preentrenado en AudioSet (526 clases). Transfer learning. |

### Configuración de entrenamiento por fase (EfficientNet / PANNs)

| Fase | Capas descongeladas | Learning rate | Épocas |
|---|---|---|---|
| 1 | Solo cabeza (head) | 1e-3 | 10 |
| 2 | Últimos 3 bloques | 1e-4 | 15 |
| 3 | Todo el backbone | 5e-5 | 25 |

---

## API REST — Endpoints principales

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/classify` | Clasificar archivo de audio (multipart/form-data) |
| `POST` | `/classify/url` | Clasificar audio desde URL |
| `GET` | `/detections` | Listar detecciones (con filtros) |
| `GET` | `/detections/{id}` | Detección por ID |
| `GET` | `/species` | Catálogo de especies |
| `GET` | `/species/{id}` | Especie por ID |
| `POST` | `/recordings` | Registrar grabación nueva |
| `GET` | `/reports/site/{id}` | Reporte de biodiversidad por sitio |
| `GET` | `/health` | Estado del servicio y modelo |
| `GET` | `/models` | Información del modelo activo |

Documentación interactiva: `http://localhost:8000/docs` (Swagger UI)

---

## Augmentación de datos

El módulo `src/data/augmentation.py` implementa transformaciones específicas para audio bioacústico:

**Dominio temporal:** PitchShift · TimeStretch · AddGaussianNoise · AddBackgroundNoise · RandomClip · VolumeJitter · TimeShift

**Dominio espectral:** SpecAugmentFreq · SpecAugmentTime · FrequencyShift · RandomErasing · GaussianBlur

Presets predefinidos por grupo taxonómico: `get_preset("bats" | "frogs" | "insects" | "mammals" | "reptiles")`

---

## Fuentes de datos

| Fuente | API | Taxones cubiertos |
|---|---|---|
| Xeno-canto | v2 REST | Aves, anfibios |
| iNaturalist | v1 REST | Todos los grupos |
| GBIF | Occurrence API | Todos los grupos |
| FrogID | Dataset descargable | Anfibios (Australia) |
| ChiroVox | Dataset descargable | Murciélagos |

---

## Configuración de entrenamiento (YAML)

```yaml
# configs/train_config.yaml — fragmento
model:
  type: efficientnet          # cnn_baseline | efficientnet | panns
  backbone: efficientnet_b0
  n_classes: 17
  dropout: 0.3
  pretrained: true

training:
  phases:
    - name: head_only
      epochs: 10
      lr: 1.0e-3
    - name: fine_tune_last3
      epochs: 15
      lr: 1.0e-4
    - name: full_backbone
      epochs: 25
      lr: 5.0e-5

audio:
  sample_rate: 22050
  n_mels: 128
  n_fft: 2048
  segment_duration: 3.0
```

---

## CI/CD

El pipeline de GitHub Actions (`.github/workflows/ci.yml`) ejecuta en cada push/PR:

1. **Lint** — ruff + black + isort
2. **Tests** — pytest con cobertura (Python 3.10 y 3.11)
3. **Security** — bandit + safety
4. **Schema** — validación de schema.sql sobre PostgreSQL real
5. **Docker** — build de imagen sin push

---

## Referencias

- Kong, Q. et al. (2020). *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition*. IEEE/ACM TASLP. https://doi.org/10.1109/TASLP.2020.3030497
- Park, D. et al. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*. INTERSPEECH. https://doi.org/10.21437/Interspeech.2019-2680
- Tan, M. & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.
- Sueur, J. et al. (2008). *Soundscape Ecology: The Science of Sound in the Landscape*. Bioscience.
- Pijanowski, B. C. et al. (2011). *Soundscape Ecology: The Science of Sound in the Landscape*. BioScience, 61(3), 203–216.

---

## Licencia

MIT License — ver `LICENSE` para detalles.

---

## Autor

**Ian** — Ingeniero en Sistemas Computacionales / Investigador en Bioacústica
`brianferbaez@gmail.com`
