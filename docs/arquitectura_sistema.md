# Arquitectura del Sistema — BioAcoustics AI

## Pipeline de procesamiento

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SISTEMA DE IDENTIFICACIÓN BIOACÚSTICA                     │
│                     Mamíferos · Anfibios · Reptiles · Insectos               │
└─────────────────────────────────────────────────────────────────────────────┘

CAPA 1: ADQUISICIÓN
┌──────────────────────────────────────────┐
│  Dispositivos de grabación               │
│  ├─ AudioMoth (ultrasonido, 192–384 kHz) │  → Quirópteros
│  ├─ Zoom H5 / Sony PCM-A10 (96 kHz)     │  → Mamíferos + anfibios
│  └─ Raspberry Pi + INMP441 (48 kHz)     │  → Despliegue IoT en campo
│                                          │
│  Fuentes externas (API):                 │
│  ├─ Xeno-canto API v2                    │
│  ├─ iNaturalist API                      │
│  ├─ FrogID / GBIF                        │
│  └─ ChiroVox / Bat DB                   │
└─────────────────┬────────────────────────┘
                  │ WAV/MP3/FLAC
                  ▼
CAPA 2: PREPROCESAMIENTO  [src/audio_processing/preprocessor.py]
┌──────────────────────────────────────────┐
│  1. Carga + remuestreo (librosa)         │
│  2. Filtro Butterworth pasa-banda        │
│  3. Reducción de ruido (noisereduce)     │
│  4. Normalización de amplitud            │
│  5. Detección de eventos VAD             │
│  6. Segmentación (3s, 50% overlap)       │
└─────────────────┬────────────────────────┘
                  │ Segmentos numpy float32
                  ▼
CAPA 3: EXTRACCIÓN DE CARACTERÍSTICAS  [src/feature_extraction/]
┌──────────────────────────────────────────┐
│  ├─ Espectrograma Mel (128 bandas, dB)   │ → shape: (128, T)
│  ├─ MFCC + Δ + ΔΔ (40 coef.)           │ → shape: (120, T)
│  ├─ Chroma STFT (12 bandas)             │ → shape: (12, T)
│  ├─ Spectral Contrast (7 bandas)        │ → shape: (7, T)
│  └─ Features escalares:                 │
│      ZCR, SC, BW, rolloff, flatness, RMS│
└─────────────────┬────────────────────────┘
                  │ Arrays + vectores
                  ▼
CAPA 4: CLASIFICACIÓN  [src/models/]
┌──────────────────────────────────────────┐
│  Modelo A: CNN baseline (Mel input)      │  F1 objetivo: ≥ 0.75
│  Modelo B: EfficientNet-B0 + TL         │  F1 objetivo: ≥ 0.82
│  Modelo C: PANNs-CNN14 (fine-tune)      │  F1 objetivo: ≥ 0.85
│  Modelo D: AST (Audio Spectrogram TF)   │  F1 objetivo: ≥ 0.87
│                                          │
│  Output: P(especie_1,...,especie_N)      │
│  Top-K predicciones con probabilidad     │
└─────────────────┬────────────────────────┘
                  │ JSON {species, prob, rank}
                  ▼
CAPA 5: ALMACENAMIENTO  [database/schema.sql]
┌──────────────────────────────────────────┐
│  PostgreSQL 15:                          │
│  ├─ species / taxonomy (7 tablas)        │
│  ├─ recording / audio_segment            │
│  ├─ segment_label (ground truth)         │
│  ├─ ml_model / detection                 │
│  └─ dataset / dataset_split              │
└─────────────────┬────────────────────────┘
                  │
                  ▼
CAPA 6: API REST  [src/api/]
┌──────────────────────────────────────────┐
│  FastAPI + uvicorn                        │
│  ├─ POST /classify         → inferencia  │
│  ├─ GET  /detections       → listado     │
│  ├─ GET  /species/{id}     → taxonomía   │
│  ├─ POST /recordings       → upload      │
│  └─ GET  /reports/site/{id}→ biodiversidad│
└──────────────────────────────────────────┘
```

## Stack tecnológico

| Capa | Tecnología | Versión |
|------|-----------|---------|
| Lenguaje principal | Python | 3.11+ |
| Procesamiento audio | librosa, soundfile, noisereduce | latest |
| ML/DL | PyTorch + torchvision | 2.x |
| Transfer learning | timm (EfficientNet, ViT) | latest |
| Base de datos | PostgreSQL | 15+ |
| ORM | SQLAlchemy + asyncpg | 2.x |
| API | FastAPI + Pydantic | 0.100+ |
| Contenedores | Docker + docker-compose | latest |
| Experimentos | MLflow | 2.x |
| Visualización | matplotlib, plotly, librosa.display | latest |

## Estructura de directorios

```
Identificacion de mamiferos/
├── docs/
│   ├── marco_teorico/      # Marco teórico (.docx)
│   ├── estado_del_arte/    # Papers y resúmenes
│   ├── metodologia/        # Protocolos de campo y lab
│   └── bibliografia/       # Referencias BibTeX
├── src/
│   ├── audio_processing/   # Preprocesamiento (preprocessor.py)
│   ├── feature_extraction/ # Extracción de características
│   ├── models/             # Arquitecturas CNN, EfficientNet, PANNs
│   ├── api/                # FastAPI endpoints
│   ├── database/           # Modelos SQLAlchemy
│   └── utils/              # Helpers, logging, configuración
├── data/
│   ├── raw/                # Grabaciones originales (no versionar)
│   ├── processed/          # Audio normalizado/segmentado
│   ├── spectrograms/       # Espectrogramas Mel (.npy)
│   └── features/           # Vectores de características (.npy)
├── notebooks/              # Exploración y experimentos Jupyter
├── tests/                  # Tests unitarios e integración
├── configs/                # YAML de configuración por modelo
├── models/
│   ├── trained/            # Pesos finales (.pt)
│   └── checkpoints/        # Checkpoints intermedios
├── results/
│   ├── reports/            # Métricas y evaluaciones
│   ├── visualizations/     # Figuras y gráficas
│   └── metrics/            # JSON con métricas por experimento
└── database/
    └── schema.sql          # Esquema PostgreSQL
```
