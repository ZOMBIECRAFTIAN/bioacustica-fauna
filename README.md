# Bioacustica Fauna

> Automatic wildlife identification through passive acoustic monitoring and deep learning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Abstract

This project implements an end-to-end **Passive Acoustic Monitoring (PAM)** pipeline for automatic biodiversity assessment from field recordings. It classifies wildlife acoustic signals using deep learning on log-mel spectrograms, targeting multiple taxonomic groups: Chiroptera (bats), Anura (frogs), Aves (birds), Orthoptera/Cicadidae (insects), terrestrial mammals, and reptiles. The system integrates taxon-adaptive bioacoustic preprocessing, three deep learning architectures (custom residual CNN, EfficientNet-B0/B4, PANNs-CNN14), a REST API for real-time inference, PostgreSQL/PostGIS for structured biodiversity data storage, and ONNX/TorchScript export for edge deployment on low-power field devices.

---

## Scientific Problem

Traditional wildlife monitoring relies on visual surveys and expert-dependent manual acoustic analysis — both are time-intensive, geographically constrained, and non-scalable. Passive acoustic monitoring (PAM) with automated species identification offers a scalable complement: low-cost recorders (AudioMoth, SM4BAT) capture audio continuously, and AI classifies the resulting recordings without expert intervention.

Key open challenges addressed by this project:

- **Multi-group classification across extreme frequency ranges** (20 Hz – 200 kHz), requiring adaptive preprocessing per taxonomic group.
- **Transfer learning with limited labeled data**, particularly for Latin American fauna underrepresented in open repositories.
- **Real-time and edge constraints** — inference must run on low-power devices (Raspberry Pi, Jetson Nano) attached to field recorders.
- **Domain shift** between open-repository training data (Xeno-canto, iNaturalist) and real field conditions.

---

## General Objective

Design and implement an automated system for species identification from passive acoustic recordings, integrating bioacoustic signal processing with state-of-the-art deep learning, evaluating performance across multiple taxonomic groups, and validating feasibility for edge deployment.

### Specific Objectives

1. Build a modular preprocessing pipeline with validated taxon-specific presets.
2. Evaluate and compare three DL architectures for mel-spectrogram classification.
3. Construct a labeled acoustic dataset from open repositories (Xeno-canto, iNaturalist, GBIF, ChiroVox).
4. Deploy a REST API for real-time inference with structured PostgreSQL storage.
5. Export and benchmark models (ONNX, TorchScript, INT8) for edge deployment.

## Master's Thesis Focus

This repository is being developed as the technical foundation for a master's thesis:

> **Sistema multitaxonómico para detección e identificación de fauna silvestre mediante monitoreo acústico pasivo, con preprocesamiento adaptativo por grupo animal y evaluación en condiciones reales de campo.**

The thesis focus is not to compete with specialized bird identification systems. Instead, it targets a broader and more research-oriented problem: **multitaxonomic bioacoustic monitoring** for acoustically detectable wildlife groups with different signal characteristics, including anurans, bats, birds, insects, and vocal mammals.

The first defendable master's scope is a pilot system with:

- **Anurans** as the main ecological group.
- **Bats** as the technically distinct ultrasonic group.
- **Birds** as a comparative baseline.
- **Insects and vocal mammals** as optional extensions depending on dataset availability.

See [`docs/tesis_maestria.md`](docs/tesis_maestria.md), [`docs/tesis_indice.md`](docs/tesis_indice.md), and [`docs/metodologia/protocolo_dataset_multitaxon.md`](docs/metodologia/protocolo_dataset_multitaxon.md).

---

## Research Relevance for Bioacoustics

### Passive Acoustic Monitoring (PAM)

PAM is a non-invasive, continuous monitoring methodology with growing adoption in conservation biology and soundscape ecology. This system contributes to PAM at multiple levels:

**Signal processing:** Taxon-adaptive preprocessing handles the extreme frequency variation across groups — from ultrasonic bat echolocation pulses (20–200 kHz, requiring 192 kHz recorders) to low-frequency mammal vocalizations (<1 kHz). Per-taxon Butterworth bandpass filters, stationary-profile spectral noise reduction (noisereduce), and energy-based VAD are implemented as named presets.

**Feature extraction:** Log-mel spectrograms with dB compression mirror auditory perception and are the current standard input for bioacoustic DL models. MFCCs + Δ + ΔΔ (40 coefficients) provide temporal dynamics for classical ML comparisons.

**Transfer learning:** PANNs-CNN14, pretrained on AudioSet (526 classes, ~5,800 hours), provides strong general acoustic representations transferable to wildlife sounds — directly addressing the small-dataset problem endemic to field bioacoustics.

**Multi-group coverage:** Bats, frogs, birds, insects, mammals, and reptiles — enabling soundscape-level biodiversity indices (Acoustic Complexity Index, Bioacoustic Index, Species Richness from audio) as complementary metrics.

**Edge deployment:** ONNX INT8 quantization (~75% size reduction) enables deployment on Raspberry Pi 4 or Jetson Nano attached directly to field recorders, removing the need for network connectivity in remote monitoring stations.

### Broader Applications

- Bat diversity surveys via echolocation pulse classification.
- Frog chorus phenology as climate change bio-indicators.
- Insect soundscape analysis as a habitat quality proxy.
- Mammal occupancy surveys from passive recordings at wildlife corridors.
- Near-real-time alerts for endangered or invasive species detection.

---

## Methodology

### Bioacoustic Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  ACQUISITION                                                     │
│  AudioMoth / SM4BAT / Zoom H5  →  .wav / .flac                   │
│  Sample rates: 22,050 Hz (frogs, mammals) — 192,000 Hz (bats)   │
└─────────────────────────┬────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  PREPROCESSING   (src/audio_processing/preprocessor.py)         │
│  1. Butterworth bandpass (taxon-specific frequency range)        │
│  2. Spectral noise reduction (noisereduce, stationary profile)   │
│  3. Peak / RMS / LUFS normalization                              │
│  4. VAD — energy-threshold event detection                       │
│  5. Fixed-length segmentation with 50% overlap                   │
└─────────────────────────┬────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION                                              │
│  • Log-Mel Spectrogram (128 bands, dB) — primary DL input        │
│  • MFCC + Δ + ΔΔ (40 coefficients) — classical ML baseline      │
│  • Spectral stats: ZCR, centroid, bandwidth, rolloff, chroma     │
└─────────────────────────┬────────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  CLASSIFICATION                                                  │
│  • BioAcousticCNN   — residual + attention, ~2M params           │
│  • EfficientNet-B0/B4 — progressive fine-tuning, 3 phases        │
│  • PANNs-CNN14      — AudioSet pretrained, transfer learning     │
└─────────────────────────┬────────────────────────────────────────┘
                          ▼
┌─────────────────────────┬────────────────────────────────────────┐
│  STORAGE                │  SERVING / EDGE                        │
│  PostgreSQL 15          │  FastAPI REST (real-time inference)    │
│  Taxonomic schema       │  ONNX / TorchScript / INT8             │
│  Detection + site data  │  Acoustic monitor (continuous PAM)     │
└─────────────────────────┴────────────────────────────────────────┘
```

### Deep Learning Architectures

| Model | Parameters | Input shape | Pretraining | Best use |
|---|---|---|---|---|
| `BioAcousticCNN` | ~2 M | (1, 128, 128) | None | Fast baseline, interpretable |
| `EfficientNetBioAcoustic` | 5–19 M | (1, 128, 128) | ImageNet | Accuracy-focused |
| `PANNSCNN14BioAcoustic` | ~80 M | (1, T, 128) | AudioSet | Best acoustic transfer |

### Progressive Fine-tuning Strategy (EfficientNet / PANNs)

Gradual unfreezing reduces catastrophic forgetting when adapting pretrained models to wildlife acoustics:

| Phase | Trainable layers | LR | Epochs |
|---|---|---|---|
| 1 | Head only | 1×10⁻³ | 15 |
| 2 | Last 3 blocks | 1×10⁻⁴ | 20 |
| 3 | Full backbone | 5×10⁻⁵ | 15 |

Data augmentation (SpecAugment frequency/time masking, pitch shift ±2 semitones, Gaussian noise, background noise injection) addresses the small-dataset problem.

---

## Dataset and Audio Sources

| Source | Access | Taxonomic Coverage | Notes |
|---|---|---|---|
| Xeno-canto | v3 REST API | Birds, anurans | API key required for downloads |
| iNaturalist | v1 REST API | All groups | CC-licensed |
| GBIF | Occurrence API | All groups | Links to iNat/XC |
| ChiroVox | Direct download | Chiroptera | Curated bat calls |
| FrogID (ANWC) | Direct download | Anura | Australia |

**Dataset status:** `src/data/dataset_builder.py` automates downloads from all sources above. A curated, geographically balanced dataset for Latin American species is in construction. See [Current Status](#current-status).

### Supported Taxonomic Groups

| Group | Frequency range | Sample rate | Notes |
|---|---|---|---|
| Chiroptera | 20 – 200 kHz | 192,000 Hz | Echolocation — requires specialized recorder |
| Anura | 100 – 8,000 Hz | 22,050 Hz | Advertisement calls |
| Aves | 200 – 12,000 Hz | 44,100 Hz | Bird songs and calls; comparative baseline |
| Orthoptera / Cicadidae | 200 – 100,000 Hz | 44,100 Hz | Stridulation |
| Mammalia (vocal) | 20 – 20,000 Hz | 44,100 Hz | Vocalizations, contact calls |
| Reptilia | 100 – 5,000 Hz | 22,050 Hz | Crocodilians, geckos |

---

## Project Structure

```
.
├── src/
│   ├── audio_processing/preprocessor.py    # Bioacoustic preprocessing + taxon presets
│   ├── data/
│   │   ├── dataset_builder.py              # Xeno-canto / iNaturalist / GBIF downloader
│   │   └── augmentation.py                 # Bioacoustic-specific augmentation
│   ├── models/
│   │   ├── cnn_baseline.py                 # BioAcousticCNN + residual blocks
│   │   ├── efficientnet_classifier.py      # EfficientNet + progressive fine-tuning
│   │   ├── panns_classifier.py             # PANNs-CNN14 transfer learning
│   │   └── train.py                        # Unified training pipeline
│   ├── feature_extraction/batch_extractor.py
│   ├── evaluation/evaluator.py             # Metrics, plots, MLflow integration
│   ├── monitoring/acoustic_monitor.py      # Real-time PAM loop
│   └── api/main.py                         # FastAPI REST API
├── database/
│   ├── schema.sql                          # PostgreSQL 15 taxonomic schema
│   └── seed.sql
├── configs/train_config.yaml               # Training hyperparameters
├── scripts/export_model.py                 # ONNX / TorchScript export + benchmarking
├── demo_inference.py                       # Minimal demo: WAV → top-5 predictions
├── notebooks/01_eda.ipynb
├── tests/
│   ├── test_preprocessor.py               # preprocessing tests
│   └── test_models.py                     # model architecture tests
├── results/
│   ├── README.md                          # Results documentation
│   ├── metrics_template.csv               # Metrics recording template
│   └── confusion_matrix_placeholder.md
├── docker/
│   ├── entrypoint.sh
│   └── pgadmin_servers.json
├── Dockerfile                             # Multi-stage builder + runtime
├── docker-compose.yml                     # API + PostgreSQL + pgAdmin + MLflow
├── .env.example
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Installation

### System Requirements

- Python 3.10–3.11
- `libsndfile`: `sudo apt install libsndfile1`
- PostgreSQL 15 (or Docker)
- PortAudio *(optional, only for real-time monitoring)*: `sudo apt install portaudio19-dev`

### Local Installation

```bash
# 1. Clone
git clone https://github.com/ZOMBIECRAFTIAN/Identificacion-de-mamiferos.git
cd Identificacion-de-mamiferos

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. PyTorch CPU (change index URL for CUDA)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 4. Project dependencies
pip install -r requirements.txt

# If Conda/pip upgraded NumPy to 2.x, force the compatible line for PyTorch 2.2.x:
pip install "numpy>=1.26,<2"

# 5. Environment variables
cp .env.example .env
```

### Docker (Recommended)

```bash
cp .env.example .env
docker-compose up -d
docker-compose logs -f api
```

| Service | URL |
|---|---|
| API — Swagger UI | http://localhost:8000/docs |
| pgAdmin 4 | http://localhost:5050 |
| MLflow *(optional profile)* | http://localhost:5000 |

Enable MLflow: `docker-compose --profile mlflow up -d`

---

## Running the Demo

No trained checkpoint required — runs with random weights to verify the pipeline end-to-end.

```bash
# Any .wav file — uses random weights by default
python demo_inference.py --audio path/to/recording.wav

# With a trained checkpoint
python demo_inference.py \
    --audio path/to/recording.wav \
    --checkpoint models/trained/best_efficientnet.pt \
    --model efficientnet \
    --preset mammals

# Available presets: bats | frogs | insects | mammals | birds | reptiles
```

**Expected output:**
```
[INFO] Audio: recording.wav  duration=4.2s  sr=44100 Hz
[INFO] Preset: mammals  |  segments: 2  |  segment_duration: 3.0s
[INFO] Inference complete — 312 ms

Top-5 Predictions (aggregated, 2 segments):
  Rank  Species                        Confidence
  ──────────────────────────────────────────────
  #1    Leopardus pardalis             0.4231
  #2    Nasua nasua                    0.2187
  #3    Odocoileus virginianus         0.1543
  #4    Dasyprocta punctata            0.1102
  #5    Tamandua mexicana              0.0937
```

---

## Training

### 1. Build Dataset

```bash
# Xeno-canto API v3 requires an API key for downloads.
# PowerShell:
#   $env:XENO_CANTO_API_KEY="your_key"
# Linux/macOS:
#   export XENO_CANTO_API_KEY="your_key"

# Download the initial Mexico birds profile
python -m src.data.dataset_builder \
    --profile mexico_birds \
    --output data/raw \
    --max-per-class 300

# Master's thesis multitaxon pilot
python -m src.data.dataset_builder \
    --profile mexico_multitaxon \
    --output data/raw/multitaxon \
    --max-per-class 150

# Individual group profiles
python -m src.data.dataset_builder --profile mexico_anurans --output data/raw/anurans
python -m src.data.dataset_builder --profile mexico_bats --output data/raw/bats
python -m src.data.dataset_builder --profile mexico_insects --output data/raw/insects
python -m src.data.dataset_builder --profile mexico_mammals --output data/raw/mammals

# Extract mel spectrogram features
python -m src.feature_extraction.batch_extractor \
    --input data/raw \
    --output data/spectrograms \
    --preset birds \
    --workers 4
```

### 2. Train

```bash
# Recommended: EfficientNet with YAML config
python -m src.models.train \
    --config configs/train_mexico_birds.yaml

# Master's thesis multitaxon pilot
python -m src.models.train \
    --config configs/train_multitaxon.yaml

# Override from CLI
python -m src.models.train \
    --config configs/train_mexico_birds.yaml \
    --model efficientnet \
    --epochs 50 \
    --device cuda
```

### 3. Evaluate

```bash
python -m src.evaluation.evaluator \
    --checkpoint models/trained/best_efficientnet.pt \
    --data-dir data/spectrograms/test \
    --output-dir results/evaluation \
    --model-type efficientnet \
    --mlflow
```

### 4. Export for Edge Deployment

```bash
# TorchScript + ONNX with latency benchmark
python scripts/export_model.py \
    --checkpoint models/trained/best_efficientnet.pt \
    --model-type efficientnet \
    --n-classes 17 \
    --output-dir models/exported \
    --formats torchscript onnx \
    --benchmark

# INT8 quantization (~75% size reduction, target: Raspberry Pi 4)
python scripts/export_model.py \
    --checkpoint models/trained/best_efficientnet.pt \
    --model-type efficientnet \
    --n-classes 17 \
    --formats onnx_int8 \
    --calibration-data data/spectrograms/mammals
```

---

## API Reference

```bash
# Classify an audio file
curl -X POST http://localhost:8000/classify \
    -F "file=@recording.wav" \
    -F "top_k=5" \
    -F "preset=mammals"

# Real-time acoustic monitor (requires pyaudio)
python -m src.monitoring.acoustic_monitor run \
    --checkpoint models/trained/best_efficientnet.pt \
    --class-names class_names.txt \
    --device-index 0 \
    --sample-rate 44100
```

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/classify` | Classify uploaded audio file (multipart) |
| `POST` | `/classify/url` | Classify audio from public URL |
| `GET` | `/detections` | List detections with filters |
| `GET` | `/species` | Species catalog with taxonomic filters |
| `GET` | `/reports/site/{id}` | Monthly biodiversity report per monitoring site |
| `GET` | `/health` | Service + model + DB status |

Interactive docs: `http://localhost:8000/docs`

---

## Tests

```bash
pytest tests/test_preprocessor.py -v          # preprocessing tests
pytest tests/test_models.py -v                # model architecture tests
pytest tests/ --cov=src --cov-report=term-missing  # full suite + coverage
```

---

## Current Status

> Honest assessment — no metrics are claimed until experimental validation is complete.

**Complete and functional:**
- ✅ Bioacoustic preprocessing with taxon-specific presets (bats, frogs, birds, insects, mammals, reptiles)
- ✅ Three DL architectures implemented and trainable (BioAcousticCNN, EfficientNet, PANNs-CNN14)
- ✅ Progressive fine-tuning trainer with SpecAugment and MixUp
- ✅ FastAPI REST API with async PostgreSQL backend
- ✅ ONNX / TorchScript export pipeline with INT8 quantization
- ✅ Real-time acoustic monitor with VAD + sliding-window inference
- ✅ Pytest suite covering preprocessing, models, API, data, and training utilities
- ✅ CI/CD pipeline (GitHub Actions: ruff/black lint, pytest, bandit, schema validation, Docker build)

**In progress — not yet validated:**
- 🔄 Curated, geographically representative labeled dataset for Latin American fauna
- 🔄 Comparative evaluation: F1-macro, top-5 accuracy, ROC-AUC across architectures
- 🔄 Latency benchmarks on target edge hardware (Raspberry Pi 4, Jetson Nano)
- 🔄 Domain shift analysis between training sources and field recordings

Results will be published in `results/` as experiments are completed.

---

## Roadmap

**Short-term (0–3 months)**
- [ ] Curate a multi-class dataset ≥500 labeled recordings per taxon group
- [ ] Run comparative training experiment across all three architectures
- [ ] Document accuracy, F1-macro, and confusion matrices in `results/`
- [ ] Validate edge latency on Raspberry Pi 4 (target: <500 ms/segment)

**Medium-term (3–9 months)**
- [ ] Expand to ≥50 Latin American species across 3 taxonomic groups
- [ ] Add temporal activity analysis (diel and seasonal patterns)
- [ ] Implement soundscape indices (ACI, ADI, BI) as complementary biodiversity metrics
- [ ] Prepare benchmark dataset + methods paper

**Long-term**
- [ ] Federated learning for multi-site model improvement without centralizing data
- [ ] Integration with bioacoustic platforms (ARBIMON, BirdNET ecosystem)
- [ ] Real-time alert pipeline for endangered or invasive species detection

---

## Known Limitations

1. **No validated accuracy metrics.** All performance figures are provisional pending experimental validation.
2. **Bat detection requires specialized hardware** (192 kHz recorders: AudioMoth v2, SM4BAT). Consumer microphones are insufficient.
3. **Domain shift unquantified.** Performance on out-of-distribution recordings (different environments, microphones, geographic regions) is unknown.
4. **PANNs-CNN14 is large (~80M params).** Not feasible on microcontrollers without extreme compression.
5. **Geographic dataset bias.** Open repositories have uneven coverage for Latin American species.
6. **`pyaudio` is optional.** Real-time monitoring requires PortAudio, which may fail on minimal server environments. Core inference and API work without it.

---

## For My Interview

### 1-Minute Version

> "I'm building a passive acoustic monitoring system that automatically identifies wildlife species from field recordings. The pipeline handles the full bioacoustic workflow: taxon-adaptive signal preprocessing, log-mel spectrogram extraction, and classification using three deep learning architectures — including PANNs-CNN14, pretrained on AudioSet's 5,800 hours of audio, which transfers remarkably well to wildlife sounds. The system serves inference through a REST API, stores structured detections in PostgreSQL, and exports models to ONNX for edge deployment on low-power field recorders. The engineering pipeline is complete; I'm currently focused on building a curated labeled dataset for Latin American mammals, frogs, and bats to run the comparative evaluation. This project directly addresses the analysis bottleneck in PAM — the gap between what recorders capture and what biologists can manually annotate."

### 3-Minute Version

> "My project addresses the core scalability problem in passive acoustic monitoring: field recorders like AudioMoth can generate hundreds of hours of audio per deployment, but traditional analysis requires expert manual annotation at roughly 3–10× real time. That doesn't scale for continental-scale biodiversity surveys.
>
> The pipeline starts with taxon-adaptive signal processing. Bat recordings need a 192 kHz sample rate and a 10–96 kHz bandpass filter to capture echolocation pulses; frog chorus analysis works at 22 kHz with a 100–10,000 Hz range. The preprocessing module applies Butterworth bandpass filtering, stationary-profile spectral noise reduction, and energy-based voice activity detection before producing fixed-length segments for the classifier.
>
> I'm evaluating three architectures. A custom residual CNN as an interpretable, low-parameter baseline. EfficientNet-B0 with progressive fine-tuning — unfreezing layers gradually to avoid catastrophic forgetting when adapting from ImageNet to acoustic spectrograms. And PANNs-CNN14, from Kong et al. 2020, pretrained on AudioSet's 526 sound classes. PANNs is the most promising for this task because it was trained specifically on audio, not images, and its representations transfer to wildlife sounds with very few labeled examples.
>
> The system is deployed as a FastAPI REST API with async PostgreSQL storage, a real-time acoustic monitor, and an export pipeline to ONNX INT8 — which reduces model size by ~75% and targets deployment on Raspberry Pi 4 attached directly to the recorder.
>
> The honest status: the engineering pipeline is complete and covered by a pytest suite plus a full CI/CD pipeline. What's missing is the validated dataset and comparative evaluation — which is exactly what I would pursue in a graduate research program. I deliberately don't report any accuracy numbers until I have a properly curated, geographically representative dataset. That's the scientifically defensible position, and building that dataset with rigorous protocols would be my first research priority."

---

## How to Cite

```bibtex
@software{bioacustica_fauna_2024,
  author  = {Ian},
  title   = {Bioacustica Fauna},
  year    = {2024},
  url     = {https://github.com/ZOMBIECRAFTIAN/bioacustica-fauna},
  note    = {PAM pipeline with deep learning classification.
             Engineering complete; experimental validation in progress.}
}
```

---

## References

- Kong, Q. et al. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. *IEEE/ACM TASLP*. https://doi.org/10.1109/TASLP.2020.3030497
- Park, D. S. et al. (2019). SpecAugment: A Simple Data Augmentation Method for ASR. *INTERSPEECH*. https://doi.org/10.21437/Interspeech.2019-2680
- Tan, M. & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*.
- Pijanowski, B. C. et al. (2011). Soundscape Ecology. *BioScience*, 61(3), 203–216.
- Sueur, J. & Farina, A. (2015). Ecoacoustics: Ecological Investigation of Environmental Sound. *Biosemiotics*, 8(3), 493–502.
- Stowell, D. et al. (2019). Automatic Acoustic Detection of Birds Through Deep Learning. *Methods in Ecology and Evolution*, 10(3), 368–380.

---

## License

MIT — see `LICENSE`.

## Author

**Ian** — Systems Engineering / Bioacoustics Research  
`brianferbaez@gmail.com` · [@ZOMBIECRAFTIAN](https://github.com/ZOMBIECRAFTIAN)
