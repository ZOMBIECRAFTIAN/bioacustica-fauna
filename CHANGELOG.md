# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

### Planned
- Alembic database migrations
- tests/conftest.py with shared fixtures
- tests/test_api.py (FastAPI integration tests)
- tests/test_augmentation.py
- INSTALL.md and setup scripts
- Full API reference documentation

---

## [0.3.0] - 2026-05-28

### Added
- `src/monitoring/acoustic_indices.py` — Soundscape ecology indices: ACI, ADI, AEI, BI, NDSI, Hf, Ht, H (Pieretti 2011; Sueur 2008; Villanueva-Rivera 2011)
- `src/monitoring/soundscape_analyzer.py` — Batch analysis, 6 visualization types, CSV/JSON output, CLI interface
- `tests/test_acoustic_indices.py` — 40+ unit tests for all acoustic indices
- `src/monitoring/__init__.py` — Public API for monitoring module
- `pyproject.toml` — PEP 517/621 project metadata, tool configuration (ruff, black, pytest, mypy, coverage)
- `VERSION` — Semantic version file

### Fixed
- `src/audio_processing/preprocessor.py`: `mel_spectrogram()` silence bug — `np.maximum(mel, 1e-10)` floor prevents `power_to_db(ref=np.max)` returning 0.0 for silent frames
- `src/audio_processing/preprocessor.py`: librosa 0.10+ API — all spectral feature calls updated to keyword-only arguments (`y=y`)
- `src/models/panns_classifier.py`: split semicolon statements onto separate lines (ruff E702)
- `src/evaluation/evaluator.py`: split semicolon statements onto separate lines
- `src/feature_extraction/batch_extractor.py`: generator comprehension rewritten as set comprehension (ruff C401)
- `src/monitoring/acoustic_monitor.py`: removed unused `prometheus_client.Gauge` import
- `src/api/main.py`: removed unused imports (`io`, `UUID`, `Depends`, `field_validator`, `EfficientNetBioAcoustic`)
- `.github/workflows/ci.yml`: removed standalone `isort` step (ruff I-rules are authoritative); removed pinned `torch==2.2.2` (Python 3.10+ compatibility)
- All source files: stripped null-byte padding (Windows Write tool artifact); ensured trailing newlines

### Changed
- CI lint job: `ruff` handles both linting and import ordering; `black` handles formatting only
- `requirements.txt`: removed `torch==2.2.2` pin; torch installed manually per platform

---

## [0.2.0] - 2026-05-20

### Added
- `src/evaluation/evaluator.py` — Full evaluation pipeline: accuracy, balanced accuracy, macro/weighted F1, ROC-AUC OvR, PR-AUC, ECE, Brier score, Top-K; confusion matrix, ROC/PR curves, calibration reliability diagram; MLflow integration
- `src/data/augmentation.py` — Audio and spectrogram augmentation: PitchShift, TimeStretch, AddGaussianNoise, AddBackgroundNoise (SNR-controlled), SpecAugment, FrequencyShift, RandomErasing; presets per taxon (bats, frogs, insects, mammals, reptiles)
- `src/monitoring/acoustic_monitor.py` — Real-time PAM system: VAD filter, ring buffer, PyAudio callback, PostgreSQL writer, JSONL daily rotation, optional Prometheus metrics
- `scripts/export_model.py` — Model export: TorchScript (trace), ONNX (opset 17, dynamic axes), INT8 quantization (static/dynamic), benchmarking
- `.github/workflows/ci.yml` — 5-stage CI: lint, test (matrix py3.10+3.11), security (bandit+safety), schema validation (PostgreSQL), docker build
- `README.md` — Full project documentation

---

## [0.1.0] - 2026-05-10

### Added
- `src/audio_processing/preprocessor.py` — AudioConfig dataclass, PRESETS per taxon (bats/frogs/insects/mammals/reptiles), AudioPreprocessor pipeline: load, bandpass filter (Butterworth), noise reduction (noisereduce), normalization, segmentation, mel spectrogram, MFCC+delta+delta-delta, spectral features
- `database/schema.sql` — PostgreSQL 15 full taxonomy schema: species, recordings, detections, acoustic_sessions, model_versions, acoustic_indices tables; indexes, constraints, sequences
- `docs/arquitectura_sistema.md` — System architecture documentation
- `docs/marco_teorico/marco_teorico_bioacustica_ia.docx` — Theoretical framework (bioacoustics + AI)
- `src/models/cnn_baseline.py` — BioAcousticCNN with residual blocks, channel attention (SE), MixUp, SpecAugment
- `src/models/efficientnet_classifier.py` — EfficientNet-B0/B2/B4 with 3-phase progressive fine-tuning
- `src/models/panns_classifier.py` — PANNs-CNN14 exact architecture, AudioSet pretrained weights loader
- `src/models/train.py` — Unified training pipeline: warmup, cosine annealing, gradient clipping, MLflow tracking
- `src/data/dataset_builder.py` — Automated dataset download: Xeno-canto, iNaturalist, GBIF APIs
- `src/api/main.py` — FastAPI REST API: /classify, /health, /models, /species endpoints
- `src/feature_extraction/batch_extractor.py` — Parallel batch feature extraction
- `notebooks/01_eda.ipynb` — Exploratory data analysis notebook
- `configs/train_config.yaml` — Training configuration
- `Dockerfile` — Multi-stage build (builder + runtime), PyTorch CPU-only, HEALTHCHECK
- `docker-compose.yml` — Services: PostgreSQL 15 + PostGIS, API, pgAdmin, MLflow (optional)
- `docker/entrypoint.sh` — Wait-for-DB, auto-schema apply, uvicorn launch
- `requirements.txt` — Full dependency list
- `tests/test_preprocessor.py` — 48 unit tests for audio preprocessing pipeline
- `tests/test_models.py` — 50+ unit tests for CNN, EfficientNet, PANNs models

---

## Version Scheme

```
MAJOR.MINOR.PATCH

MAJOR — breaking API change or architectural redesign
MINOR — new module, feature, or significant enhancement
PATCH — bug fix, documentation update, refactor
```

Git tags: `git tag v0.3.0 && git push origin v0.3.0`
