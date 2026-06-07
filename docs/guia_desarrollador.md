# Developer Guide

## Architecture Overview

```
bioacustica-fauna/
├── src/
│   ├── audio_processing/     # Signal processing pipeline
│   │   └── preprocessor.py   # AudioConfig, AudioPreprocessor, PRESETS
│   ├── models/               # Deep learning models
│   │   ├── cnn_baseline.py   # BioAcousticCNN (residual + attention)
│   │   ├── efficientnet_classifier.py  # EfficientNet B0/B2/B4
│   │   ├── panns_classifier.py         # PANNs-CNN14 (AudioSet pretrained)
│   │   └── train.py          # Unified training pipeline
│   ├── data/
│   │   ├── dataset_builder.py  # Xeno-canto, iNaturalist, GBIF download
│   │   └── augmentation.py     # Audio + spectrogram augmentation
│   ├── feature_extraction/
│   │   └── batch_extractor.py  # Parallel batch feature extraction
│   ├── evaluation/
│   │   └── evaluator.py        # ModelEvaluator, metrics, figures
│   ├── monitoring/
│   │   ├── acoustic_monitor.py   # Real-time PAM (PyAudio + threads)
│   │   ├── acoustic_indices.py   # ACI, ADI, AEI, BI, NDSI, H
│   │   └── soundscape_analyzer.py # Batch analysis + visualization
│   └── api/
│       └── main.py           # FastAPI REST API
├── database/
│   ├── schema.sql            # PostgreSQL 15 schema
│   └── seed.sql              # Reference data seed
├── scripts/
│   ├── export_model.py       # ONNX / TorchScript export + quantization
│   ├── setup_dev.sh          # Linux/macOS setup
│   ├── setup_dev.bat         # Windows setup
│   └── verify_install.py     # Dependency checker
├── tests/                    # pytest unit tests
├── configs/
│   └── train_config.yaml     # Training hyperparameters
├── docs/                     # Documentation
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
└── docker/
    ├── entrypoint.sh
    └── pgadmin_servers.json
```

---

## Adding a New Model

### 1. Create the model file

```python
# src/models/my_model.py
from __future__ import annotations
import torch
import torch.nn as nn

class MyBioAcousticModel(nn.Module):
    def __init__(self, n_classes: int, **kwargs):
        super().__init__()
        # ... architecture
        self.head = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T) mel spectrogram
        # returns: (B, n_classes) logits
        ...
```

### 2. Register in train.py

In `src/models/train.py`, add to the `build_model()` factory:

```python
elif cfg.model_type == "my_model":
    from src.models.my_model import MyBioAcousticModel
    model = MyBioAcousticModel(n_classes=cfg.n_classes)
```

### 3. Register in export_model.py

In `scripts/export_model.py`, add to `load_model()`:

```python
elif model_type == "my_model":
    from src.models.my_model import MyBioAcousticModel
    model = MyBioAcousticModel(n_classes=n_classes)
```

### 4. Add tests

```python
# tests/test_models.py
class TestMyModel:
    @pytest.fixture
    def my_model(self):
        return MyBioAcousticModel(n_classes=10)

    def test_forward_shape(self, my_model, dummy_mel):
        logits = my_model(dummy_mel)
        assert logits.shape == (BATCH_SIZE, 10)
```

---

## Adding a New Taxon Preset

In `src/audio_processing/preprocessor.py`:

```python
PRESETS: dict[str, AudioConfig] = {
    # ... existing presets ...
    "fish":  AudioConfig(          # Passive acoustic for fish
        sample_rate=44_100,
        freq_low=20,
        freq_high=20_000,
        n_mels=128,
        fmin=20,
        segment_duration=5.0,
        hop_duration=2.5,
    ),
}
```

Then add the augmentation preset in `src/data/augmentation.py`:

```python
def get_preset(taxon: str) -> AugmentationPipeline:
    presets = {
        # ...
        "fish": AugmentationPipeline(
            transforms=[AddBackgroundNoise(min_snr_db=5, max_snr_db=20)],
            p_apply=0.8,
        ),
    }
```

---

## Adding a New Acoustic Index

In `src/monitoring/acoustic_indices.py`:

```python
class AcousticIndices:
    def compute_my_index(self) -> float:
        """
        My Index description.

        Reference: Author (Year). Title. Journal.
        """
        Sxx, freqs, times = self._compute_spectrogram(self._y)
        Sxx_db = self._power_to_db(Sxx)
        # ... computation
        return float(result)

    def compute_all(self) -> IndicesResult:
        # Add to the result:
        return IndicesResult(
            # ...
            my_index=self.compute_my_index(),
        )
```

Also add `my_index: float = 0.0` to `IndicesResult` dataclass.

---

## Database: Adding a New Table

1. Add to `database/schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS my_table (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- ... columns
);

CREATE INDEX IF NOT EXISTS idx_my_table_created
    ON my_table (created_at DESC);
```

2. Create Alembic migration (once Alembic is configured):

```bash
alembic revision --autogenerate -m "add my_table"
alembic upgrade head
```

---

## Configuration

All model hyperparameters are in `configs/train_config.yaml`.

Key sections:

```yaml
model:
  type: cnn_baseline          # cnn_baseline | efficientnet | panns
  n_classes: 150
  dropout: 0.3

training:
  epochs: 50
  batch_size: 32
  lr: 1e-3
  weight_decay: 1e-4
  scheduler: cosine           # cosine | plateau | step

audio:
  preset: mammals             # bats | frogs | insects | mammals | reptiles
  augmentation: true
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_models.py::TestBioAcousticCNN -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run only fast tests
python -m pytest tests/ -m "not slow" -v

# Run only GPU tests (if available)
python -m pytest tests/ -m gpu -v
```

---

## Coding Standards

- Formatter: `black` (line-length=100)
- Linter: `ruff` (pyflakes + pyupgrade + bugbear + isort)
- All public functions must have type annotations and docstrings
- Scientific variable names (X, Y, Sxx, etc.) are allowed — ruff N803/N806 ignored
- Tests must cover happy path + edge cases (silence, single sample, max classes)

```bash
# Run lint + format before committing:
ruff check src/ tests/ scripts/ --fix
black src/ tests/ scripts/
```
