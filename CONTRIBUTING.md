# Bioacustica Fauna - Contributing Guide

## Requirements

- Python >= 3.10
- Git
- Docker (optional, for integration tests)

## Setup

```bash
git clone https://github.com/ZOMBIECRAFTIAN/bioacustica-fauna.git
cd bioacustica-fauna
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install ruff black pytest pytest-cov
```

## Branch Strategy

```
main        production-ready releases (tagged)
develop     integration branch
feature/*   new features
fix/*       bug fixes
docs/*      documentation only
```

All PRs target `develop`. Merges to `main` are tagged releases.

## Code Standards

**Formatter:** black (line-length = 100)
**Linter:** ruff (includes isort I-rules, pyflakes, pyupgrade, bugbear)

Run before committing:
```bash
ruff check src/ tests/ scripts/ --fix
black src/ tests/ scripts/
python -m pytest tests/ -v
```

All CI checks must pass before merge.

## Commit Messages

Format: `<type>(<scope>): <description>`

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`

Examples:
```
feat(models): add EfficientNet-B4 variant
fix(preprocessor): silence mel spectrogram returns -100dB
docs(api): add curl examples to /classify endpoint
test(acoustic_indices): add windowed analysis tests
```

## Adding a New Model

1. Create `src/models/<model_name>.py`
2. Implement `forward(x) -> logits` interface
3. Add to `src/models/__init__.py`
4. Add tests in `tests/test_models.py`
5. Add to `scripts/export_model.py` load_model factory
6. Document in `docs/guia_desarrollador.md`

## Adding a New Taxon Preset

In `src/audio_processing/preprocessor.py`, add to `PRESETS` dict:

```python
"new_taxon": AudioConfig(
    sample_rate=...,
    freq_low=...,
    freq_high=...,
    n_mels=...,
    segment_duration=...,
),
```

## Reporting Bugs

Open an issue with:
- Python version
- OS
- Minimal reproducible example
- Full traceback

## Pull Request Checklist

- [ ] Tests pass locally (`python -m pytest tests/ -v`)
- [ ] Ruff and black pass
- [ ] Docstring added/updated for public functions
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] No secrets or credentials committed
