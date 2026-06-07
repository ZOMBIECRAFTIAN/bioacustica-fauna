"""
tests/conftest.py
-----------------------------------------------------------------------------
Fixtures compartidos para todos los tests del proyecto.
pytest los descubre automaticamente -- no requiere importacion explicita.

Uso en cualquier test:
    def test_algo(sine_audio, default_config, tmp_wav_file):
        ...
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Asegurar que la raiz del proyecto este en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Constantes globales de test
# -----------------------------------------------------------------------------
SR = 22_050  # sample rate estandar para tests
SR_BATS = 192_000  # sample rate quiropteros
DURATION = 3.0  # segundos
N_SAMPLES = int(SR * DURATION)
N_CLASSES = 10
BATCH_SIZE = 4
N_MELS = 128
N_FRAMES = 128


# -----------------------------------------------------------------------------
# Fixtures de audio
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sine_audio():
    """Senal sinusoidal pura 440 Hz, 3s, mono, float32."""
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture(scope="session")
def white_noise():
    """Ruido blanco gaussiano, 3s, RMS ~0.1."""
    rng = np.random.default_rng(seed=42)
    return (rng.normal(0, 0.1, N_SAMPLES)).astype(np.float32)


@pytest.fixture(scope="session")
def silence():
    """Senal de silencio (ceros), 3s."""
    return np.zeros(N_SAMPLES, dtype=np.float32)


@pytest.fixture(scope="session")
def chirp_audio():
    """Chirp lineal 200 Hz -> 8000 Hz, 3s (simula vocalizacion de anfibio)."""
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    freq = np.linspace(200, 8000, N_SAMPLES)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return (0.3 * np.sin(phase)).astype(np.float32)


@pytest.fixture(scope="session")
def short_audio():
    """Senal corta de 0.5s (para tests de segmentacion)."""
    t = np.linspace(0, 0.5, int(SR * 0.5), endpoint=False)
    return (0.4 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


# -----------------------------------------------------------------------------
# Fixtures de archivos temporales
# -----------------------------------------------------------------------------


@pytest.fixture
def tmp_wav_file(sine_audio):
    """Archivo WAV temporal con senal sinusoidal. Se elimina al final del test."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    sf.write(path, sine_audio, SR, subtype="PCM_16")
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def tmp_wav_dir(sine_audio, white_noise, chirp_audio):
    """Directorio temporal con 3 archivos WAV de diferentes senales."""
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        sf.write(d / "sine.wav", sine_audio, SR, subtype="PCM_16")
        sf.write(d / "noise.wav", white_noise, SR, subtype="PCM_16")
        sf.write(d / "chirp.wav", chirp_audio, SR, subtype="PCM_16")
        yield d


@pytest.fixture
def tmp_output_dir():
    """Directorio temporal de salida para resultados."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# -----------------------------------------------------------------------------
# Fixtures de configuracion
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def default_config():
    """AudioConfig por defecto."""
    from src.audio_processing.preprocessor import AudioConfig

    return AudioConfig(sample_rate=SR, n_mels=N_MELS)


@pytest.fixture(scope="session")
def frogs_config():
    """AudioConfig preset para anfibios."""
    from src.audio_processing.preprocessor import PRESETS

    return PRESETS["frogs"]


@pytest.fixture(scope="session")
def preprocessor(default_config):
    """AudioPreprocessor con config por defecto."""
    from src.audio_processing.preprocessor import AudioPreprocessor

    return AudioPreprocessor(default_config)


# -----------------------------------------------------------------------------
# Fixtures de modelos PyTorch
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dummy_mel():
    """Batch de mel spectrograms sinteticos: (BATCH_SIZE, 1, N_MELS, N_FRAMES)."""
    try:
        import torch

        rng = np.random.default_rng(seed=0)
        arr = rng.uniform(-80, 0, (BATCH_SIZE, 1, N_MELS, N_FRAMES)).astype(np.float32)
        return torch.from_numpy(arr)
    except ImportError:
        pytest.skip("PyTorch no disponible")


@pytest.fixture(scope="session")
def dummy_labels():
    """Labels enteros para BATCH_SIZE muestras, N_CLASSES clases."""
    try:
        import torch

        return torch.randint(0, N_CLASSES, (BATCH_SIZE,))
    except ImportError:
        pytest.skip("PyTorch no disponible")


@pytest.fixture(scope="session")
def cnn_model():
    """BioAcousticCNN pequeno para tests (n_classes=N_CLASSES)."""
    try:
        from src.models.cnn_baseline import BioAcousticCNN

        model = BioAcousticCNN(n_classes=N_CLASSES, n_mels=N_MELS)
        model.eval()
        return model
    except ImportError:
        pytest.skip("PyTorch o src.models no disponible")


# -----------------------------------------------------------------------------
# Fixtures de indices acusticos
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def indices_config():
    """IndicesConfig por defecto."""
    from src.monitoring.acoustic_indices import IndicesConfig

    return IndicesConfig(sample_rate=SR)


@pytest.fixture(scope="session")
def acoustic_indices(indices_config):
    """AcousticIndices instanciado con sine_audio."""
    from src.monitoring.acoustic_indices import AcousticIndices

    # Se crea con un array placeholder; se reemplaza en cada test
    return AcousticIndices(np.zeros(N_SAMPLES, dtype=np.float32), indices_config)
