"""
tests/test_augmentation.py
-----------------------------------------------------------------------------
Tests unitarios para src/data/augmentation.py.

Cubre:
  - Transforms de audio: PitchShift, TimeStretch, AddGaussianNoise,
    AddBackgroundNoise, RandomClip, VolumeJitter, TimeShift
  - Transforms de espectrograma: SpecAugmentFreq, SpecAugmentTime,
    FrequencyShift, RandomErasing, GaussianBlur
  - AugmentationPipeline y SpectrogramAugmentationPipeline
  - Presets por taxon

Ejecutar:
    pytest tests/test_augmentation.py -v
    pytest tests/test_augmentation.py -v -k "noise"
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.data.augmentation import (
    AddBackgroundNoise,
    AddGaussianNoise,
    AugmentationPipeline,
    GaussianBlur,
    RandomClip,
    RandomErasing,
    SpecAugmentFreq,
    SpecAugmentTime,
    SpectrogramAugmentationPipeline,
    TimeShift,
    TimeStretch,
    VolumeJitter,
    get_preset,
)

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------
SR = 22_050
DURATION = 3.0
N = int(SR * DURATION)
N_MELS = 128
N_FRAMES = 128

# -----------------------------------------------------------------------------
# Fixtures locales
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sine():
    t = np.linspace(0, DURATION, N, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture(scope="module")
def noise_audio():
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.1, N).astype(np.float32)


@pytest.fixture(scope="module")
def mel_spec():
    """Mel spectrogram sintetico (N_MELS, N_FRAMES) en dB."""
    rng = np.random.default_rng(0)
    return rng.uniform(-80, 0, (N_MELS, N_FRAMES)).astype(np.float32)


# =============================================================================
# 1. TRANSFORMS DE AUDIO
# =============================================================================


class TestAddGaussianNoise:
    def test_output_shape_preserved(self, sine):
        t = AddGaussianNoise(snr_db_range=(10, 30), p=1.0)
        out = t(sine, SR)
        assert out.shape == sine.shape

    def test_output_dtype_float32(self, sine):
        t = AddGaussianNoise(p=1.0)
        assert t(sine, SR).dtype == np.float32

    def test_signal_is_modified(self, sine):
        t = AddGaussianNoise(snr_db_range=(5, 10), p=1.0)
        out = t(sine, SR)
        assert not np.allclose(out, sine)

    def test_p_zero_returns_original(self, sine):
        t = AddGaussianNoise(p=0.0)
        out = t(sine, SR)
        np.testing.assert_array_equal(out, sine)

    def test_snr_high_minimal_noise(self, sine):
        """SNR muy alto => poca distorsion."""
        t = AddGaussianNoise(snr_db_range=(60, 80), p=1.0)
        out = t(sine, SR)
        diff = np.mean(np.abs(out - sine))
        assert diff < 0.01


class TestTimeStretch:
    def test_output_is_audio(self, sine):
        t = TimeStretch(rate_range=(0.9, 1.1), p=1.0)
        out = t(sine, SR)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert len(out) > 0

    def test_p_zero_unchanged(self, sine):
        t = TimeStretch(p=0.0)
        out = t(sine, SR)
        np.testing.assert_array_equal(out, sine)

    def test_stretch_rate_applied(self, sine):
        """TimeStretch con rate != 1.0 produce audio valido (puede redondear longitud)."""
        t = TimeStretch(rate_range=(0.8, 0.8), p=1.0)
        out = t(sine, SR)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert len(out) > 0
        assert np.all(np.isfinite(out))


class TestRandomClip:
    def test_output_length_correct(self, sine):
        target = int(SR * 2.0)
        t = RandomClip(clip_duration_s=2.0, p=1.0)
        out = t(sine, SR)
        assert len(out) == target

    def test_short_input_padded(self, sine):
        """Si el input es mas corto que target, se hace padding."""
        short = sine[:SR]  # 1 segundo
        target = int(SR * 2.0)
        t = RandomClip(clip_duration_s=2.0, p=1.0)
        out = t(short, SR)
        assert len(out) == target

    def test_p_zero_unchanged(self, sine):
        t = RandomClip(clip_duration_s=DURATION, p=0.0)
        out = t(sine, SR)
        np.testing.assert_array_equal(out, sine)


class TestVolumeJitter:
    def test_shape_preserved(self, sine):
        t = VolumeJitter(gain_db_range=(-6, 6), p=1.0)
        out = t(sine, SR)
        assert out.shape == sine.shape

    def test_amplitude_changes(self, sine):
        t = VolumeJitter(gain_db_range=(6, 6), p=1.0)
        out = t(sine, SR)
        ratio = np.max(np.abs(out)) / (np.max(np.abs(sine)) + 1e-9)
        assert abs(ratio - 2.0) < 0.1  # +6 dB ~ x2 amplitud


class TestTimeShift:
    def test_shape_preserved(self, sine):
        t = TimeShift(shift_max_s=0.9, p=1.0)
        out = t(sine, SR)
        assert out.shape == sine.shape

    def test_dtype_preserved(self, sine):
        t = TimeShift(p=1.0)
        assert t(sine, SR).dtype == np.float32


class TestAddBackgroundNoise:
    def test_without_noise_files(self, sine):
        """Sin archivos de ruido, retorna senal original."""
        t = AddBackgroundNoise(noise_dir=None, p=1.0)
        out = t(sine, SR)
        assert out.shape == sine.shape

    def test_with_noise_array(self, sine, noise_audio, tmp_path):
        """Con archivo de ruido real, la senal cambia."""
        import soundfile as sf

        nf = tmp_path / "noise.wav"
        sf.write(nf, noise_audio, SR)
        t = AddBackgroundNoise(
            noise_dir=str(tmp_path), target_snr_db=15.0, snr_jitter_db=5.0, p=1.0
        )
        out = t(sine, SR)
        assert out.shape == sine.shape
        assert not np.allclose(out, sine)


# =============================================================================
# 2. TRANSFORMS DE ESPECTROGRAMA
# =============================================================================


class TestSpecAugmentFreq:
    def test_shape_preserved(self, mel_spec):
        t = SpecAugmentFreq(max_mask_pct=0.15, n_masks=2, p=1.0)
        out = t(mel_spec)
        assert out.shape == mel_spec.shape

    def test_zeros_introduced(self, mel_spec):
        """Freq masking introduce ceros en filas."""
        t = SpecAugmentFreq(max_mask_pct=0.23, n_masks=3, p=1.0)
        out = t(mel_spec)
        assert np.any(out == 0.0)

    def test_p_zero_unchanged(self, mel_spec):
        t = SpecAugmentFreq(p=0.0)
        out = t(mel_spec)
        np.testing.assert_array_equal(out, mel_spec)


class TestSpecAugmentTime:
    def test_shape_preserved(self, mel_spec):
        t = SpecAugmentTime(max_mask_pct=0.15, n_masks=2, p=1.0)
        out = t(mel_spec)
        assert out.shape == mel_spec.shape

    def test_zeros_introduced(self, mel_spec):
        t = SpecAugmentTime(max_mask_pct=0.23, n_masks=3, p=1.0)
        out = t(mel_spec)
        assert np.any(out == 0.0)


class TestRandomErasing:
    def test_shape_preserved(self, mel_spec):
        t = RandomErasing(p=1.0)
        out = t(mel_spec)
        assert out.shape == mel_spec.shape

    def test_modifies_spectrogram(self, mel_spec):
        t = RandomErasing(p=1.0, area_pct=0.5)
        out = t(mel_spec)
        assert not np.allclose(out, mel_spec)


class TestGaussianBlur:
    def test_shape_preserved(self, mel_spec):
        t = GaussianBlur(sigma_range=(0.5, 1.5), p=1.0)
        out = t(mel_spec)
        assert out.shape == mel_spec.shape

    def test_smoothing_applied(self, mel_spec):
        """Blur reduce la varianza local."""
        t = GaussianBlur(sigma_range=(2.0, 2.0), p=1.0)
        out = t(mel_spec)
        # Varianza global debe reducirse
        assert np.var(out) <= np.var(mel_spec) + 1e-3


# =============================================================================
# 3. PIPELINES
# =============================================================================


class TestAugmentationPipeline:
    def test_empty_pipeline_returns_original(self, sine):
        pipeline = AugmentationPipeline(transforms=[], p_apply=1.0)
        out = pipeline(sine, SR)
        np.testing.assert_array_equal(out, sine)

    def test_pipeline_with_noise(self, sine):
        pipeline = AugmentationPipeline(
            transforms=[AddGaussianNoise(snr_db_range=(10, 20), p=1.0)],
            p_apply=1.0,
        )
        out = pipeline(sine, SR)
        assert out.shape == sine.shape
        assert not np.allclose(out, sine)

    def test_p_apply_zero(self, sine):
        pipeline = AugmentationPipeline(
            transforms=[AddGaussianNoise(p=1.0)],
            p_apply=0.0,
        )
        out = pipeline(sine, SR)
        np.testing.assert_array_equal(out, sine)

    def test_multiple_transforms_chain(self, sine):
        pipeline = AugmentationPipeline(
            transforms=[
                AddGaussianNoise(snr_db_range=(20, 30), p=1.0),
                VolumeJitter(gain_db_range=(-3, 3), p=1.0),
                TimeShift(shift_max_s=0.3, p=1.0),
            ],
            p_apply=1.0,
        )
        out = pipeline(sine, SR)
        assert out.shape == sine.shape
        assert out.dtype == np.float32

    def test_output_finite(self, sine):
        pipeline = AugmentationPipeline(
            transforms=[
                AddGaussianNoise(p=0.8),
                VolumeJitter(p=0.8),
            ],
            p_apply=1.0,
        )
        out = pipeline(sine, SR)
        assert np.all(np.isfinite(out))


class TestSpectrogramAugmentationPipeline:
    def test_empty_pipeline(self, mel_spec):
        pipeline = SpectrogramAugmentationPipeline(transforms=[])
        out = pipeline(mel_spec)
        np.testing.assert_array_equal(out, mel_spec)

    def test_shape_preserved(self, mel_spec):
        pipeline = SpectrogramAugmentationPipeline(
            transforms=[
                SpecAugmentFreq(max_mask_pct=0.12, p=1.0),
                SpecAugmentTime(max_mask_pct=0.12, p=1.0),
            ]
        )
        out = pipeline(mel_spec)
        assert out.shape == mel_spec.shape

    def test_output_dtype(self, mel_spec):
        pipeline = SpectrogramAugmentationPipeline(transforms=[RandomErasing(p=1.0)])
        out = pipeline(mel_spec)
        assert out.dtype == np.float32


# =============================================================================
# 4. PRESETS POR TAXON
# =============================================================================


class TestPresets:
    @pytest.mark.parametrize("taxon", ["bats", "birds", "frogs", "insects", "mammals", "reptiles"])
    def test_preset_returns_pipeline(self, taxon):
        pipeline = get_preset(taxon)
        assert isinstance(pipeline, AugmentationPipeline)

    @pytest.mark.parametrize("taxon", ["bats", "birds", "frogs", "insects", "mammals", "reptiles"])
    def test_preset_runs_on_audio(self, taxon, sine):
        pipeline = get_preset(taxon)
        out = pipeline(sine, SR)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert len(out) > 0

    def test_unknown_taxon_raises(self):
        with pytest.raises((KeyError, ValueError)):
            get_preset("unicorn")

    def test_bats_preset_high_snr_tolerance(self):
        """Bats: alta frecuencia, augmentation mas conservadora."""
        pipeline = get_preset("bats")
        # El preset de bats debe existir y ser aplicable
        t = np.linspace(0, 0.5, int(192_000 * 0.5), endpoint=False)
        bat_audio = (0.3 * np.sin(2 * np.pi * 40_000 * t)).astype(np.float32)
        out = pipeline(bat_audio, 192_000)
        assert len(out) > 0
        assert np.all(np.isfinite(out))
