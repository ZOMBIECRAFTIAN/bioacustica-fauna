"""
tests/test_preprocessor.py
─────────────────────────────────────────────────────────────────────────────
Tests unitarios para src/audio_processing/preprocessor.py.

Ejecutar:
    pytest tests/test_preprocessor.py -v
    pytest tests/ -v --tb=short

Autor: Ian
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import tempfile
import soundfile as sf

from src.audio_processing.preprocessor import (
    AudioConfig,
    AudioPreprocessor,
    PRESETS,
    validate_audio_file,
    batch_validate,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

SR = 22_050

@pytest.fixture
def sine_audio():
    """Señal sinusoidal pura de 3 segundos a 440 Hz."""
    t = np.linspace(0, 3.0, 3 * SR, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def noisy_audio():
    """Señal con ruido gaussiano."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.1, 3 * SR).astype(np.float32)


@pytest.fixture
def silence_audio():
    return np.zeros(3 * SR, dtype=np.float32)


@pytest.fixture
def tmp_wav(sine_audio):
    """Archivo WAV temporal de una señal sinusoidal."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, sine_audio, SR, subtype="PCM_16")
        yield Path(f.name)


@pytest.fixture
def default_proc():
    return AudioPreprocessor(AudioConfig(
        sample_rate=SR,
        apply_noise_reduction=False,
        apply_bandpass=True,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE AUDIO
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioLoad:

    def test_load_wav(self, default_proc, tmp_wav):
        y, sr = default_proc.load(tmp_wav)
        assert isinstance(y, np.ndarray)
        assert sr == SR
        assert y.dtype == np.float32
        assert len(y) > 0

    def test_load_nonexistent_raises(self, default_proc):
        with pytest.raises(FileNotFoundError):
            default_proc.load("nonexistent_file.wav")

    def test_load_shape_mono(self, default_proc, tmp_wav):
        y, sr = default_proc.load(tmp_wav, mono=True)
        assert y.ndim == 1

    def test_load_duration(self, default_proc, tmp_wav):
        y, sr = default_proc.load(tmp_wav)
        duration = len(y) / sr
        assert abs(duration - 3.0) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# 2. FILTRO PASA-BANDA
# ─────────────────────────────────────────────────────────────────────────────

class TestBandpassFilter:

    def test_output_shape_preserved(self, default_proc, sine_audio):
        filtered = default_proc.bandpass_filter(sine_audio)
        assert filtered.shape == sine_audio.shape

    def test_output_dtype(self, default_proc, sine_audio):
        filtered = default_proc.bandpass_filter(sine_audio)
        assert filtered.dtype == np.float32

    def test_filters_high_frequency(self):
        """Señal de 10kHz debe ser atenuada con freq_high=5000."""
        cfg = AudioConfig(sample_rate=SR, freq_low=100, freq_high=5_000)
        proc = AudioPreprocessor(cfg)
        t = np.linspace(0, 1, SR, dtype=np.float32)
        high_freq = np.sin(2 * np.pi * 10_000 * t)
        filtered = proc.bandpass_filter(high_freq)
        # Energía RMS debe reducirse
        assert np.sqrt(np.mean(filtered**2)) < np.sqrt(np.mean(high_freq**2))

    def test_silence_stays_near_zero(self, default_proc, silence_audio):
        filtered = default_proc.bandpass_filter(silence_audio)
        assert np.allclose(filtered, 0.0, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 3. NORMALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalization:

    @pytest.mark.parametrize("method", ["peak", "rms", "lufs"])
    def test_range_after_normalization(self, default_proc, sine_audio, method):
        norm = default_proc.normalize(sine_audio, method=method)
        assert norm.max() <= 1.0 + 1e-6
        assert norm.min() >= -1.0 - 1e-6

    def test_peak_normalization_max_is_one(self, default_proc, sine_audio):
        norm = default_proc.normalize(sine_audio, method="peak")
        assert abs(np.max(np.abs(norm)) - 1.0) < 1e-5

    def test_silence_normalization_no_nan(self, default_proc, silence_audio):
        norm = default_proc.normalize(silence_audio, method="peak")
        assert not np.any(np.isnan(norm))
        assert not np.any(np.isinf(norm))

    def test_output_dtype(self, default_proc, sine_audio):
        norm = default_proc.normalize(sine_audio)
        assert norm.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# 4. DETECCIÓN DE EVENTOS (VAD)
# ─────────────────────────────────────────────────────────────────────────────

class TestVAD:

    def test_silence_returns_no_events(self, default_proc, silence_audio):
        events = default_proc.detect_events(silence_audio)
        assert len(events) == 0

    def test_active_signal_returns_events(self, default_proc, sine_audio):
        events = default_proc.detect_events(sine_audio)
        assert len(events) >= 1

    def test_event_timestamps_valid(self, default_proc, sine_audio):
        events = default_proc.detect_events(sine_audio)
        for t_start, t_end in events:
            assert t_start >= 0.0
            assert t_end > t_start
            assert t_end <= len(sine_audio) / SR + 0.1

    def test_event_tuple_format(self, default_proc, sine_audio):
        events = default_proc.detect_events(sine_audio)
        for ev in events:
            assert len(ev) == 2
            assert isinstance(ev[0], float)
            assert isinstance(ev[1], float)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SEGMENTACIÓN
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentation:

    def test_segment_length(self, default_proc, sine_audio):
        segs = default_proc.segment(sine_audio)
        expected_len = int(default_proc.cfg.segment_duration * SR)
        for seg in segs:
            assert len(seg) == expected_len

    def test_segments_not_empty(self, default_proc, sine_audio):
        segs = default_proc.segment(sine_audio)
        assert len(segs) >= 1

    def test_short_audio_gets_padded(self, default_proc):
        short = np.zeros(100, dtype=np.float32)
        segs = default_proc.segment(short)
        assert len(segs) == 1
        assert len(segs[0]) == int(default_proc.cfg.segment_duration * SR)

    def test_segment_dtype(self, default_proc, sine_audio):
        segs = default_proc.segment(sine_audio)
        for seg in segs:
            assert seg.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# 6. ESPECTROGRAMA MEL
# ─────────────────────────────────────────────────────────────────────────────

class TestMelSpectrogram:

    def test_output_shape(self, default_proc, sine_audio):
        segs = default_proc.segment(sine_audio)
        mel  = default_proc.mel_spectrogram(segs[0])
        assert mel.ndim == 2
        assert mel.shape[0] == default_proc.cfg.n_mels

    def test_output_dtype(self, default_proc, sine_audio):
        segs = default_proc.segment(sine_audio)
        mel  = default_proc.mel_spectrogram(segs[0])
        assert mel.dtype == np.float32

    def test_mel_is_db(self, default_proc, sine_audio):
        """El espectrograma en dB tiene valores negativos y el máximo ≤ 0."""
        segs = default_proc.segment(sine_audio)
        mel  = default_proc.mel_spectrogram(segs[0])
        assert mel.max() <= 0.1     # dB: referencia np.max → max ≈ 0

    def test_silence_mel_all_very_negative(self, default_proc, silence_audio):
        """Silencio debe producir mel muy negativo."""
        segs = default_proc.segment(silence_audio)
        mel  = default_proc.mel_spectrogram(segs[0])
        assert mel.max() < -40


# ─────────────────────────────────────────────────────────────────────────────
# 7. MFCC
# ─────────────────────────────────────────────────────────────────────────────

class TestMFCC:

    def test_shape_without_delta(self, default_proc, sine_audio):
        segs = default_proc.segment(sine_audio)
        mfcc = default_proc.mfcc(segs[0], include_delta=False)
        assert mfcc.shape[0] == default_proc.cfg.n_mfcc

    def test_shape_with_delta(self, default_proc, sine_audio):
        segs = default_proc.segment(sine_audio)
        mfcc = default_proc.mfcc(segs[0], include_delta=True)
        assert mfcc.shape[0] == default_proc.cfg.n_mfcc * 3

    def test_dtype(self, default_proc, sine_audio):
        segs = default_proc.segment(sine_audio)
        mfcc = default_proc.mfcc(segs[0])
        assert mfcc.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURES ESPECTRALES
# ─────────────────────────────────────────────────────────────────────────────

class TestSpectralFeatures:

    EXPECTED_KEYS = {
        "zcr_mean", "zcr_std",
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_flatness_mean", "spectral_flatness_std",
        "chroma_mean", "chroma_std",
        "rms_mean", "rms_std",
    }

    def test_keys_present(self, default_proc, sine_audio):
        segs  = default_proc.segment(sine_audio)
        feats = default_proc.spectral_features(segs[0])
        assert self.EXPECTED_KEYS.issubset(set(feats.keys()))

    def test_values_are_finite(self, default_proc, sine_audio):
        segs  = default_proc.segment(sine_audio)
        feats = default_proc.spectral_features(segs[0])
        for k, v in feats.items():
            assert np.isfinite(v), f"Feature {k} no es finito: {v}"

    def test_spectral_centroid_positive(self, default_proc, sine_audio):
        segs  = default_proc.segment(sine_audio)
        feats = default_proc.spectral_features(segs[0])
        assert feats["spectral_centroid_mean"] > 0

    def test_zcr_between_zero_one(self, default_proc, sine_audio):
        segs  = default_proc.segment(sine_audio)
        feats = default_proc.spectral_features(segs[0])
        assert 0 <= feats["zcr_mean"] <= 1


# ─────────────────────────────────────────────────────────────────────────────
# 9. PRESETS
# ─────────────────────────────────────────────────────────────────────────────

class TestPresets:

    @pytest.mark.parametrize("preset_name", ["bats", "frogs", "insects", "mammals", "reptiles"])
    def test_preset_exists(self, preset_name):
        assert preset_name in PRESETS

    def test_bats_preset_high_sr(self):
        cfg = PRESETS["bats"]
        assert cfg.sample_rate >= 100_000

    def test_frogs_preset_valid_range(self):
        cfg = PRESETS["frogs"]
        assert cfg.freq_low < cfg.freq_high

    @pytest.mark.parametrize("preset_name", ["frogs", "insects", "mammals"])
    def test_preset_produces_mel(self, preset_name, sine_audio):
        cfg  = PRESETS[preset_name]
        # Resamplear dummy a la SR del preset
        import librosa
        y = librosa.resample(sine_audio, orig_sr=SR, target_sr=cfg.sample_rate)
        proc = AudioPreprocessor(cfg)
        segs = proc.segment(y)
        if segs:
            mel = proc.mel_spectrogram(segs[0])
            assert mel.shape[0] == cfg.n_mels


# ─────────────────────────────────────────────────────────────────────────────
# 10. PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineComplete:

    def test_process_returns_expected_keys(self, default_proc, tmp_wav):
        result = default_proc.process(tmp_wav)
        assert "file" in result
        assert "sr" in result
        assert "duration" in result
        assert "events" in result
        assert "segments" in result

    def test_process_segment_has_features(self, default_proc, tmp_wav):
        result = default_proc.process(tmp_wav)
        assert len(result["segments"]) >= 1
        seg = result["segments"][0]
        assert "features" in seg
        assert "mel_spectrogram" in seg["features"]
        assert "mfcc" in seg["features"]
        assert "spectral" in seg["features"]

    def test_process_with_return_segments(self, default_proc, tmp_wav):
        result = default_proc.process(tmp_wav, return_segments=True)
        for seg in result["segments"]:
            assert "audio" in seg
            assert isinstance(seg["audio"], np.ndarray)

    def test_validate_audio_file(self, tmp_wav):
        info = validate_audio_file(tmp_wav)
        assert info["valid"] is True
        assert info["sr"] == SR
        assert info["duration"] > 0

    def test_validate_nonexistent(self):
        info = validate_audio_file("nonexistent.wav")
        assert info["valid"] is False
        assert info["error"] is not None
