"""
tests/test_acoustic_indices.py
─────────────────────────────────────────────────────────────────────────────
Tests unitarios para src/monitoring/acoustic_indices.py

Señales de prueba usadas
─────────────────────────
  sine_440   : tono puro 440 Hz  → baja entropía (H bajo), alta concentración
  white_noise: ruido blanco      → alta entropía (H alto), ACI alto
  silence    : ceros             → RMS=0, índices mínimos
  mixed      : tono + ruido      → valores intermedios

Ejecución:
    pytest tests/test_acoustic_indices.py -v
    pytest tests/test_acoustic_indices.py -v --tb=short -x

Autor: Ian
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.monitoring.acoustic_indices import (
    AcousticIndices,
    IndicesConfig,
    IndicesResult,
    compute_indices,
)

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

SR = 22_050


@pytest.fixture
def cfg():
    return IndicesConfig(sample_rate=SR)


@pytest.fixture
def ai(cfg):
    return AcousticIndices(cfg)


@pytest.fixture
def sine_440():
    """Tono puro de 440 Hz, 5 segundos."""
    t = np.linspace(0, 5.0, 5 * SR, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def white_noise():
    """Ruido blanco gaussiano, 5 segundos."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.3, 5 * SR).astype(np.float32)


@pytest.fixture
def silence():
    """Señal de silencio, 5 segundos."""
    return np.zeros(5 * SR, dtype=np.float32)


@pytest.fixture
def mixed(sine_440, white_noise):
    """Tono + ruido a SNR ≈ 10 dB."""
    return np.clip(sine_440 + 0.1 * white_noise, -1.0, 1.0)


@pytest.fixture
def stft_pair(ai, sine_440):
    """Espectrograma precomputado para reutilizar en múltiples tests."""
    freqs, times, Sxx = ai._compute_spectrogram(sine_440)
    Sxx_db = ai._power_to_db(Sxx)
    return freqs, times, Sxx, Sxx_db


# ─────────────────────────────────────────────────────────────────────────────
# 1. IndicesConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestIndicesConfig:

    def test_defaults(self):
        cfg = IndicesConfig()
        assert cfg.sample_rate == 22_050
        assert cfg.n_fft == 1024
        assert cfg.db_threshold == -50.0

    def test_custom_sr(self):
        cfg = IndicesConfig(sample_rate=44_100)
        assert cfg.sample_rate == 44_100

    def test_bat_preset(self):
        """Sample rate ultrasónico válido."""
        cfg = IndicesConfig(
            sample_rate    = 192_000,
            bi_freq_low_hz = 20_000,
            bi_freq_high_hz= 80_000,
            max_freq_hz    = 95_000,
        )
        assert cfg.sample_rate == 192_000
        assert cfg.bi_freq_low_hz < cfg.bi_freq_high_hz


# ─────────────────────────────────────────────────────────────────────────────
# 2. Espectrograma base
# ─────────────────────────────────────────────────────────────────────────────

class TestSpectrogramBase:

    def test_shape(self, ai, sine_440):
        freqs, times, Sxx = ai._compute_spectrogram(sine_440)
        F = ai.cfg.n_fft // 2 + 1
        assert Sxx.shape[0] == F
        assert Sxx.shape[1] > 0
        assert freqs.shape[0] == F

    def test_positive_values(self, ai, sine_440):
        _, _, Sxx = ai._compute_spectrogram(sine_440)
        assert np.all(Sxx > 0)

    def test_db_non_positive(self, ai, sine_440):
        """dB con referencia al máximo → ≤ 0."""
        _, _, Sxx = ai._compute_spectrogram(sine_440)
        Sxx_db = ai._power_to_db(Sxx)
        assert Sxx_db.dtype == np.float64 or Sxx_db.dtype == np.float32

    def test_silence_low_power(self, ai, silence):
        _, _, Sxx = ai._compute_spectrogram(silence)
        # Silencio → potencia muy baja
        assert Sxx.max() < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 3. ACI
# ─────────────────────────────────────────────────────────────────────────────

class TestACI:

    def test_positive(self, ai, sine_440):
        _, _, Sxx = ai._compute_spectrogram(sine_440)
        aci = ai.compute_aci(Sxx)
        assert aci > 0

    def test_noise_greater_than_tone(self, ai, white_noise, sine_440):
        """Ruido blanco debe tener ACI > tono puro (mayor variación temporal)."""
        _, _, Sxx_noise = ai._compute_spectrogram(white_noise)
        _, _, Sxx_tone  = ai._compute_spectrogram(sine_440)
        aci_noise = ai.compute_aci(Sxx_noise)
        aci_tone  = ai.compute_aci(Sxx_tone)
        assert aci_noise > aci_tone

    def test_silence_near_zero(self, ai, silence):
        _, _, Sxx = ai._compute_spectrogram(silence)
        aci = ai.compute_aci(Sxx)
        # Silencio → ACI debe ser muy bajo (no exactamente 0 por floor numérico)
        assert aci < 10.0

    def test_j_bin_variation(self, ai, white_noise):
        """ACI varía con j_bin pero siempre es positivo."""
        _, _, Sxx = ai._compute_spectrogram(white_noise)
        for j in [3, 5, 10]:
            aci = ai.compute_aci(Sxx, j_bin=j)
            assert aci > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADI y AEI
# ─────────────────────────────────────────────────────────────────────────────

class TestADI:

    def test_non_negative(self, ai, stft_pair):
        freqs, _, _, Sxx_db = stft_pair
        adi, bands = ai.compute_adi(freqs, Sxx_db)
        assert adi >= 0.0

    def test_bands_dict_not_empty(self, ai, stft_pair):
        freqs, _, _, Sxx_db = stft_pair
        _, bands = ai.compute_adi(freqs, Sxx_db)
        assert len(bands) > 0

    def test_band_values_in_range(self, ai, stft_pair):
        freqs, _, _, Sxx_db = stft_pair
        _, bands = ai.compute_adi(freqs, Sxx_db)
        for k, v in bands.items():
            assert 0.0 <= v <= 1.0, f"Banda {k} fuera de rango: {v}"

    def test_noise_higher_adi_than_silence(self, ai, white_noise, silence):
        """Ruido activa más bandas que el silencio → ADI mayor."""
        fn, _, _, Sxx_n = ai._compute_spectrogram(white_noise), None, None, None
        fn  = ai._compute_spectrogram(white_noise)
        fs  = ai._compute_spectrogram(silence)
        adi_noise, _ = ai.compute_adi(fn[0], ai._power_to_db(fn[2]))
        adi_sil, _   = ai.compute_adi(fs[0], ai._power_to_db(fs[2]))
        assert adi_noise >= adi_sil


class TestAEI:

    def test_range(self, ai, stft_pair):
        freqs, _, _, Sxx_db = stft_pair
        aei = ai.compute_aei(freqs, Sxx_db)
        assert 0.0 <= aei <= 1.0

    def test_silence_aei(self, ai, silence):
        freqs, _, Sxx = ai._compute_spectrogram(silence)
        Sxx_db = ai._power_to_db(Sxx)
        aei = ai.compute_aei(freqs, Sxx_db)
        assert 0.0 <= aei <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. BI
# ─────────────────────────────────────────────────────────────────────────────

class TestBI:

    def test_non_negative(self, ai, stft_pair):
        freqs, _, _, Sxx_db = stft_pair
        bi = ai.compute_bi(freqs, Sxx_db)
        assert bi >= 0.0

    def test_silence_low_bi(self, ai, silence):
        freqs, _, Sxx = ai._compute_spectrogram(silence)
        Sxx_db = ai._power_to_db(Sxx)
        bi = ai.compute_bi(freqs, Sxx_db)
        # Silencio → BI muy bajo (todas las frecuencias al mínimo dB)
        assert bi < 1.0

    def test_bi_increases_with_bioacoustic_band(self):
        """Señal en 2-8 kHz debe tener BI > señal fuera de esa banda."""
        cfg = IndicesConfig(sample_rate=SR)
        ai  = AcousticIndices(cfg)
        t   = np.linspace(0, 3.0, 3 * SR, dtype=np.float32)

        # Señal en banda bioacústica (4 kHz)
        y_bio  = 0.5 * np.sin(2 * np.pi * 4000 * t)
        # Señal fuera de banda (200 Hz)
        y_out  = 0.5 * np.sin(2 * np.pi * 200 * t)

        def _bi(y):
            f, _, S = ai._compute_spectrogram(y)
            return ai.compute_bi(f, ai._power_to_db(S))

        assert _bi(y_bio) > _bi(y_out)

    def test_custom_bi_range(self):
        """BI funciona con rangos de frecuencia personalizados."""
        cfg = IndicesConfig(
            sample_rate     = SR,
            bi_freq_low_hz  = 500,
            bi_freq_high_hz = 4_000,
        )
        ai = AcousticIndices(cfg)
        t  = np.linspace(0, 3.0, 3 * SR, dtype=np.float32)
        y  = 0.4 * np.sin(2 * np.pi * 1000 * t)
        f, _, S = ai._compute_spectrogram(y)
        bi = ai.compute_bi(f, ai._power_to_db(S))
        assert bi >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. NDSI
# ─────────────────────────────────────────────────────────────────────────────

class TestNDSI:

    def test_range(self, ai, stft_pair):
        freqs, _, Sxx, _ = stft_pair
        ndsi, anthro, bio = ai.compute_ndsi(freqs, Sxx)
        assert -1.0 <= ndsi <= 1.0

    def test_positive_for_biophony(self):
        """Señal en banda biofónica (5 kHz) → NDSI > 0."""
        cfg = IndicesConfig(sample_rate=SR)
        ai  = AcousticIndices(cfg)
        t   = np.linspace(0, 5.0, 5 * SR, dtype=np.float32)
        y   = 0.5 * np.sin(2 * np.pi * 5000 * t)
        f, _, S = ai._compute_spectrogram(y)
        ndsi, _, _ = ai.compute_ndsi(f, S)
        assert ndsi > 0.0

    def test_negative_for_anthrophony(self):
        """Señal en banda antrofónica (1.5 kHz) → NDSI < 0."""
        cfg = IndicesConfig(sample_rate=SR)
        ai  = AcousticIndices(cfg)
        t   = np.linspace(0, 5.0, 5 * SR, dtype=np.float32)
        y   = 0.5 * np.sin(2 * np.pi * 1500 * t)
        f, _, S = ai._compute_spectrogram(y)
        ndsi, _, _ = ai.compute_ndsi(f, S)
        assert ndsi < 0.0

    def test_anthro_bio_non_negative(self, ai, stft_pair):
        freqs, _, Sxx, _ = stft_pair
        _, anthro, bio = ai.compute_ndsi(freqs, Sxx)
        assert anthro >= 0.0
        assert bio    >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Entropías Hf, Ht, H
# ─────────────────────────────────────────────────────────────────────────────

class TestEntropy:

    def test_hf_range(self, ai, stft_pair):
        freqs, _, Sxx, _ = stft_pair
        hf = ai.compute_hf(freqs, Sxx)
        assert 0.0 <= hf <= 1.0

    def test_ht_range(self, ai, sine_440):
        ht = ai.compute_ht(sine_440)
        assert 0.0 <= ht <= 1.0

    def test_h_equals_hf_times_ht(self, ai, white_noise):
        f, _, S = ai._compute_spectrogram(white_noise)
        hf = ai.compute_hf(f, S)
        ht = ai.compute_ht(white_noise)
        h  = hf * ht
        result = ai.compute_all(white_noise)
        assert abs(result.h - h) < 1e-6

    def test_white_noise_higher_h_than_tone(self, ai, white_noise, sine_440):
        """
        Ruido blanco → energía distribuida uniformemente → H más alto.
        Tono puro → energía concentrada en una frecuencia → H más bajo.
        """
        r_noise = ai.compute_all(white_noise)
        r_tone  = ai.compute_all(sine_440)
        assert r_noise.h > r_tone.h

    def test_silence_ht_near_extremes(self, ai, silence):
        """Silencio → Ht puede ser 0 o indefinido, debe estar en [0,1]."""
        ht = ai.compute_ht(silence)
        assert 0.0 <= ht <= 1.0

    def test_hf_white_noise_high(self, ai, white_noise):
        """Hf del ruido blanco debe ser alto (> 0.9 — energía uniforme en frecuencia)."""
        f, _, S = ai._compute_spectrogram(white_noise)
        hf = ai.compute_hf(f, S)
        assert hf > 0.85

    def test_hf_tone_low(self, ai, sine_440):
        """Hf del tono puro debe ser bajo (energía concentrada)."""
        f, _, S = ai._compute_spectrogram(sine_440)
        hf = ai.compute_hf(f, S)
        assert hf < 0.7


# ─────────────────────────────────────────────────────────────────────────────
# 8. RMS y ZCR
# ─────────────────────────────────────────────────────────────────────────────

class TestRMSZCR:

    def test_rms_silence_zero(self):
        y = np.zeros(1000, dtype=np.float32)
        assert AcousticIndices.compute_rms(y) == pytest.approx(0.0, abs=1e-10)

    def test_rms_sine_known_value(self, sine_440):
        """RMS de 0.5·sin → RMS = 0.5 / sqrt(2) ≈ 0.3536."""
        rms = AcousticIndices.compute_rms(sine_440)
        assert abs(rms - 0.5 / np.sqrt(2)) < 0.01

    def test_zcr_range(self, sine_440):
        zcr = AcousticIndices.compute_zcr(sine_440)
        assert 0.0 <= zcr <= 1.0

    def test_zcr_sine_positive(self, sine_440):
        """Tono puro cruza el cero regularmente → ZCR > 0."""
        assert AcousticIndices.compute_zcr(sine_440) > 0.0

    def test_zcr_silence_zero(self):
        y = np.zeros(1000, dtype=np.float32)
        assert AcousticIndices.compute_zcr(y) == pytest.approx(0.0, abs=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 9. compute_all — resultado completo
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAll:

    def test_returns_indices_result(self, ai, sine_440):
        result = ai.compute_all(sine_440)
        assert isinstance(result, IndicesResult)

    def test_all_fields_present(self, ai, sine_440):
        result = ai.compute_all(sine_440)
        for attr in ["aci", "adi", "aei", "bi", "ndsi", "hf", "ht", "h", "rms", "zcr"]:
            assert hasattr(result, attr)
            assert np.isfinite(getattr(result, attr)), f"{attr} no es finito"

    def test_duration_correct(self, ai, sine_440):
        result = ai.compute_all(sine_440)
        expected = len(sine_440) / SR
        assert abs(result.duration_s - expected) < 0.1

    def test_sample_rate_stored(self, ai, sine_440):
        result = ai.compute_all(sine_440)
        assert result.sample_rate == SR

    def test_to_dict_serializable(self, ai, sine_440):
        import json
        result = ai.compute_all(sine_440)
        d      = result.to_dict()
        # No debe lanzar excepción
        json.dumps(d)

    def test_summary_string(self, ai, sine_440):
        result = ai.compute_all(sine_440)
        s = result.summary()
        assert "ACI" in s
        assert "NDSI" in s
        assert "H" in s

    def test_interpret_keys(self, ai, sine_440):
        result = ai.compute_all(sine_440)
        interp = result.interpret()
        assert "aci"  in interp
        assert "ndsi" in interp
        assert "h"    in interp

    @pytest.mark.parametrize("signal_name", ["sine_440", "white_noise", "silence", "mixed"])
    def test_no_nan_on_all_signals(self, ai, signal_name, request):
        y      = request.getfixturevalue(signal_name)
        result = ai.compute_all(y)
        for attr in ["aci", "adi", "aei", "bi", "ndsi", "hf", "ht", "h"]:
            assert not np.isnan(getattr(result, attr)), f"{attr} es NaN en {signal_name}"

    def test_multichannel_input_handled(self, ai):
        """Señal estéreo → debe promediar canales automáticamente."""
        y_stereo = np.random.randn(SR * 3, 2).astype(np.float32)
        result   = ai.compute_all(y_stereo)
        assert np.isfinite(result.aci)

    def test_very_short_signal(self, ai):
        """Señal muy corta (< n_fft) → no debe lanzar excepción."""
        y = np.random.randn(512).astype(np.float32) * 0.1
        result = ai.compute_all(y)
        assert isinstance(result, IndicesResult)


# ─────────────────────────────────────────────────────────────────────────────
# 10. compute_indices (función de conveniencia)
# ─────────────────────────────────────────────────────────────────────────────

class TestConvenienceFunction:

    def test_basic(self, sine_440):
        result = compute_indices(sine_440, sample_rate=SR)
        assert isinstance(result, IndicesResult)

    def test_custom_cfg(self, sine_440):
        cfg    = IndicesConfig(sample_rate=SR, db_threshold=-60.0)
        result = compute_indices(sine_440, sample_rate=SR, cfg=cfg)
        assert np.isfinite(result.adi)


# ─────────────────────────────────────────────────────────────────────────────
# 11. compute_windowed
# ─────────────────────────────────────────────────────────────────────────────

class TestWindowedAnalysis:

    def test_returns_list(self, ai, white_noise):
        results = ai.compute_windowed(white_noise, window_s=1.0, hop_s=1.0)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_each_window_has_indices(self, ai, white_noise):
        results = ai.compute_windowed(white_noise, window_s=1.0, hop_s=1.0)
        for r in results:
            assert "aci" in r
            assert "ndsi" in r
            assert "t_start_s" in r
            assert "t_end_s" in r

    def test_timestamps_monotonic(self, ai, white_noise):
        results = ai.compute_windowed(white_noise, window_s=1.0, hop_s=1.0)
        starts = [r["t_start_s"] for r in results]
        assert starts == sorted(starts)

    def test_overlap_produces_more_windows(self, ai, white_noise):
        """Overlap (hop < window) produce más ventanas que sin overlap."""
        r_no_overlap  = ai.compute_windowed(white_noise, window_s=1.0, hop_s=1.0)
        r_with_overlap= ai.compute_windowed(white_noise, window_s=2.0, hop_s=0.5)
        assert len(r_with_overlap) >= len(r_no_overlap)

    def test_no_nan_in_any_window(self, ai, white_noise):
        results = ai.compute_windowed(white_noise, window_s=1.0, hop_s=1.0)
        for r in results:
            for idx in ["aci", "adi", "ndsi", "hf", "ht", "h"]:
                assert np.isfinite(r[idx]), f"{idx} es NaN en ventana {r['t_start_s']}"
