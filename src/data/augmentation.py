"""
src/data/augmentation.py
─────────────────────────────────────────────────────────────────────────────
Augmentación bioacústica avanzada para entrenamiento de modelos.

Transformaciones implementadas
──────────────────────────────
DOMINIO TEMPORAL (waveform):
  - PitchShift         : desplazamiento de tono ±n semitonos (librosa)
  - TimeStretch        : estiramiento temporal con phase vocoder
  - AddBackgroundNoise : mezcla con grabaciones de ruido a SNR objetivo
  - AddGaussianNoise   : ruido blanco gaussiano
  - RandomClip         : recorte aleatorio de la señal
  - VolumeJitter       : ganancia aleatoria en dB
  - TimeShift          : desplazamiento temporal circular

DOMINIO ESPECTRAL (spectrogram):
  - SpecAugmentFreq    : enmascarado de bandas de frecuencia (Park et al. 2019)
  - SpecAugmentTime    : enmascarado de franjas temporales
  - FrequencyShift     : desplazamiento vertical en el espectrograma mel
  - RandomErasing      : borrado aleatorio de parches rectangulares
  - GaussianBlur       : suavizado gaussiano

PIPELINES PREDEFINIDOS:
  - light_augment()    : pitch ± time stretch (bajo costo)
  - standard_augment() : noise + SpecAugment (recomendado entrenamiento)
  - heavy_augment()    : todas las transformaciones (máxima variabilidad)
  - get_preset()       : pipeline por grupo taxonómico

Uso
───
    from src.data.augmentation import standard_augment, AugmentationPipeline

    pipeline = AugmentationPipeline([
        AddBackgroundNoise(noise_dir="data/noise", target_snr_db=15),
        PitchShift(n_steps_range=(-2, 2)),
        SpecAugmentFreq(max_mask_pct=0.15),
    ], p_apply=0.8)

    y_aug = pipeline(y, sr=22050)

Referencia
──────────
  Park et al. (2019). SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition. INTERSPEECH.
  Salamon & Bello (2017). Deep Convolutional Neural Networks and Data
    Augmentation for Environmental Sound Classification. IEEE SPL.

Autor: Ian
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Importación opcional de librosa (puede no estar instalado en entornos mínimos)
try:
    import librosa

    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    logger.warning("librosa no disponible — PitchShift y TimeStretch desactivados.")

try:
    import soundfile as sf

    _SF_OK = True
except ImportError:
    _SF_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ BASE
# ─────────────────────────────────────────────────────────────────────────────


class AudioTransform(ABC):
    """
    Transformación base. Todas las clases derivan de ésta.

    Parameters
    ----------
    p : float — probabilidad de aplicar la transformación [0, 1]
    """

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between 0.0 and 1.0")
        self.p = p

    def __call__(self, y: np.ndarray, sr: int) -> np.ndarray:
        if random.random() < self.p:
            return self.apply(y, sr)
        return y

    @abstractmethod
    def apply(self, y: np.ndarray, sr: int) -> np.ndarray: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMACIONES TEMPORALES (WAVEFORM)
# ─────────────────────────────────────────────────────────────────────────────


class PitchShift(AudioTransform):
    """
    Desplaza el tono ±n semitonos sin alterar la duración.
    Usa librosa.effects.pitch_shift (phase vocoder).

    Parámetros acústicos relevantes
    ────────────────────────────────
    Cantos de aves / ranas: ±1-3 semitonos simula variación inter-individual.
    Murciélagos: ±1-2 kHz en escala absoluta → equivale a pocos semitonos.
    """

    def __init__(self, n_steps_range: tuple[float, float] = (-2.0, 2.0), p: float = 0.5):
        super().__init__(p)
        self.lo, self.hi = n_steps_range

    def apply(self, y: np.ndarray, sr: int) -> np.ndarray:
        if not _LIBROSA_OK:
            return y
        n = random.uniform(self.lo, self.hi)
        try:
            return librosa.effects.pitch_shift(y, sr=sr, n_steps=n)
        except Exception as e:
            logger.debug("PitchShift falló: %s", e)
            return y


class TimeStretch(AudioTransform):
    """
    Estiramiento temporal con phase vocoder.
    rate < 1 → más lento; rate > 1 → más rápido.
    Preserva el tono.

    Límites recomendados
    ─────────────────────
    aves/ranas : rate ∈ [0.8, 1.2]
    insectos   : rate ∈ [0.9, 1.1]  (cantos muy periódicos, sensibles)
    murciélagos: no recomendado (pulsos de ecolocalización temporalmente críticos)
    """

    def __init__(self, rate_range: tuple[float, float] = (0.85, 1.15), p: float = 0.5):
        super().__init__(p)
        self.lo, self.hi = rate_range

    def apply(self, y: np.ndarray, sr: int) -> np.ndarray:
        if not _LIBROSA_OK:
            return y
        rate = random.uniform(self.lo, self.hi)
        try:
            y_stretch = librosa.effects.time_stretch(y, rate=rate)
            # Ajustar longitud original con padding/trim
            if len(y_stretch) > len(y):
                return y_stretch[: len(y)]
            elif len(y_stretch) < len(y):
                return np.pad(y_stretch, (0, len(y) - len(y_stretch)))
            return y_stretch
        except Exception as e:
            logger.debug("TimeStretch falló: %s", e)
            return y


class AddGaussianNoise(AudioTransform):
    """
    Ruido blanco gaussiano a SNR objetivo.

    SNR (dB) = 10 * log10(P_señal / P_ruido)
    P_ruido  = P_señal / 10^(SNR/10)

    Rango útil: SNR ∈ [10, 40] dB para especies con vocalización clara.
    """

    def __init__(self, snr_db_range: tuple[float, float] = (20.0, 40.0), p: float = 0.5):
        super().__init__(p)
        self.lo, self.hi = snr_db_range

    def apply(self, y: np.ndarray, sr: int) -> np.ndarray:
        snr_db = random.uniform(self.lo, self.hi)
        p_sig = np.mean(y**2) + 1e-10
        p_noise = p_sig / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(p_noise), len(y)).astype(y.dtype)
        return np.clip(y + noise, -1.0, 1.0)


class AddBackgroundNoise(AudioTransform):
    """
    Mezcla la señal con grabaciones de ruido de fondo (soundscapes, viento, lluvia).

    noise_dir debe contener archivos .wav/.flac de ruido.
    Si el directorio está vacío, se degrada a ruido gaussiano.

    SNR = 10 * log10(P_señal / P_mezcla)
    """

    def __init__(
        self,
        noise_dir: Union[str, Path] | None = None,
        target_snr_db: float = 15.0,
        snr_jitter_db: float = 5.0,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.target_snr = target_snr_db
        self.snr_jitter = snr_jitter_db
        self._noise_files: list[Path] = []

        if noise_dir is not None:
            noise_dir = Path(noise_dir)
            if noise_dir.is_dir():
                self._noise_files = list(noise_dir.glob("**/*.wav")) + list(
                    noise_dir.glob("**/*.flac")
                )
        if not self._noise_files:
            logger.debug("AddBackgroundNoise: sin archivos de ruido → modo gaussiano.")

    def _load_random_noise(self, target_len: int, sr: int) -> np.ndarray:
        path = random.choice(self._noise_files)
        try:
            if _SF_OK:
                noise, n_sr = sf.read(str(path), dtype="float32", always_2d=False)
            elif _LIBROSA_OK:
                noise, n_sr = librosa.load(str(path), sr=sr, mono=True)
            else:
                return np.zeros(target_len, dtype=np.float32)

            # Resamplear si es necesario
            if _LIBROSA_OK and n_sr != sr:
                noise = librosa.resample(noise, orig_sr=n_sr, target_sr=sr)

            # Ajustar longitud
            if len(noise) > target_len:
                start = random.randint(0, len(noise) - target_len)
                noise = noise[start : start + target_len]
            elif len(noise) < target_len:
                noise = np.tile(noise, target_len // len(noise) + 1)[:target_len]
            return noise.astype(np.float32)
        except Exception as e:
            logger.debug("Error cargando ruido %s: %s", path, e)
            return np.zeros(target_len, dtype=np.float32)

    def apply(self, y: np.ndarray, sr: int) -> np.ndarray:
        snr_db = self.target_snr + random.uniform(-self.snr_jitter, self.snr_jitter)

        if self._noise_files:
            noise = self._load_random_noise(len(y), sr)
        else:
            p_sig = np.mean(y**2) + 1e-10
            p_noise = p_sig / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(p_noise), len(y)).astype(y.dtype)
            return np.clip(y + noise, -1.0, 1.0)

        p_sig = np.mean(y**2) + 1e-10
        p_noise = np.mean(noise**2) + 1e-10
        # Escalar ruido al SNR objetivo
        scale = np.sqrt(p_sig / (p_noise * (10 ** (snr_db / 10))))
        return np.clip(y + scale * noise, -1.0, 1.0)


class RandomClip(AudioTransform):
    """
    Recorte aleatorio → devuelve un segmento de duración `clip_duration_s`.
    El segmento se rellena con ceros si la señal es más corta.
    """

    def __init__(self, clip_duration_s: float = 3.0, p: float = 0.5):
        super().__init__(p)
        self.clip_duration = clip_duration_s

    def apply(self, y: np.ndarray, sr: int) -> np.ndarray:
        target = int(self.clip_duration * sr)
        if len(y) <= target:
            return np.pad(y, (0, target - len(y)))
        start = random.randint(0, len(y) - target)
        return y[start : start + target]


class VolumeJitter(AudioTransform):
    """
    Ganancia aleatoria uniforme en dB.
    gain_db ~ U(lo, hi)
    y_out   = y * 10^(gain_db / 20)
    """

    def __init__(self, gain_db_range: tuple[float, float] = (-6.0, 6.0), p: float = 0.5):
        super().__init__(p)
        self.lo, self.hi = gain_db_range

    def apply(self, y: np.ndarray, sr: int) -> np.ndarray:
        gain_db = random.uniform(self.lo, self.hi)
        gain_lin = 10 ** (gain_db / 20.0)
        return np.clip(y * gain_lin, -1.0, 1.0)


class TimeShift(AudioTransform):
    """
    Desplazamiento circular aleatorio en el eje temporal.
    shift_max_s : desplazamiento máximo en segundos.
    """

    def __init__(self, shift_max_s: float = 0.5, p: float = 0.5):
        super().__init__(p)
        self.shift_max = shift_max_s

    def apply(self, y: np.ndarray, sr: int) -> np.ndarray:
        max_samples = int(self.shift_max * sr)
        shift = random.randint(-max_samples, max_samples)
        return np.roll(y, shift)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMACIONES ESPECTRALES (SPECTROGRAM 2D)
# ─────────────────────────────────────────────────────────────────────────────


class SpectrogramTransform(ABC):
    """Interfaz para transformaciones sobre espectrogramas 2D (np.ndarray o Tensor)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return self.apply(spec)
        return spec

    @abstractmethod
    def apply(self, spec: np.ndarray) -> np.ndarray: ...


class SpecAugmentFreq(SpectrogramTransform):
    """
    Enmascara bandas de frecuencia aleatorias → cero.
    spec shape: (n_mels, T) o (C, n_mels, T)

    Park et al. (2019): f ~ U(0, F), donde F = max_mask_pct * n_mels
    """

    def __init__(self, max_mask_pct: float = 0.15, n_masks: int = 2, p: float = 0.5):
        super().__init__(p)
        self.max_pct = max_mask_pct
        self.n_masks = n_masks

    def apply(self, spec: np.ndarray) -> np.ndarray:
        out = spec.copy()
        n_mels = out.shape[-2]
        for _ in range(self.n_masks):
            f = random.randint(0, max(1, int(self.max_pct * n_mels)))
            f0 = random.randint(0, max(0, n_mels - f))
            out[..., f0 : f0 + f, :] = 0.0
        return out


class SpecAugmentTime(SpectrogramTransform):
    """
    Enmascara franjas temporales aleatorias → cero.
    spec shape: (n_mels, T) o (C, n_mels, T)

    Park et al. (2019): t ~ U(0, T), donde T = max_mask_pct * T_total
    """

    def __init__(self, max_mask_pct: float = 0.15, n_masks: int = 2, p: float = 0.5):
        super().__init__(p)
        self.max_pct = max_mask_pct
        self.n_masks = n_masks

    def apply(self, spec: np.ndarray) -> np.ndarray:
        out = spec.copy()
        T = out.shape[-1]
        for _ in range(self.n_masks):
            t = random.randint(0, max(1, int(self.max_pct * T)))
            t0 = random.randint(0, max(0, T - t))
            out[..., :, t0 : t0 + t] = 0.0
        return out


class FrequencyShift(SpectrogramTransform):
    """
    Desplaza el espectrograma verticalmente (eje mel) por ±n bins.
    Las filas que quedan fuera se rellenan con el mínimo (≈ silencio en dB).
    """

    def __init__(self, max_shift_bins: int = 4, p: float = 0.5):
        super().__init__(p)
        self.max_shift = max_shift_bins

    def apply(self, spec: np.ndarray) -> np.ndarray:
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return spec
        out = np.full_like(spec, spec.min())
        n_mel = spec.shape[-2]
        if shift > 0:
            out[..., shift:, :] = spec[..., : n_mel - shift, :]
        else:
            out[..., : n_mel + shift, :] = spec[..., -shift:, :]
        return out


class RandomErasing(SpectrogramTransform):
    """
    Borra parches rectangulares aleatorios en el espectrograma.
    Equivalente a Cutout (DeVries & Taylor, 2017) adaptado para audio.

    area_pct: fracción del área total a borrar por parche.
    """

    def __init__(
        self,
        n_patches: int = 1,
        area_pct: float = 0.05,
        aspect_ratio: tuple[float, float] = (0.3, 3.3),
        fill_value: float | None = None,  # None → usa mínimo del spec
        p: float = 0.5,
    ):
        super().__init__(p)
        self.n_patches = n_patches
        self.area_pct = area_pct
        self.ar_lo, self.ar_hi = aspect_ratio
        self.fill = fill_value

    def apply(self, spec: np.ndarray) -> np.ndarray:
        out = spec.copy()
        h = spec.shape[-2]
        w = spec.shape[-1]
        fill = self.fill if self.fill is not None else spec.min()
        area = h * w * self.area_pct

        for _ in range(self.n_patches):
            ar = random.uniform(self.ar_lo, self.ar_hi)
            ph = int(np.sqrt(area * ar))
            pw = int(np.sqrt(area / ar))
            ph = min(ph, h)
            pw = min(pw, w)
            y0 = random.randint(0, h - ph) if h > ph else 0
            x0 = random.randint(0, w - pw) if w > pw else 0
            out[..., y0 : y0 + ph, x0 : x0 + pw] = fill

        return out


class GaussianBlur(SpectrogramTransform):
    """
    Suavizado gaussiano sobre el espectrograma.
    Simula degradación del sensor o condiciones de baja SNR.
    """

    def __init__(self, sigma_range: tuple[float, float] = (0.5, 1.5), p: float = 0.3):
        super().__init__(p)
        self.lo, self.hi = sigma_range

    def apply(self, spec: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        sigma = random.uniform(self.lo, self.hi)
        return gaussian_filter(spec, sigma=sigma).astype(spec.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPUESTO
# ─────────────────────────────────────────────────────────────────────────────


class AugmentationPipeline:
    """
    Encadena múltiples AudioTransform en secuencia.
    p_apply : probabilidad global de activar el pipeline completo.

    Uso:
        pipeline = AugmentationPipeline([
            AddGaussianNoise(snr_db_range=(20, 35)),
            PitchShift(n_steps_range=(-1.5, 1.5)),
        ], p_apply=0.85)
        y_aug = pipeline(y, sr=22050)
    """

    def __init__(
        self,
        transforms: list[AudioTransform],
        p_apply: float = 1.0,
    ):
        self.transforms = transforms
        self.p_apply = p_apply

    def __call__(self, y: np.ndarray, sr: int) -> np.ndarray:
        if random.random() > self.p_apply:
            return y
        for t in self.transforms:
            y = t(y, sr)
        return y

    def __repr__(self) -> str:
        lines = [f"AugmentationPipeline(p_apply={self.p_apply}):"]
        for t in self.transforms:
            lines.append(f"  {t}")
        return "\n".join(lines)


class SpectrogramAugmentationPipeline:
    """Encadena múltiples SpectrogramTransform."""

    def __init__(self, transforms: list[SpectrogramTransform], p_apply: float = 1.0):
        self.transforms = transforms
        self.p_apply = p_apply

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        if random.random() > self.p_apply:
            return spec
        for t in self.transforms:
            spec = t(spec)
        return spec


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINES PREDEFINIDOS
# ─────────────────────────────────────────────────────────────────────────────


def light_augment(sr: int = 22_050) -> AugmentationPipeline:
    """Augmentación ligera: solo pitch y volumen. Bajo costo computacional."""
    return AugmentationPipeline(
        [
            VolumeJitter(gain_db_range=(-4.0, 4.0), p=0.7),
            PitchShift(n_steps_range=(-1.0, 1.0), p=0.5),
            TimeShift(shift_max_s=0.3, p=0.4),
        ],
        p_apply=0.9,
    )


def standard_augment(
    noise_dir: Union[str, Path] | None = None,
) -> AugmentationPipeline:
    """
    Augmentación estándar recomendada para entrenamiento.
    Incluye ruido de fondo, stretch y SpecAugment.
    """
    return AugmentationPipeline(
        [
            VolumeJitter(gain_db_range=(-6.0, 6.0), p=0.6),
            AddBackgroundNoise(noise_dir=noise_dir, target_snr_db=18, snr_jitter_db=6, p=0.5),
            AddGaussianNoise(snr_db_range=(25.0, 45.0), p=0.3),
            TimeStretch(rate_range=(0.9, 1.1), p=0.4),
            PitchShift(n_steps_range=(-2.0, 2.0), p=0.4),
            TimeShift(shift_max_s=0.4, p=0.3),
        ],
        p_apply=0.95,
    )


def heavy_augment(
    noise_dir: Union[str, Path] | None = None,
) -> AugmentationPipeline:
    """
    Augmentación agresiva. Útil con datasets pequeños.
    Puede reducir accuracy si se aplica sin moderación.
    """
    return AugmentationPipeline(
        [
            VolumeJitter(gain_db_range=(-8.0, 8.0), p=0.8),
            AddBackgroundNoise(noise_dir=noise_dir, target_snr_db=12, snr_jitter_db=8, p=0.7),
            AddGaussianNoise(snr_db_range=(15.0, 30.0), p=0.5),
            TimeStretch(rate_range=(0.8, 1.2), p=0.5),
            PitchShift(n_steps_range=(-3.0, 3.0), p=0.5),
            RandomClip(clip_duration_s=3.0, p=0.3),
            TimeShift(shift_max_s=0.5, p=0.4),
        ],
        p_apply=1.0,
    )


def get_spectrogram_augment(intensity: str = "standard") -> SpectrogramAugmentationPipeline:
    """
    Pipeline de augmentación espectral según intensidad.

    intensity : "light" | "standard" | "heavy"
    """
    if intensity == "light":
        return SpectrogramAugmentationPipeline(
            [
                SpecAugmentFreq(max_mask_pct=0.08, n_masks=1, p=0.5),
                SpecAugmentTime(max_mask_pct=0.08, n_masks=1, p=0.5),
            ]
        )
    elif intensity == "heavy":
        return SpectrogramAugmentationPipeline(
            [
                SpecAugmentFreq(max_mask_pct=0.20, n_masks=3, p=0.7),
                SpecAugmentTime(max_mask_pct=0.20, n_masks=3, p=0.7),
                FrequencyShift(max_shift_bins=6, p=0.5),
                RandomErasing(n_patches=2, area_pct=0.06, p=0.4),
                GaussianBlur(sigma_range=(0.5, 1.5), p=0.3),
            ]
        )
    else:  # standard
        return SpectrogramAugmentationPipeline(
            [
                SpecAugmentFreq(max_mask_pct=0.15, n_masks=2, p=0.6),
                SpecAugmentTime(max_mask_pct=0.15, n_masks=2, p=0.6),
                FrequencyShift(max_shift_bins=4, p=0.4),
                RandomErasing(n_patches=1, area_pct=0.04, p=0.3),
            ]
        )


# ─────────────────────────────────────────────────────────────────────────────
# PRESETS POR GRUPO TAXONÓMICO
# ─────────────────────────────────────────────────────────────────────────────

_TAXON_PRESETS: dict = {
    "bats": {
        # Murciélagos: ultrasónico 20-200 kHz
        # TimeStretch no recomendado (pulsos de ecolocalización temporalmente críticos)
        # Pitch shift mínimo
        "waveform": lambda noise_dir=None: AugmentationPipeline(
            [
                VolumeJitter(gain_db_range=(-4.0, 4.0), p=0.7),
                AddBackgroundNoise(noise_dir=noise_dir, target_snr_db=20, snr_jitter_db=5, p=0.5),
                AddGaussianNoise(snr_db_range=(30.0, 50.0), p=0.4),
                PitchShift(n_steps_range=(-1.0, 1.0), p=0.3),
            ],
            p_apply=0.9,
        ),
        "spectrogram": lambda: get_spectrogram_augment("light"),
    },
    "frogs": {
        # Anuros: cantos en 200-8000 Hz, muy repetitivos
        # Pitch shift y time stretch moderados
        "waveform": lambda noise_dir=None: standard_augment(noise_dir),
        "spectrogram": lambda: get_spectrogram_augment("standard"),
    },
    "insects": {
        # Ortópteros / cicadas: 200-100000 Hz, cantos estridentes periódicos
        "waveform": lambda noise_dir=None: AugmentationPipeline(
            [
                VolumeJitter(gain_db_range=(-5.0, 5.0), p=0.7),
                AddBackgroundNoise(noise_dir=noise_dir, target_snr_db=15, snr_jitter_db=5, p=0.6),
                AddGaussianNoise(snr_db_range=(20.0, 35.0), p=0.4),
                TimeStretch(rate_range=(0.9, 1.1), p=0.3),
                PitchShift(n_steps_range=(-1.5, 1.5), p=0.4),
            ],
            p_apply=0.9,
        ),
        "spectrogram": lambda: get_spectrogram_augment("standard"),
    },
    "mammals": {
        # Mamíferos audibles (no murciélagos): 20-20000 Hz
        "waveform": lambda noise_dir=None: heavy_augment(noise_dir),
        "spectrogram": lambda: get_spectrogram_augment("standard"),
    },
    "birds": {
        # Aves audibles: cantos/llamadas con variación individual y ambiental moderada
        "waveform": lambda noise_dir=None: AugmentationPipeline(
            [
                VolumeJitter(gain_db_range=(-5.0, 5.0), p=0.7),
                AddBackgroundNoise(noise_dir=noise_dir, target_snr_db=18, snr_jitter_db=6, p=0.5),
                AddGaussianNoise(snr_db_range=(25.0, 45.0), p=0.3),
                TimeStretch(rate_range=(0.88, 1.12), p=0.4),
                PitchShift(n_steps_range=(-2.0, 2.0), p=0.5),
                TimeShift(shift_max_s=0.35, p=0.3),
            ],
            p_apply=0.95,
        ),
        "spectrogram": lambda: get_spectrogram_augment("standard"),
    },
    "reptiles": {
        # Reptiles: vocalización débil 100-5000 Hz
        # Augmentación agresiva para compensar variabilidad baja
        "waveform": lambda noise_dir=None: heavy_augment(noise_dir),
        "spectrogram": lambda: get_spectrogram_augment("heavy"),
    },
}


def get_preset(
    taxon: str,
    component: str = "waveform",
    noise_dir: Union[str, Path] | None = None,
) -> Union[AugmentationPipeline, SpectrogramAugmentationPipeline]:
    """
    Devuelve el pipeline de augmentación recomendado para un grupo taxonómico.

    Parameters
    ----------
    taxon     : "bats" | "birds" | "frogs" | "insects" | "mammals" | "reptiles"
    component : "waveform" | "spectrogram"
    noise_dir : directorio con archivos de ruido de fondo

    Returns
    -------
    AugmentationPipeline (waveform) o SpectrogramAugmentationPipeline
    """
    if taxon not in _TAXON_PRESETS:
        raise ValueError(f"Taxon '{taxon}' no reconocido. Opciones: {list(_TAXON_PRESETS)}")
    factory = _TAXON_PRESETS[taxon][component]
    if component == "waveform":
        return factory(noise_dir=noise_dir)
    return factory()


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRACIÓN CON PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────


class AugmentedSpectrogramDataset(torch.utils.data.Dataset):
    """
    Wrapper que aplica augmentación on-the-fly sobre un SpectrogramDataset.

    Parámetros
    ──────────
    base_dataset     : Dataset base que devuelve (spectrogram: Tensor, label: int)
    waveform_aug     : pipeline de augmentación temporal (opcional)
    spectrogram_aug  : pipeline de augmentación espectral (opcional)
    raw_audio_dir    : directorio con audios crudos (para waveform_aug)
    sr               : sample rate de los audios
    training         : si False, deshabilita augmentación
    """

    def __init__(
        self,
        base_dataset,
        waveform_aug: AugmentationPipeline | None = None,
        spectrogram_aug: SpectrogramAugmentationPipeline | None = None,
        sr: int = 22_050,
        training: bool = True,
    ):
        self.base = base_dataset
        self.wav_aug = waveform_aug
        self.spec_aug = spectrogram_aug
        self.sr = sr
        self.training = training

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]

        if isinstance(item, list | tuple):
            spec, label = item[0], item[1]
        else:
            spec, label = item["spectrogram"], item["label"]

        if self.training and self.spec_aug is not None:
            arr = spec.numpy() if isinstance(spec, torch.Tensor) else spec
            arr = self.spec_aug(arr)
            spec = torch.from_numpy(arr)

        return spec, label

    @property
    def classes(self):
        return self.base.classes if hasattr(self.base, "classes") else []
