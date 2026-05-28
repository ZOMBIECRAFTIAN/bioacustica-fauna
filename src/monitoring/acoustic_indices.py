"""
src/monitoring/acoustic_indices.py
─────────────────────────────────────────────────────────────────────────────
Índices acústicos de paisaje sonoro (Soundscape Ecology).

Índices implementados
──────────────────────
  ACI   Acoustic Complexity Index        Pieretti et al. (2011)
  ADI   Acoustic Diversity Index         Villanueva-Rivera et al. (2011)
  AEI   Acoustic Evenness Index          Villanueva-Rivera et al. (2011)
  BI    Bioacoustic Index                Boelman et al. (2007)
  NDSI  Normalized Difference SI         Kasten et al. (2012)
  Hf    Spectral Entropy                 Sueur et al. (2008)
  Ht    Temporal Entropy                 Sueur et al. (2008)
  H     Total Acoustic Entropy           Sueur et al. (2008)
  RMS   Root Mean Square (energía)       —
  ZCR   Zero Crossing Rate               —

Fundamento matemático
──────────────────────
  ACI:
    Para cada bin de frecuencia k y ventana temporal j:
      D_j = Σ_t |I_{k,t+1} - I_{k,t}|     (variación absoluta temporal)
      S_j = Σ_t  I_{k,t}                   (suma de intensidades)
      ACI_j = D_j / S_j
    ACI_total = Σ_j ACI_j

  ADI (Shannon sobre bandas de frecuencia):
    Dividir espectro en B bandas de igual ancho.
    p_b = proporción de energía en banda b sobre umbral threshold_db.
    ADI = -Σ_b (p_b · ln(p_b))             [Shannon diversity]

  AEI (Gini sobre bandas):
    AEI = Σ_b |p_b - 1/B|                  [desviación de uniformidad]

  BI:
    Área bajo la curva del espectro promedio en [f_low, f_high] (típico 2–8 kHz).
    BI = Σ_{f=f_low}^{f_high} (dB_f - dB_min)  [en Hz·dB]

  NDSI:
    A = energía en banda antrofónica  (1–2 kHz  por defecto)
    B = energía en banda biofónica    (2–11 kHz por defecto)
    NDSI = (B - A) / (B + A)  ∈ [-1, 1]
    NDSI > 0 → predomina biofonia   (ecosistema saludable)
    NDSI < 0 → predomina antrofonia (perturbación humana)

  Hf (Spectral Entropy):
    W(f) = espectro de potencia promediado temporalmente.
    W_n  = W(f) / Σ W(f)             [normalización a distribución de prob.]
    Hf   = -Σ W_n · log2(W_n) / log2(N)  ∈ [0, 1]

  Ht (Temporal Entropy):
    A(t) = amplitud RMS por ventana temporal.
    A_n  = A(t) / Σ A(t)
    Ht   = -Σ A_n · log2(A_n) / log2(N)  ∈ [0, 1]

  H = Hf × Ht  ∈ [0, 1]

Referencias
───────────
  Pieretti, N., Farina, A., Morri, D. (2011). A new methodology to infer the
    singing activity of an avian community: the Acoustic Complexity Index (ACI).
    Ecological Indicators, 11(3), 868–873.

  Sueur, J., Pavoine, S., Hamerlynck, O., Duvail, S. (2008). Rapid acoustic
    survey for biodiversity appraisal. PLOS ONE, 3(12), e4065.

  Villanueva-Rivera, L. J., Pijanowski, B. C., Doucette, J., Pekin, B. (2011).
    A primer of acoustic analysis for landscape ecologists.
    Landscape Ecology, 26(9), 1233–1246.

  Boelman, N. T., Asner, G. P., Hart, P. J., Martin, R. E. (2007).
    Multi-trophic invasion resistance in Hawaii: bioacoustics, field surveys,
    and airborne remote sensing. Ecological Applications, 17(8), 2137–2144.

  Kasten, E. P., Gage, S. H., Fox, J., Joo, W. (2012). The remote environmental
    assessment laboratory's acoustic library: an archive for studying soundscape
    ecology. Ecological Informatics, 12, 50–67.

Uso
───
    from src.monitoring.acoustic_indices import AcousticIndices, IndicesConfig

    cfg    = IndicesConfig(sample_rate=22050)
    ai     = AcousticIndices(cfg)
    result = ai.compute_all(y)          # waveform np.ndarray float32
    print(result.summary())

Autor: Ian
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndicesConfig:
    # Audio
    sample_rate:    int   = 22_050

    # STFT para índices
    n_fft:          int   = 1024          # resolución frecuencial
    hop_length:     int   = 512
    window:         str   = "hann"

    # ADI / AEI
    freq_step_hz:   float = 1000.0        # ancho de banda por banda (Hz)
    db_threshold:   float = -50.0         # umbral en dB para considerar actividad
    max_freq_hz:    float = 10_000.0      # frecuencia máxima de análisis

    # BI — Bioacoustic Index
    bi_freq_low_hz:  float = 2_000.0
    bi_freq_high_hz: float = 8_000.0

    # NDSI
    ndsi_anthro_low_hz:  float = 1_000.0  # banda antrofónica inferior
    ndsi_anthro_high_hz: float = 2_000.0
    ndsi_bio_low_hz:     float = 2_000.0  # banda biofónica inferior
    ndsi_bio_high_hz:    float = 11_000.0

    # Entropy
    entropy_n_frames: int = 512           # ventanas para Ht


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndicesResult:
    """Contenedor de todos los índices calculados para una grabación."""

    # Identidad
    duration_s: float = 0.0
    sample_rate: int  = 0

    # Índices principales
    aci:   float = 0.0   # Acoustic Complexity Index        [0, +∞)
    adi:   float = 0.0   # Acoustic Diversity Index         [0, ln(B)]
    aei:   float = 0.0   # Acoustic Evenness Index          [0, 1]
    bi:    float = 0.0   # Bioacoustic Index                [dB·Hz]
    ndsi:  float = 0.0   # Normalized Difference SI         [-1, 1]
    hf:    float = 0.0   # Spectral Entropy                 [0, 1]
    ht:    float = 0.0   # Temporal Entropy                 [0, 1]
    h:     float = 0.0   # Total Entropy (Hf × Ht)          [0, 1]

    # Estadísticas auxiliares
    rms:   float = 0.0   # energía RMS global
    zcr:   float = 0.0   # zero crossing rate global

    # Distribución de energía por banda (para ADI)
    band_proportions: Dict[str, float] = field(default_factory=dict)

    # Metadatos del espectrograma
    n_freq_bins:  int = 0
    n_time_frames: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        sep = "─" * 52
        lines = [
            sep,
            f"  ÍNDICES ACÚSTICOS  ({self.duration_s:.2f} s @ {self.sample_rate} Hz)",
            sep,
            f"  ACI   {self.aci:>10.4f}   Acoustic Complexity",
            f"  ADI   {self.adi:>10.4f}   Acoustic Diversity",
            f"  AEI   {self.aei:>10.4f}   Acoustic Evenness",
            f"  BI    {self.bi:>10.4f}   Bioacoustic Index  [dB·Hz]",
            f"  NDSI  {self.ndsi:>10.4f}   [-1=anthro | +1=bio]",
            f"  Hf    {self.hf:>10.4f}   Spectral Entropy",
            f"  Ht    {self.ht:>10.4f}   Temporal Entropy",
            f"  H     {self.h:>10.4f}   Total Entropy",
            sep,
            f"  RMS   {self.rms:>10.6f}   Energía RMS",
            f"  ZCR   {self.zcr:>10.6f}   Zero Crossing Rate",
            sep,
        ]
        return "\n".join(lines)

    def interpret(self) -> Dict[str, str]:
        """Interpretación ecológica cualitativa de cada índice."""
        interp: Dict[str, str] = {}

        # ACI
        if self.aci < 500:
            interp["aci"] = "Baja complejidad — ambiente poco activo o silencioso."
        elif self.aci < 1500:
            interp["aci"] = "Complejidad moderada — actividad bioacústica media."
        else:
            interp["aci"] = "Alta complejidad — rica actividad bioacústica."

        # ADI
        if self.adi < 0.5:
            interp["adi"] = "Baja diversidad espectral — pocas especies activas."
        elif self.adi < 1.5:
            interp["adi"] = "Diversidad espectral moderada."
        else:
            interp["adi"] = "Alta diversidad espectral — múltiples especies activas."

        # NDSI
        if self.ndsi > 0.5:
            interp["ndsi"] = "Ecosistema dominado por biofonia — entorno natural saludable."
        elif self.ndsi > 0.0:
            interp["ndsi"] = "Leve predominancia biofónica — perturbación antrópica baja."
        elif self.ndsi > -0.5:
            interp["ndsi"] = "Predominancia antrópica moderada — impacto humano detectable."
        else:
            interp["ndsi"] = "Alta perturbación antrópica — tráfico, maquinaria, urbanización."

        # H
        if self.h > 0.8:
            interp["h"] = "Entropía alta — ambiente acústicamente complejo y diverso."
        elif self.h > 0.5:
            interp["h"] = "Entropía media — mezcla de señales bióticas y abióticas."
        else:
            interp["h"] = "Entropía baja — dominancia de pocas frecuencias o silencio."

        return interp


# ─────────────────────────────────────────────────────────────────────────────
# CALCULADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class AcousticIndices:
    """
    Calcula todos los índices acústicos estándar de soundscape ecology.

    Parameters
    ----------
    cfg : IndicesConfig
    """

    def __init__(self, cfg: IndicesConfig = None):
        self.cfg = cfg or IndicesConfig()

    # ── STFT helper ───────────────────────────────────────────────────────────

    def _compute_spectrogram(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula espectrograma de potencia vía scipy.

        Returns
        -------
        freqs : (F,)     — frecuencias en Hz
        times : (T,)     — tiempos en s
        Sxx   : (F, T)   — potencia lineal (V²/Hz)
        """
        freqs, times, Sxx = scipy_spectrogram(
            y,
            fs          = self.cfg.sample_rate,
            nperseg     = self.cfg.n_fft,
            noverlap    = self.cfg.n_fft - self.cfg.hop_length,
            window      = self.cfg.window,
            scaling     = "spectrum",
        )
        Sxx = np.maximum(Sxx, 1e-12)   # evitar log(0)
        return freqs, times, Sxx

    def _power_to_db(self, Sxx: np.ndarray) -> np.ndarray:
        """Convierte potencia lineal a dB: 10·log10(Sxx)."""
        return 10.0 * np.log10(np.maximum(Sxx, 1e-12))

    # ── ACI ───────────────────────────────────────────────────────────────────

    def compute_aci(
        self, Sxx: np.ndarray, j_bin: int = 5
    ) -> float:
        """
        Acoustic Complexity Index — Pieretti et al. (2011).

        Divide el espectrograma en sub-ventanas de j_bin columnas.
        Para cada sub-ventana y cada bin de frecuencia:
          ACI_sub = Σ|I_{t+1} - I_t| / Σ I_t

        Parameters
        ----------
        Sxx   : (F, T) espectrograma de potencia lineal
        j_bin : número de frames por sub-ventana temporal

        Returns
        -------
        ACI total (escalar positivo)
        """
        F, T  = Sxx.shape
        aci   = 0.0
        n_sub = T // j_bin

        for j in range(n_sub):
            seg = Sxx[:, j * j_bin: (j + 1) * j_bin]   # (F, j_bin)
            D   = np.sum(np.abs(np.diff(seg, axis=1)), axis=1)   # (F,)
            S   = np.sum(seg, axis=1) + 1e-12                     # (F,)
            aci += float(np.sum(D / S))

        return aci

    # ── ADI / AEI ─────────────────────────────────────────────────────────────

    def _band_proportions(
        self, freqs: np.ndarray, Sxx_db: np.ndarray
    ) -> Tuple[List[float], List[str]]:
        """
        Calcula la proporción de energía activa (> db_threshold) en cada banda.

        Returns
        -------
        proportions : lista de p_b por banda
        labels      : etiquetas de banda (ej. "0-1kHz")
        """
        max_freq = min(self.cfg.max_freq_hz, freqs[-1])
        step     = self.cfg.freq_step_hz
        n_bands  = int(max_freq / step)

        props:  List[float] = []
        labels: List[str]   = []

        for b in range(n_bands):
            f_lo = b * step
            f_hi = (b + 1) * step
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if mask.sum() == 0:
                continue

            band_db   = Sxx_db[mask, :]            # (bins_en_banda, T)
            # Fracción de celdas por encima del umbral
            active    = float(np.mean(band_db > self.cfg.db_threshold))
            props.append(active)
            labels.append(f"{int(f_lo/1000)}-{int(f_hi/1000)}kHz")

        return props, labels

    def compute_adi(
        self, freqs: np.ndarray, Sxx_db: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Acoustic Diversity Index — Villanueva-Rivera et al. (2011).

        ADI = -Σ_b (p_b · ln(p_b))   [Shannon sobre bandas de frecuencia]

        Bandas con p_b = 0 contribuyen 0 al sumatorio (0·ln(0) ≡ 0).

        Returns
        -------
        adi   : escalar [0, ln(B)]
        bands : dict etiqueta → proporción
        """
        props, labels = self._band_proportions(freqs, Sxx_db)
        bands         = dict(zip(labels, props))

        adi = 0.0
        for p in props:
            if p > 0:
                adi -= p * np.log(p)
        return float(adi), bands

    def compute_aei(
        self, freqs: np.ndarray, Sxx_db: np.ndarray
    ) -> float:
        """
        Acoustic Evenness Index — Villanueva-Rivera et al. (2011).

        AEI = Σ_b |p_b - 1/B|   (desviación de la uniformidad perfecta)
        AEI = 0 → todas las bandas igualmente activas (máxima uniformidad).
        AEI → 1 → energía concentrada en pocas bandas.

        Returns
        -------
        aei : escalar [0, 1]
        """
        props, _ = self._band_proportions(freqs, Sxx_db)
        if not props:
            return 0.0
        uniform = 1.0 / len(props)
        aei     = float(np.sum(np.abs(np.array(props) - uniform)))
        # Normalizar al máximo teórico
        aei_max = (len(props) - 1) * uniform + (1.0 - uniform)
        return float(np.clip(aei / (aei_max + 1e-12), 0.0, 1.0))

    # ── BI ────────────────────────────────────────────────────────────────────

    def compute_bi(
        self, freqs: np.ndarray, Sxx_db: np.ndarray
    ) -> float:
        """
        Bioacoustic Index — Boelman et al. (2007).

        Área bajo la curva del espectro medio en [bi_freq_low, bi_freq_high].

        BI = Σ_{f ∈ [f_low, f_high]} (μ_dB(f) - dB_min)  × Δf

        donde μ_dB(f) es la media temporal de Sxx_db en la frecuencia f,
        y dB_min es el valor mínimo en esa banda.

        Unidades: dB·Hz

        Returns
        -------
        bi : escalar positivo [dB·Hz]
        """
        mask       = (freqs >= self.cfg.bi_freq_low_hz) & \
                     (freqs <= self.cfg.bi_freq_high_hz)
        if mask.sum() == 0:
            logger.warning("BI: sin bins en [%.0f, %.0f] Hz",
                           self.cfg.bi_freq_low_hz, self.cfg.bi_freq_high_hz)
            return 0.0

        band_db    = Sxx_db[mask, :]                    # (F_band, T)
        mean_db    = np.mean(band_db, axis=1)           # (F_band,)
        db_min     = float(mean_db.min())
        delta_f    = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        bi         = float(np.sum(mean_db - db_min) * delta_f)
        return bi

    # ── NDSI ─────────────────────────────────────────────────────────────────

    def compute_ndsi(
        self, freqs: np.ndarray, Sxx: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Normalized Difference Soundscape Index — Kasten et al. (2012).

        Compara energía biofónica vs. antrofónica.

        NDSI = (β - α) / (β + α)  ∈ [-1, 1]

        donde:
          α = energía total en banda antrofónica [anthro_low, anthro_high]
          β = energía total en banda biofónica   [bio_low,   bio_high  ]

        Returns
        -------
        ndsi   : escalar [-1, 1]
        anthro : energía antrofónica (α)
        bio    : energía biofónica   (β)
        """
        def _band_energy(f_lo: float, f_hi: float) -> float:
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if mask.sum() == 0:
                return 0.0
            return float(np.sum(Sxx[mask, :]))

        anthro = _band_energy(
            self.cfg.ndsi_anthro_low_hz,
            self.cfg.ndsi_anthro_high_hz,
        )
        bio = _band_energy(
            self.cfg.ndsi_bio_low_hz,
            min(self.cfg.ndsi_bio_high_hz, freqs[-1]),
        )

        denom = bio + anthro
        ndsi  = float((bio - anthro) / denom) if denom > 0 else 0.0
        return float(np.clip(ndsi, -1.0, 1.0)), anthro, bio

    # ── ENTROPÍA ──────────────────────────────────────────────────────────────

    def compute_hf(self, freqs: np.ndarray, Sxx: np.ndarray) -> float:
        """
        Spectral Entropy — Sueur et al. (2008).

        W(f) = Σ_t Sxx(f, t)            (espectro de potencia acumulado)
        W_n  = W(f) / Σ W(f)            (normalización → distribución prob.)
        Hf   = -Σ W_n · log2(W_n) / log2(N)  ∈ [0, 1]

        Hf = 1 → energía distribuida uniformemente en frecuencia (ruido blanco).
        Hf = 0 → energía concentrada en una sola frecuencia (tono puro).

        Returns
        -------
        hf : escalar [0, 1]
        """
        W   = np.sum(Sxx, axis=1)          # (F,)
        W   = W / (W.sum() + 1e-12)
        N   = len(W)
        # Evitar log(0)
        mask = W > 0
        hf  = -np.sum(W[mask] * np.log2(W[mask])) / np.log2(N)
        return float(np.clip(hf, 0.0, 1.0))

    def compute_ht(self, y: np.ndarray) -> float:
        """
        Temporal Entropy — Sueur et al. (2008).

        Divide la señal en N_frames ventanas de igual duración.
        A(t) = RMS de cada ventana.
        A_n  = A(t) / Σ A(t)
        Ht   = -Σ A_n · log2(A_n) / log2(N)  ∈ [0, 1]

        Ht = 1 → energía distribuida uniformemente en el tiempo (ruido estacionario).
        Ht = 0 → toda la energía en un instante (impulso).

        Returns
        -------
        ht : escalar [0, 1]
        """
        N      = self.cfg.entropy_n_frames
        frames = np.array_split(y, N)
        rms_t  = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
        total  = rms_t.sum() + 1e-12
        A_n    = rms_t / total
        mask   = A_n > 0
        ht     = -np.sum(A_n[mask] * np.log2(A_n[mask])) / np.log2(N)
        return float(np.clip(ht, 0.0, 1.0))

    # ── RMS / ZCR ─────────────────────────────────────────────────────────────

    @staticmethod
    def compute_rms(y: np.ndarray) -> float:
        """RMS global de la señal."""
        return float(np.sqrt(np.mean(y ** 2)))

    @staticmethod
    def compute_zcr(y: np.ndarray) -> float:
        """Zero Crossing Rate normalizado [0, 1]."""
        return float(np.mean(np.abs(np.diff(np.sign(y)))) / 2.0)

    # ── Método principal ──────────────────────────────────────────────────────

    def compute_all(self, y: np.ndarray) -> IndicesResult:
        """
        Calcula todos los índices sobre una señal de audio.

        Parameters
        ----------
        y : np.ndarray float32, señal mono normalizada [-1, 1]

        Returns
        -------
        IndicesResult con todos los índices calculados.
        """
        sr = self.cfg.sample_rate

        if y.ndim > 1:
            y = y.mean(axis=0)
        y = y.astype(np.float32)

        # Espectrograma base
        freqs, times, Sxx = self._compute_spectrogram(y)
        Sxx_db            = self._power_to_db(Sxx)

        # Calcular índices
        aci              = self.compute_aci(Sxx)
        adi, band_props  = self.compute_adi(freqs, Sxx_db)
        aei              = self.compute_aei(freqs, Sxx_db)
        bi               = self.compute_bi(freqs, Sxx_db)
        ndsi, anthro, bio = self.compute_ndsi(freqs, Sxx)
        hf               = self.compute_hf(freqs, Sxx)
        ht               = self.compute_ht(y)
        h                = hf * ht
        rms              = self.compute_rms(y)
        zcr              = self.compute_zcr(y)

        result = IndicesResult(
            duration_s      = len(y) / sr,
            sample_rate     = sr,
            aci             = aci,
            adi             = adi,
            aei             = aei,
            bi              = bi,
            ndsi            = ndsi,
            hf              = hf,
            ht              = ht,
            h               = h,
            rms             = rms,
            zcr             = zcr,
            band_proportions= band_props,
            n_freq_bins     = len(freqs),
            n_time_frames   = len(times),
        )

        logger.debug(
            "Índices: ACI=%.2f ADI=%.3f NDSI=%.3f H=%.3f",
            aci, adi, ndsi, h,
        )
        return result

    # ── Análisis por ventanas temporales ─────────────────────────────────────

    def compute_windowed(
        self,
        y:              np.ndarray,
        window_s:       float = 60.0,
        hop_s:          float = 60.0,
    ) -> List[Dict]:
        """
        Calcula índices en ventanas temporales sucesivas.
        Útil para analizar variación circadiana en grabaciones largas.

        Parameters
        ----------
        y        : señal completa (puede ser horas de grabación)
        window_s : duración de cada ventana en segundos (def. 60 s)
        hop_s    : salto entre ventanas en segundos

        Returns
        -------
        Lista de dicts con: t_start_s, t_end_s + todos los índices.
        """
        sr         = self.cfg.sample_rate
        win_samples = int(window_s * sr)
        hop_samples = int(hop_s * sr)
        results:   List[Dict] = []

        start = 0
        while start < len(y):
            end   = min(start + win_samples, len(y))
            chunk = y[start:end]

            # Rellenar si el último chunk es corto
            if len(chunk) < win_samples:
                chunk = np.pad(chunk, (0, win_samples - len(chunk)))

            idx           = self.compute_all(chunk)
            row           = idx.to_dict()
            row["t_start_s"] = start / sr
            row["t_end_s"]   = end   / sr
            results.append(row)
            start += hop_samples

        logger.info(
            "Análisis ventaneado: %d ventanas de %.0fs",
            len(results), window_s,
        )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE CONVENIENCIA
# ─────────────────────────────────────────────────────────────────────────────

def compute_indices(
    y:           np.ndarray,
    sample_rate: int = 22_050,
    cfg:         Optional[IndicesConfig] = None,
) -> IndicesResult:
    """
    Función de conveniencia para calcular todos los índices en una línea.

    Ejemplo
    -------
    >>> import soundfile as sf
    >>> y, sr = sf.read("grabacion.wav")
    >>> result = compute_indices(y, sr)
    >>> print(result.summary())
    """
    if cfg is None:
        cfg = IndicesConfig(sample_rate=sample_rate)
    return AcousticIndices(cfg).compute_all(y)


def indices_from_file(
    filepath:    str,
    cfg:         Optional[IndicesConfig] = None,
    windowed:    bool = False,
    window_s:    float = 60.0,
):
    """
    Carga un archivo de audio y calcula sus índices acústicos.

    Parameters
    ----------
    filepath : path al archivo .wav / .flac / .mp3
    cfg      : IndicesConfig (None → usa defaults)
    windowed : si True, calcula índices por ventanas temporales
    window_s : duración de ventana en segundos (solo si windowed=True)

    Returns
    -------
    IndicesResult (windowed=False) o List[Dict] (windowed=True)
    """
    import soundfile as sf

    y, sr = sf.read(filepath, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    if cfg is None:
        cfg = IndicesConfig(sample_rate=sr)
    else:
        cfg = IndicesConfig(**{**asdict(cfg), "sample_rate": sr})

    ai = AcousticIndices(cfg)

    if windowed:
        return ai.compute_windowed(y, window_s=window_s)
    return ai.compute_all(y)
