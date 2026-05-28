"""
audio_processing/preprocessor.py
─────────────────────────────────────────────────────────────────────────────
Pipeline de preprocesamiento de señales de audio para bioacústica.
Soporta: mamíferos (ultrasonido quirópteros), anfibios, insectos, reptiles.

Dependencias:
    pip install librosa soundfile numpy scipy noisereduce

Autor: Ian
Versión: 1.0.0
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN DE PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioConfig:
    """
    Parámetros de configuración para el pipeline de preprocesamiento.
    Ajusta según el grupo taxonómico objetivo.
    """
    # ── Tasa de muestreo ─────────────────────────────────────────────────────
    sample_rate: int = 44_100          # Hz — usar 192_000 para quirópteros

    # ── Segmentación ─────────────────────────────────────────────────────────
    segment_duration: float = 3.0      # segundos por segmento de análisis
    hop_duration: float = 1.5          # solapamiento entre segmentos (50% overlap)

    # ── Filtrado frecuencial ──────────────────────────────────────────────────
    apply_bandpass: bool = True
    freq_low: float = 200.0            # Hz — límite inferior del filtro pasa-banda
    freq_high: float = 20_000.0        # Hz — límite superior (20kHz para audible)
    # Para quirópteros usar: freq_low=10_000, freq_high=96_000

    # ── Reducción de ruido ────────────────────────────────────────────────────
    apply_noise_reduction: bool = True
    noise_prop_decrease: float = 0.85  # agresividad de reducción (0–1)

    # ── Normalización ─────────────────────────────────────────────────────────
    normalize: bool = True
    target_lufs: float = -23.0         # LUFS objetivo (EBU R128 para campo)

    # ── Espectrograma Mel ─────────────────────────────────────────────────────
    n_fft: int = 2048                  # tamaño de la FFT
    hop_length: int = 512              # salto en muestras entre frames
    n_mels: int = 128                  # bandas Mel (64 para aves, 128 para multi-grupo)
    fmin: float = 50.0                 # frecuencia mínima del banco Mel
    fmax: Optional[float] = None       # None = sr/2 (Nyquist)
    power: float = 2.0                 # potencia del espectrograma (2=energía, 1=amplitud)

    # ── MFCC ─────────────────────────────────────────────────────────────────
    n_mfcc: int = 40                   # número de coeficientes MFCC

    # ── Detección de actividad vocal (VAD) ────────────────────────────────────
    vad_energy_threshold: float = 0.02  # umbral de energía RMS relativa
    vad_min_duration: float = 0.05      # duración mínima de evento (segundos)


# Presets por grupo taxonómico
PRESETS: dict[str, AudioConfig] = {
    "bats": AudioConfig(
        sample_rate=192_000,
        freq_low=10_000, freq_high=96_000,
        n_mels=64, fmin=10_000, fmax=96_000,
        segment_duration=0.5, hop_duration=0.25,
        n_fft=1024, hop_length=128,
    ),
    "frogs": AudioConfig(
        sample_rate=22_050,
        freq_low=100, freq_high=10_000,
        n_mels=128, fmin=100, fmax=10_000,
        segment_duration=3.0, hop_duration=1.0,
    ),
    "insects": AudioConfig(
        sample_rate=44_100,
        freq_low=200, freq_high=20_000,
        n_mels=128, fmin=200,
        segment_duration=2.0, hop_duration=1.0,
    ),
    "mammals": AudioConfig(
        sample_rate=44_100,
        freq_low=50, freq_high=18_000,
        n_mels=128, fmin=50,
        segment_duration=3.0, hop_duration=1.5,
    ),
    "reptiles": AudioConfig(
        sample_rate=22_050,
        freq_low=50, freq_high=8_000,
        n_mels=64, fmin=50, fmax=8_000,
        segment_duration=5.0, hop_duration=2.5,
    ),
    "default": AudioConfig(),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLASE PRINCIPAL DE PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

class AudioPreprocessor:
    """
    Pipeline de preprocesamiento de audio para bioacústica multiespecie.

    Flujo interno:
        cargar → resamplear → filtro pasa-banda → reducción de ruido
            → normalización → segmentar → extraer features

    Ejemplo de uso:
        config = PRESETS["frogs"]
        proc   = AudioPreprocessor(config)
        audio, sr = proc.load("grabacion.wav")
        segments  = proc.segment(audio)
        features  = [proc.extract_features(s) for s in segments]
    """

    def __init__(self, config: AudioConfig = AudioConfig()):
        self.cfg = config

    # ── Carga y remuestreo ────────────────────────────────────────────────────

    def load(self, filepath: str | Path, mono: bool = True) -> Tuple[np.ndarray, int]:
        """
        Carga un archivo de audio y lo remuestrea a config.sample_rate.

        Returns:
            (audio_array, sample_rate)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

        # librosa convierte a float32 en [-1, 1] y remuestrea automáticamente
        y, sr = librosa.load(
            filepath,
            sr=self.cfg.sample_rate,
            mono=mono,
            dtype=np.float32,
        )
        return y, sr

    # ── Filtrado pasa-banda ───────────────────────────────────────────────────

    def bandpass_filter(self, y: np.ndarray, order: int = 4) -> np.ndarray:
        """
        Filtro Butterworth pasa-banda para eliminar frecuencias fuera
        del rango de interés biológico.

        Complejidad: O(n) — procesamiento lineal en el tiempo.
        """
        nyq = self.cfg.sample_rate / 2.0
        low  = self.cfg.freq_low  / nyq
        high = self.cfg.freq_high / nyq

        low  = np.clip(low,  1e-6, 0.9999)
        high = np.clip(high, low + 1e-6, 0.9999)

        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, y).astype(np.float32)

    # ── Reducción de ruido ────────────────────────────────────────────────────

    def reduce_noise(self, y: np.ndarray) -> np.ndarray:
        """
        Reducción espectral de ruido estacionario usando noisereduce.
        Estima el perfil de ruido desde los primeros 0.5 segundos o
        los primeros n_frames si el audio es corto.
        """
        n_noise = min(int(0.5 * self.cfg.sample_rate), len(y) // 4)
        noise_clip = y[:n_noise] if n_noise > 0 else y
        return nr.reduce_noise(
            y=y,
            y_noise=noise_clip,
            sr=self.cfg.sample_rate,
            prop_decrease=self.cfg.noise_prop_decrease,
            stationary=True,
        ).astype(np.float32)

    # ── Normalización de amplitud ─────────────────────────────────────────────

    def normalize(self, y: np.ndarray, method: str = "peak") -> np.ndarray:
        """
        Normaliza la amplitud de la señal.

        Args:
            method: 'peak'  → divide por el valor absoluto máximo
                    'rms'   → normaliza a RMS = 0.1
                    'lufs'  → aproximación a loudness EBU R128
        """
        if method == "peak":
            peak = np.max(np.abs(y))
            if peak > 1e-8:
                y = y / peak
        elif method == "rms":
            rms = np.sqrt(np.mean(y ** 2))
            if rms > 1e-8:
                y = y * (0.1 / rms)
        elif method == "lufs":
            rms = np.sqrt(np.mean(y ** 2))
            if rms > 1e-8:
                target_rms = 10 ** (self.cfg.target_lufs / 20.0)
                y = y * (target_rms / rms)
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    # ── Detección de actividad vocal (VAD) ────────────────────────────────────

    def detect_events(self, y: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detecta eventos de actividad sonora mediante umbral de energía RMS.

        Returns:
            Lista de tuplas (t_start_s, t_end_s) de cada evento detectado.
        """
        frame_length = self.cfg.n_fft
        hop          = self.cfg.hop_length
        sr           = self.cfg.sample_rate
        min_frames   = int(self.cfg.vad_min_duration * sr / hop)

        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop
        )[0]

        # Umbral adaptativo: proporción relativa a la energía máxima
        threshold = self.cfg.vad_energy_threshold * rms.max()
        active = rms > threshold

        # Detectar bordes activo/inactivo
        events: List[Tuple[float, float]] = []
        in_event = False
        start_frame = 0

        for i, is_active in enumerate(active):
            if is_active and not in_event:
                in_event = True
                start_frame = i
            elif not is_active and in_event:
                in_event = False
                if (i - start_frame) >= min_frames:
                    t_start = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop)
                    t_end   = librosa.frames_to_time(i, sr=sr, hop_length=hop)
                    events.append((float(t_start), float(t_end)))

        if in_event:
            t_start = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop)
            t_end   = len(y) / sr
            events.append((float(t_start), float(t_end)))

        return events

    # ── Segmentación uniforme ─────────────────────────────────────────────────

    def segment(self, y: np.ndarray) -> List[np.ndarray]:
        """
        Divide el audio en segmentos de duración fija con solapamiento.

        Returns:
            Lista de arrays NumPy, cada uno de longitud segment_length muestras.
        """
        sr              = self.cfg.sample_rate
        segment_length  = int(self.cfg.segment_duration * sr)
        hop_length      = int(self.cfg.hop_duration * sr)

        segments: List[np.ndarray] = []
        start = 0

        while start + segment_length <= len(y):
            seg = y[start : start + segment_length].copy()
            segments.append(seg)
            start += hop_length

        # Último segmento incompleto: padding con ceros
        if start < len(y):
            seg = np.zeros(segment_length, dtype=np.float32)
            tail = y[start:]
            seg[:len(tail)] = tail
            segments.append(seg)

        return segments

    # ─────────────────────────────────────────────────────────────────────────
    # 3. EXTRACCIÓN DE CARACTERÍSTICAS
    # ─────────────────────────────────────────────────────────────────────────

    def mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Computa el espectrograma Mel en escala logarítmica (dB).

        Returns:
            Array shape (n_mels, T) en dB.
        """
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            power=self.cfg.power,
        )
        mel      = np.maximum(mel, 1e-10)                       # floor: evita log(0)
        ref_val  = float(np.max(mel))
        # Si el frame es silencio (todo ≤ floor), ref=np.max daría 0 dB por normalización;
        # usar ref=1.0 produce ≈ −100 dB — representación física correcta.
        ref      = ref_val if ref_val > 1e-9 else 1.0
        return librosa.power_to_db(mel, ref=ref).astype(np.float32)

    def mfcc(self, y: np.ndarray, include_delta: bool = True) -> np.ndarray:
        """
        Extrae MFCCs + delta + delta-delta.

        Returns:
            Array shape (n_mfcc * 3, T) si include_delta=True,
                        (n_mfcc,     T) si include_delta=False.
        """
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.cfg.sample_rate,
            n_mfcc=self.cfg.n_mfcc,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
        )
        if not include_delta:
            return mfcc.astype(np.float32)

        delta   = librosa.feature.delta(mfcc, order=1)
        delta2  = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, delta, delta2]).astype(np.float32)

    def spectral_features(self, y: np.ndarray) -> dict:
        """
        Extrae características espectrales estadísticas por frame.

        Returns:
            Diccionario con arrays 1D (media + std por feature).
        """
        sr  = self.cfg.sample_rate
        hop = self.cfg.hop_length
        n_fft = self.cfg.n_fft

        def stats(arr: np.ndarray) -> Tuple[float, float]:
            return float(np.mean(arr)), float(np.std(arr))

        zcr      = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        sc       = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
        sb       = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
        sr_feat  = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
        sf_feat  = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)[0]
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
        rms      = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]

        return {
            "zcr_mean":           stats(zcr)[0],        "zcr_std":           stats(zcr)[1],
            "spectral_centroid_mean": stats(sc)[0],     "spectral_centroid_std": stats(sc)[1],
            "spectral_bandwidth_mean": stats(sb)[0],    "spectral_bandwidth_std": stats(sb)[1],
            "spectral_rolloff_mean": stats(sr_feat)[0], "spectral_rolloff_std": stats(sr_feat)[1],
            "spectral_flatness_mean": stats(sf_feat)[0],"spectral_flatness_std": stats(sf_feat)[1],
            "chroma_mean":        float(chroma.mean()),  "chroma_std":        float(chroma.std()),
            "spectral_contrast_mean": float(contrast.mean()),
            "rms_mean":           stats(rms)[0],         "rms_std":           stats(rms)[1],
        }

    def extract_features(self, y: np.ndarray) -> dict:
        """
        Extrae el conjunto completo de características de un segmento.

        Returns:
            {
              'mel_spectrogram': np.ndarray (n_mels, T),
              'mfcc':            np.ndarray (n_mfcc*3, T),
              'spectral':        dict de scalars,
            }
        """
        return {
            "mel_spectrogram": self.mel_spectrogram(y),
            "mfcc":            self.mfcc(y, include_delta=True),
            "spectral":        self.spectral_features(y),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 4. PIPELINE COMPLETO
    # ─────────────────────────────────────────────────────────────────────────

    def process(
        self,
        filepath: str | Path,
        return_segments: bool = False,
    ) -> dict:
        """
        Ejecuta el pipeline completo sobre un archivo de audio.

        Args:
            filepath:        Ruta al archivo WAV/MP3/FLAC/OGG.
            return_segments: Si True, incluye los arrays de audio crudos
                             por segmento en el output.

        Returns:
            {
              'file':     nombre de archivo,
              'sr':       tasa de muestreo,
              'duration': duración total en segundos,
              'events':   lista de (t_start, t_end) de eventos VAD,
              'segments': [{'features': {...}, 'audio': np.ndarray}]
            }
        """
        y, sr = self.load(filepath)

        # Preprocesamiento secuencial
        if self.cfg.apply_bandpass:
            y = self.bandpass_filter(y)

        if self.cfg.apply_noise_reduction:
            y = self.reduce_noise(y)

        if self.cfg.normalize:
            y = self.normalize(y, method="peak")

        events   = self.detect_events(y)
        segments = self.segment(y)

        result = {
            "file":     Path(filepath).name,
            "sr":       sr,
            "duration": len(y) / sr,
            "events":   events,
            "n_events": len(events),
            "segments": [],
        }

        for i, seg in enumerate(segments):
            t_start = i * self.cfg.hop_duration
            entry = {
                "index":   i,
                "t_start": t_start,
                "t_end":   t_start + self.cfg.segment_duration,
                "features": self.extract_features(seg),
            }
            if return_segments:
                entry["audio"] = seg
            result["segments"].append(entry)

        return result

    # ── Guardar segmentos procesados ──────────────────────────────────────────

    def save_segment(
        self, audio: np.ndarray, output_path: str | Path
    ) -> None:
        """Guarda un segmento de audio como WAV 16-bit."""
        sf.write(
            str(output_path),
            audio,
            self.cfg.sample_rate,
            subtype="PCM_16",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. UTILIDADES DE VALIDACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def validate_audio_file(filepath: str | Path) -> dict:
    """
    Valida un archivo de audio y retorna sus metadatos.

    Returns:
        {'valid': bool, 'sr': int, 'duration': float, 'channels': int,
         'format': str, 'error': str | None}
    """
    try:
        info = sf.info(str(filepath))
        return {
            "valid":    True,
            "sr":       info.samplerate,
            "duration": info.duration,
            "channels": info.channels,
            "format":   info.format,
            "error":    None,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def batch_validate(directory: str | Path, extensions: tuple = (".wav", ".mp3", ".flac", ".ogg")) -> list:
    """Valida todos los archivos de audio en un directorio."""
    directory = Path(directory)
    results = []
    for ext in extensions:
        for fp in sorted(directory.rglob(f"*{ext}")):
            info = validate_audio_file(fp)
            info["file"] = str(fp)
            results.append(info)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. DEMO / TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Uso: python preprocessor.py <archivo.wav> [preset]")
        print("Presets disponibles:", list(PRESETS.keys()))
        sys.exit(0)

    filepath = sys.argv[1]
    preset   = sys.argv[2] if len(sys.argv) > 2 else "default"

    cfg  = PRESETS.get(preset, AudioConfig())
    proc = AudioPreprocessor(cfg)

    print(f"[INFO] Procesando: {filepath} con preset='{preset}'")
    result = proc.process(filepath, return_segments=False)

    # Serialización básica (sin arrays NumPy)
    summary = {
        "file":     result["file"],
        "sr":       result["sr"],
        "duration": round(result["duration"], 3),
        "n_events": result["n_events"],
        "events":   [(round(s, 3), round(e, 3)) for s, e in result["events"]],
        "n_segments": len(result["segments"]),
        "first_segment_features_keys": list(result["segments"][0]["features"].keys()) if result["segments"] else [],
        "spectral_sample": result["segments"][0]["features"]["spectral"] if result["segments"] else {},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
