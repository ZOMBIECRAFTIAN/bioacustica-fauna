"""
feature_extraction/batch_extractor.py
─────────────────────────────────────────────────────────────────────────────
Pipeline de extracción masiva de espectrogramas Mel y MFCC desde
grabaciones de audio crudas hacia arrays NumPy (.npy).

Flujo:
    data/raw/{clase}/{archivo}.wav
        → preprocesamiento
        → segmentación
        → espectrograma Mel (128×T)
        → data/spectrograms/{clase}/{id}.npy

Optimizaciones:
  - Multiproceso con ProcessPoolExecutor
  - Caché de archivos ya procesados (skip si .npy existe)
  - Reporte de cobertura por clase al finalizar
  - Soporte para múltiples grupos taxonómicos con presets

Uso:
    python -m src.feature_extraction.batch_extractor
    python -m src.feature_extraction.batch_extractor --preset frogs --workers 4
    python -m src.feature_extraction.batch_extractor --validate

Autor: Ian
Versión: 1.0.0
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExtractionConfig:
    # Rutas
    input_dir: str = "data/raw"
    output_dir: str = "data/spectrograms"
    mfcc_dir: str = "data/features/mfcc"

    # Audio
    sample_rate: int = 22_050
    segment_duration: float = 3.0
    hop_duration: float = 1.5

    # Grupo taxonómico (preset)
    preset: str = "default"

    # Espectrograma Mel
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float | None = None  # None = Nyquist

    # MFCC
    n_mfcc: int = 40
    save_mfcc: bool = True

    # Filtrado de calidad
    min_duration_s: float = 1.0
    max_duration_s: float = 60.0
    apply_bandpass: bool = True
    apply_nr: bool = True

    # Proceso
    overwrite: bool = False  # sobreescribir .npy existentes
    n_workers: int = 0  # 0 = auto (cpu_count - 1)
    max_per_class: int | None = None  # límite por clase (None = sin límite)

    # Formato de salida
    normalize_output: bool = True  # escalar mel a [-1, 1] antes de guardar
    dtype: str = "float32"


# Presets alineados con audio/preprocessor.py
PRESET_OVERRIDES: dict[str, dict] = {
    "bats": {
        "sample_rate": 192_000,
        "n_mels": 64,
        "fmin": 10_000,
        "fmax": 96_000,
        "segment_duration": 0.5,
        "hop_duration": 0.25,
        "n_fft": 1024,
        "hop_length": 128,
    },
    "frogs": {
        "sample_rate": 22_050,
        "n_mels": 128,
        "fmin": 100,
        "fmax": 10_000,
        "segment_duration": 3.0,
    },
    "insects": {
        "sample_rate": 44_100,
        "n_mels": 128,
        "fmin": 200,
    },
    "mammals": {
        "sample_rate": 44_100,
        "n_mels": 128,
        "fmin": 50,
    },
    "multitaxon": {
        "sample_rate": 44_100,
        "n_mels": 128,
        "fmin": 50,
        "fmax": None,
        "segment_duration": 3.0,
        "hop_duration": 1.5,
        "max_duration_s": 180.0,
    },
    "birds": {
        "sample_rate": 44_100,
        "n_mels": 128,
        "fmin": 200,
        "fmax": 12_000,
        "segment_duration": 3.0,
        "hop_duration": 1.5,
        "max_duration_s": 180.0,
    },
    "reptiles": {
        "sample_rate": 22_050,
        "n_mels": 64,
        "fmin": 50,
        "fmax": 8_000,
        "segment_duration": 5.0,
    },
}


def apply_preset(cfg: ExtractionConfig, preset: str) -> ExtractionConfig:
    """Aplica los overrides del preset sobre la configuración base."""
    if preset in PRESET_OVERRIDES:
        overrides = PRESET_OVERRIDES[preset]
        for k, v in overrides.items():
            setattr(cfg, k, v)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 2. WORKER: PROCESAMIENTO DE UN ARCHIVO (ejecuta en proceso separado)
# ─────────────────────────────────────────────────────────────────────────────


def _process_single_file(args: tuple) -> dict:
    """
    Worker: extrae espectrogramas Mel de un archivo de audio.
    Diseñado para ser serializable por pickle (multiprocessing).

    Returns:
        {'file': str, 'status': ok|skip|error,
         'n_segments': int, 'error': str|None}
    """
    filepath, class_label, out_dir, mfcc_dir, cfg_dict, idx = args
    cfg = ExtractionConfig(**cfg_dict)

    # Import dentro del worker para compatibilidad con multiproceso
    import librosa
    import soundfile as sf

    filepath = Path(filepath)
    out_dir = Path(out_dir) / class_label
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.save_mfcc:
        mfcc_out = Path(mfcc_dir) / class_label
        mfcc_out.mkdir(parents=True, exist_ok=True)

    # ── Verificar duración sin cargar el archivo completo ─────────────────────
    try:
        info = sf.info(str(filepath))
        if not (cfg.min_duration_s <= info.duration <= cfg.max_duration_s):
            return {
                "file": filepath.name,
                "class": class_label,
                "status": "skip",
                "n_segments": 0,
                "error": f"duration={info.duration:.1f}s",
            }
    except Exception as e:
        return {
            "file": filepath.name,
            "class": class_label,
            "status": "error",
            "n_segments": 0,
            "error": str(e),
        }

    # ── Verificar si ya está procesado ────────────────────────────────────────
    file_hash = hashlib.sha256(filepath.name.encode()).hexdigest()[:8]
    pattern = f"{file_hash}_s*.npy"
    existing = list(out_dir.glob(pattern))
    if existing and not cfg.overwrite:
        return {
            "file": filepath.name,
            "class": class_label,
            "status": "skip",
            "n_segments": len(existing),
            "error": None,
        }

    # ── Carga y preprocesamiento ──────────────────────────────────────────────
    try:
        y, sr = librosa.load(
            str(filepath),
            sr=cfg.sample_rate,
            mono=True,
            dtype=np.float32,
        )
    except Exception as e:
        return {
            "file": filepath.name,
            "class": class_label,
            "status": "error",
            "n_segments": 0,
            "error": f"load: {e}",
        }

    # Filtro pasa-banda
    if cfg.apply_bandpass:
        try:
            from scipy.signal import butter, filtfilt

            nyq = cfg.sample_rate / 2.0
            low = max(1e-6, cfg.fmin / nyq)
            high = min(0.9999, (cfg.fmax or nyq * 0.95) / nyq)
            if low < high:
                b, a = butter(4, [low, high], btype="band")
                y = filtfilt(b, a, y).astype(np.float32)
        except Exception:
            pass

    # Reducción de ruido
    if cfg.apply_nr:
        try:
            import noisereduce as nr

            n_noise = min(int(0.5 * cfg.sample_rate), len(y) // 4)
            if n_noise > 0:
                y = nr.reduce_noise(
                    y=y,
                    y_noise=y[:n_noise],
                    sr=cfg.sample_rate,
                    stationary=True,
                    prop_decrease=0.75,
                ).astype(np.float32)
        except ImportError:
            pass
        except Exception:
            pass

    # Normalización pico
    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y = y / peak

    # ── Segmentación ─────────────────────────────────────────────────────────
    seg_len = int(cfg.segment_duration * cfg.sample_rate)
    hop_len = int(cfg.hop_duration * cfg.sample_rate)
    segments: list[np.ndarray] = []

    start = 0
    while start + seg_len <= len(y):
        segments.append(y[start : start + seg_len])
        start += hop_len

    if start < len(y):
        pad = np.zeros(seg_len, dtype=np.float32)
        tail = y[start:]
        pad[: len(tail)] = tail
        segments.append(pad)

    if not segments:
        return {
            "file": filepath.name,
            "class": class_label,
            "status": "skip",
            "n_segments": 0,
            "error": "audio demasiado corto",
        }

    # ── Extracción de features por segmento ──────────────────────────────────
    saved = 0
    for seg_idx, seg in enumerate(segments):
        out_path = out_dir / f"{file_hash}_s{seg_idx:04d}.npy"
        if out_path.exists() and not cfg.overwrite:
            saved += 1
            continue

        # Mel spectrogram (n_mels, T)
        mel = librosa.feature.melspectrogram(
            y=seg,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        # Normalizar a [-1, 1]
        if cfg.normalize_output:
            mn, mx = mel_db.min(), mel_db.max()
            if mx - mn > 1e-8:
                mel_db = 2 * (mel_db - mn) / (mx - mn) - 1

        np.save(str(out_path), mel_db)

        # MFCC (40 × T) — guardado separado
        if cfg.save_mfcc:
            mfcc_path = Path(mfcc_dir) / class_label / f"{file_hash}_s{seg_idx:04d}.npy"
            if not mfcc_path.exists() or cfg.overwrite:
                mfcc = librosa.feature.mfcc(
                    y=seg,
                    sr=cfg.sample_rate,
                    n_mfcc=cfg.n_mfcc,
                    n_fft=cfg.n_fft,
                    hop_length=cfg.hop_length,
                    fmin=cfg.fmin,
                    fmax=cfg.fmax,
                )
                delta = librosa.feature.delta(mfcc, order=1)
                delta2 = librosa.feature.delta(mfcc, order=2)
                mfcc_full = np.vstack([mfcc, delta, delta2]).astype(np.float32)
                np.save(str(mfcc_path), mfcc_full)

        saved += 1

    return {
        "file": filepath.name,
        "class": class_label,
        "status": "ok",
        "n_segments": saved,
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. ORQUESTADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────


class BatchExtractor:
    """
    Extrae espectrogramas Mel de todo el dataset de audio en batch.

    Ejemplo:
        cfg = ExtractionConfig(input_dir="data/raw", output_dir="data/spectrograms")
        ext = BatchExtractor(cfg)
        report = ext.run()
    """

    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"}

    def __init__(self, cfg: ExtractionConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        if cfg.save_mfcc:
            Path(cfg.mfcc_dir).mkdir(parents=True, exist_ok=True)

    def _collect_files(self) -> list[tuple[str, str]]:
        """
        Recorre input_dir y recopila (filepath, class_label).
        Respeta max_per_class si está configurado.
        """
        input_dir = Path(self.cfg.input_dir)
        files: list[tuple[str, str]] = []

        for class_dir in sorted(input_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_files = [fp for fp in class_dir.iterdir() if fp.suffix.lower() in self.AUDIO_EXTS]
            if self.cfg.max_per_class:
                class_files = class_files[: self.cfg.max_per_class]
            for fp in class_files:
                files.append((str(fp), class_dir.name))

        logger.info(f"Archivos encontrados: {len(files)} en {len({c for _, c in files})} clases")
        return files

    def run(self, workers: int | None = None) -> dict:
        """
        Ejecuta la extracción en batch con multiprocessing.

        Returns:
            Reporte con estadísticas de cobertura por clase.
        """
        files = self._collect_files()
        if not files:
            logger.error(f"No se encontraron archivos en: {self.cfg.input_dir}")
            return {"total": 0, "classes": {}}

        n_workers = workers or self.cfg.n_workers
        if n_workers <= 0:
            n_workers = max(1, multiprocessing.cpu_count() - 1)

        logger.info(f"Extrayendo features con {n_workers} worker(s)...")

        # Serializar config para pickle
        cfg_dict = asdict(self.cfg)

        # Preparar argumentos del worker
        work_args = [
            (fp, cls, self.cfg.output_dir, self.cfg.mfcc_dir, cfg_dict, i)
            for i, (fp, cls) in enumerate(files)
        ]

        results: list[dict] = []
        errors: list[str] = []
        skipped: int = 0

        # Usar n_workers=1 como single-process si hay problemas con multiprocessing
        if n_workers == 1 or os.name == "nt":  # Windows puede tener problemas con fork
            for args in work_args:
                r = _process_single_file(args)
                results.append(r)
                if r["status"] == "error":
                    errors.append(f"{r['file']}: {r['error']}")
                elif r["status"] == "skip":
                    skipped += 1
                # Progreso cada 100 archivos
                if (len(results)) % 100 == 0:
                    logger.info(f"  Progreso: {len(results)}/{len(files)}")
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_args = {
                    executor.submit(_process_single_file, args): args for args in work_args
                }
                for i, future in enumerate(as_completed(future_to_args)):
                    try:
                        r = future.result(timeout=120)
                        results.append(r)
                        if r["status"] == "error":
                            errors.append(f"{r['file']}: {r['error']}")
                        elif r["status"] == "skip":
                            skipped += 1
                    except Exception as e:
                        errors.append(f"Future error: {e}")

                    if (i + 1) % 100 == 0:
                        logger.info(f"  Progreso: {i+1}/{len(files)}")

        # ── Reporte de cobertura ──────────────────────────────────────────────
        report = self._build_report(results, errors, skipped)
        report_path = Path(self.cfg.output_dir) / "extraction_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n{'='*50}")
        logger.info("EXTRACCIÓN COMPLETADA")
        logger.info(f"  Total archivos:   {len(files)}")
        logger.info(f"  OK:               {report['n_ok']}")
        logger.info(f"  Saltados (cache): {skipped}")
        logger.info(f"  Errores:          {len(errors)}")
        logger.info(f"  Segmentos totales:{report['total_segments']}")
        logger.info(f"  Reporte: {report_path}")
        logger.info(f"{'='*50}")

        return report

    def _build_report(self, results: list[dict], errors: list[str], skipped: int) -> dict:
        """Construye el reporte de cobertura por clase."""
        from collections import defaultdict

        class_stats: dict[str, dict] = defaultdict(lambda: {"files": 0, "segments": 0, "errors": 0})

        # Obtener clases del output_dir actual
        out_dir = Path(self.cfg.output_dir)
        for class_dir in sorted(out_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            n_segs = len(list(class_dir.glob("*.npy")))
            class_stats[class_dir.name]["segments"] = n_segs

        for r in results:
            if r["status"] == "ok":
                class_stats[r.get("class", "unknown")]["files"] += 1
            elif r["status"] == "error":
                class_stats[r.get("class", "unknown")]["errors"] += 1

        total_segments = sum(v["segments"] for v in class_stats.values())

        return {
            "n_files": len(results),
            "n_ok": sum(1 for r in results if r["status"] == "ok"),
            "n_skipped": skipped,
            "n_errors": len(errors),
            "total_segments": total_segments,
            "classes": dict(class_stats),
            "errors": errors[:50],  # máximo 50 errores en el reporte
        }

    def validate(self) -> dict:
        """
        Valida la integridad de los .npy extraídos.
        Verifica que cada archivo sea cargable y tenga la forma correcta.
        """
        out_dir = Path(self.cfg.output_dir)
        corrupt: list[str] = []
        shape_errors: list[str] = []
        class_counts: dict[str, int] = {}

        for class_dir in sorted(out_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            count = 0
            for npy_path in class_dir.glob("*.npy"):
                try:
                    arr = np.load(str(npy_path))
                    if arr.ndim != 2 or arr.shape[0] != self.cfg.n_mels:
                        shape_errors.append(
                            f"{npy_path.name}: shape={arr.shape}, expected=({self.cfg.n_mels}, T)"
                        )
                    count += 1
                except Exception as e:
                    corrupt.append(f"{npy_path.name}: {e}")
            class_counts[class_dir.name] = count

        total = sum(class_counts.values())
        min_c = min(class_counts.values()) if class_counts else 0
        max_c = max(class_counts.values()) if class_counts else 0

        report = {
            "total_valid_files": total,
            "n_classes": len(class_counts),
            "min_segments_class": min_c,
            "max_segments_class": max_c,
            "imbalance_ratio": round(max_c / max(min_c, 1), 2),
            "corrupt_files": corrupt,
            "shape_errors": shape_errors[:20],
            "class_counts": class_counts,
        }

        logger.info(f"Validación: {total} segmentos válidos en {len(class_counts)} clases")
        if corrupt:
            logger.warning(f"  {len(corrupt)} archivos corruptos")
        if shape_errors:
            logger.warning(f"  {len(shape_errors)} errores de forma")

        return report


# ─────────────────────────────────────────────────────────────────────────────
# 4. UTILIDAD: VISUALIZAR MUESTRA DE ESPECTROGRAMAS
# ─────────────────────────────────────────────────────────────────────────────


def preview_spectrograms(
    output_dir: str = "data/spectrograms",
    n_per_class: int = 2,
    save_path: str = "results/visualizations/spectrograms_preview.png",
):
    """
    Visualiza una muestra aleatoria de espectrogramas del dataset extraído.
    Útil para verificar la calidad del preprocesamiento.
    """
    import random

    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    class_dirs = sorted([d for d in out_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        logger.warning("No hay espectrogramas para previsualizar.")
        return

    fig, axes = plt.subplots(
        len(class_dirs),
        n_per_class,
        figsize=(n_per_class * 4, len(class_dirs) * 2.5),
    )
    if len(class_dirs) == 1:
        axes = [axes]

    for row, cls_dir in enumerate(class_dirs):
        npys = list(cls_dir.glob("*.npy"))
        sample = random.sample(npys, min(n_per_class, len(npys)))

        for col in range(n_per_class):
            ax = axes[row][col] if n_per_class > 1 else axes[row]
            if col < len(sample):
                mel = np.load(str(sample[col]))
                ax.imshow(mel, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
                ax.set_title(f"{cls_dir.name}\n{mel.shape}", fontsize=7)
            ax.axis("off")

    plt.suptitle("Preview de Espectrogramas Mel — Dataset Extraído", fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()
    logger.info(f"Preview guardado: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="BioAcoustics — Batch Feature Extractor")
    parser.add_argument("--input", default="data/raw", help="Directorio de audio crudo")
    parser.add_argument("--output", default="data/spectrograms", help="Directorio de salida .npy")
    parser.add_argument("--mfcc-dir", default="data/features/mfcc", help="Directorio de MFCC")
    parser.add_argument(
        "--preset",
        default="default",
        choices=list(PRESET_OVERRIDES.keys()) + ["default"],
        help="Preset de configuración por grupo taxonómico",
    )
    parser.add_argument("--workers", type=int, default=0, help="Número de workers (0=auto)")
    parser.add_argument("--overwrite", action="store_true", help="Sobreescribir .npy existentes")
    parser.add_argument("--validate", action="store_true", help="Solo validar dataset extraído")
    parser.add_argument("--preview", action="store_true", help="Generar preview de espectrogramas")
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    cfg = ExtractionConfig(
        input_dir=args.input,
        output_dir=args.output,
        mfcc_dir=args.mfcc_dir,
        preset=args.preset,
        overwrite=args.overwrite,
        n_workers=args.workers,
        max_per_class=args.max_per_class,
    )
    cfg = apply_preset(cfg, args.preset)

    extractor = BatchExtractor(cfg)

    if args.validate:
        report = extractor.validate()
        print(json.dumps(report, indent=2))

    elif args.preview:
        preview_spectrograms(args.output)

    else:
        report = extractor.run(workers=args.workers)
        print(
            json.dumps(
                {
                    "total_files": report["n_files"],
                    "ok": report["n_ok"],
                    "skipped": report["n_skipped"],
                    "errors": report["n_errors"],
                    "total_segments": report["total_segments"],
                    "classes": {k: v["segments"] for k, v in report["classes"].items()},
                },
                indent=2,
            )
        )
