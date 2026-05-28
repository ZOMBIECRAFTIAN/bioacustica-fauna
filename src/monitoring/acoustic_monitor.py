"""
src/monitoring/acoustic_monitor.py
─────────────────────────────────────────────────────────────────────────────
Monitor Acústico Pasivo (PAM — Passive Acoustic Monitoring) en tiempo real.

Arquitectura
─────────────
  ┌──────────────┐   PCM chunks    ┌──────────────┐   detecciones  ┌─────────┐
  │  AudioStream │ ──────────────► │  ProcessLoop │ ─────────────► │   DB    │
  │  (PyAudio)   │                 │  VAD+Preproc │                │  (PG)   │
  └──────────────┘                 │  +Inference  │                └─────────┘
                                   └──────────────┘
                                          │
                                    ┌─────▼──────┐
                                    │  JSONLines  │
                                    │  log file   │
                                    └────────────┘

Componentes
───────────
  AudioStream    : captura PCM vía PyAudio en chunks circulares (ring buffer).
  VADFilter      : energía RMS + zero-crossing para descartar silencio.
  ProcessLoop    : hilo de procesamiento: preprocesado → mel → inferencia.
  AcousticMonitor: orquestador principal con arranque/parada graceful.

Persistencia
─────────────
  - Detecciones escritas a PostgreSQL (tabla `detection`) vía asyncpg.
  - Log estructurado JSONL en results/logs/monitor_YYYYMMDD.jsonl.
  - Métricas Prometheus opcionales (si prometheus_client está instalado).

Uso
───
    from src.monitoring.acoustic_monitor import AcousticMonitor, MonitorConfig
    from src.models.cnn_baseline import load_model

    model = load_model("models/checkpoint_best.pth", "cpu")
    cfg   = MonitorConfig(
        device_index=0,
        sample_rate=22050,
        site_id="uuid-del-sitio",
        model_id="uuid-del-modelo",
        db_dsn="postgresql://user:pass@localhost/bioacoustics",
    )
    monitor = AcousticMonitor(model, cfg, class_names=["bat", "frog", ...])
    monitor.run()   # Ctrl+C para detener

CLI:
    python -m src.monitoring.acoustic_monitor \
        --checkpoint models/best.pth \
        --device-index 0 \
        --site-id <UUID> \
        --model-id <UUID>

Autor: Ian
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MonitorConfig:
    # Hardware / captura
    device_index:       int   = 0          # índice PyAudio del micrófono
    sample_rate:        int   = 22_050     # Hz; usar 192000 para murciélagos
    channels:           int   = 1
    chunk_samples:      int   = 1024       # frames por callback PyAudio
    buffer_seconds:     float = 3.0        # duración del bloque de análisis
    overlap_seconds:    float = 0.5        # solapamiento entre bloques

    # VAD (Voice Activity Detection)
    vad_rms_threshold:  float = 0.005      # señal por debajo → silencio
    vad_zcr_max:        float = 0.4        # ZCR muy alta → artefacto
    min_event_duration: float = 0.2        # s mínimos de actividad para procesar

    # Modelo
    model_device:       str   = "cpu"
    top_k:              int   = 3
    confidence_min:     float = 0.4        # umbral mínimo para registrar detección

    # IDs para BD
    site_id:            str   = ""
    model_id:           str   = ""
    device_record_id:   str   = ""

    # Persistencia
    db_dsn:             str   = ""         # vacío → sin BD
    log_dir:            str   = "results/logs"
    log_detections:     bool  = True

    # Prometheus (opcional)
    prometheus_port:    int   = 8001
    enable_prometheus:  bool  = False

    # Control
    max_queue_size:     int   = 50         # descarta bloques si la cola se llena
    n_worker_threads:   int   = 1


# ─────────────────────────────────────────────────────────────────────────────
# VAD — DETECTOR DE ACTIVIDAD VOCAL
# ─────────────────────────────────────────────────────────────────────────────

class VADFilter:
    """
    Filtro de actividad vocal simple basado en energía RMS y ZCR.

    RMS  > threshold      → señal presente
    ZCR  < zcr_max        → no es artefacto puro de HF
    """

    def __init__(self, cfg: MonitorConfig):
        self.rms_th  = cfg.vad_rms_threshold
        self.zcr_max = cfg.vad_zcr_max

    def is_active(self, y: np.ndarray) -> Tuple[bool, Dict]:
        rms = float(np.sqrt(np.mean(y ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2)
        active = rms > self.rms_th and zcr < self.zcr_max
        return active, {"rms": rms, "zcr": zcr}


# ─────────────────────────────────────────────────────────────────────────────
# PROCESADOR DE BLOQUES
# ─────────────────────────────────────────────────────────────────────────────

class BlockProcessor:
    """
    Preprocesa un bloque de audio crudo y ejecuta inferencia.

    Pipeline: raw_pcm → float32 → bandpass → normalize → mel_db → model
    """

    def __init__(
        self,
        model:       torch.nn.Module,
        class_names: List[str],
        cfg:         MonitorConfig,
    ):
        self.model       = model.to(cfg.model_device)
        self.class_names = class_names
        self.cfg         = cfg
        self.device      = cfg.model_device
        self.model.eval()

        # Importar preprocessor
        from src.audio_processing.preprocessor import AudioConfig, AudioPreprocessor
        audio_cfg   = AudioConfig(sample_rate=cfg.sample_rate, apply_noise_reduction=False)
        self.proc   = AudioPreprocessor(audio_cfg)

    @torch.no_grad()
    def process(self, y: np.ndarray, sr: int) -> Optional[Dict]:
        """
        Returns dict con predicciones o None si el bloque es descartable.
        """
        # 1. Preprocessado
        y = self.proc.bandpass_filter(y)
        y = self.proc.normalize(y, method="peak")

        # 2. Mel spectrogram
        mel = self.proc.mel_spectrogram(y)     # (n_mels, T)

        # 3. Tensor → (1, 1, n_mels, T)
        spec = torch.from_numpy(mel).float()
        spec = spec.unsqueeze(0).unsqueeze(0).to(self.device)

        # 4. Inferencia
        try:
            logits = self.model(spec)          # (1, n_classes)
        except Exception as exc:
            logger.warning("Inferencia fallida: %s", exc)
            return None

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # 5. Top-K
        top_idx   = np.argsort(probs)[::-1][: self.cfg.top_k]
        top_preds = [
            {
                "rank":        int(i + 1),
                "species":     self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}",
                "probability": float(probs[idx]),
            }
            for i, idx in enumerate(top_idx)
        ]

        best_prob = float(probs[top_idx[0]])
        if best_prob < self.cfg.confidence_min:
            return None

        return {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "top_k":      top_preds,
            "best":       top_preds[0],
            "mel_shape":  list(mel.shape),
        }


# ─────────────────────────────────────────────────────────────────────────────
# LOGGER JSONL
# ─────────────────────────────────────────────────────────────────────────────

class DetectionLogger:
    """Escribe detecciones en formato JSONL rotado diariamente."""

    def __init__(self, log_dir: str):
        self._dir  = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._date = None
        self._lock = threading.Lock()

    def _rotate(self) -> None:
        today = datetime.now().strftime("%Y%m%d")
        if today != self._date:
            if self._file:
                self._file.close()
            path       = self._dir / f"monitor_{today}.jsonl"
            self._file = open(path, "a", encoding="utf-8", buffering=1)
            self._date = today
            logger.info("Log rotado: %s", path)

    def write(self, record: Dict) -> None:
        with self._lock:
            self._rotate()
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._file:
            self._file.close()


# ─────────────────────────────────────────────────────────────────────────────
# WRITER POSTGRESQL
# ─────────────────────────────────────────────────────────────────────────────

class PostgresWriter:
    """
    Escribe detecciones en la tabla `detection` de PostgreSQL.
    Usa psycopg2 (síncrono) para evitar dependencias de event loop
    en el hilo de procesamiento.
    """

    def __init__(self, dsn: str, site_id: str, model_id: str):
        self.dsn      = dsn
        self.site_id  = site_id
        self.model_id = model_id
        self._conn    = None
        self._connect()

    def _connect(self) -> None:
        try:
            import psycopg2
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
            logger.info("PostgreSQL conectado.")
        except Exception as exc:
            logger.warning("PostgreSQL no disponible: %s — modo solo-log.", exc)
            self._conn = None

    def write(self, pred: Dict, segment_id: Optional[str] = None) -> None:
        if self._conn is None:
            return
        try:
            best    = pred["best"]
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO detection
                        (id, segment_id, model_id, detected_at,
                         confidence, raw_output)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    str(uuid.uuid4()),
                    segment_id,
                    self.model_id or None,
                    pred["timestamp"],
                    best["probability"],
                    json.dumps(pred["top_k"]),
                ))
        except Exception as exc:
            logger.warning("Error escribiendo en PostgreSQL: %s", exc)
            # Intentar reconexión
            try:
                self._connect()
            except Exception:
                pass

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# MONITOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class AcousticMonitor:
    """
    Orquestador principal del sistema PAM.

    Parameters
    ----------
    model        : modelo PyTorch cargado
    cfg          : MonitorConfig
    class_names  : lista de nombres de clase en orden de índice
    """

    def __init__(
        self,
        model:       torch.nn.Module,
        cfg:         MonitorConfig,
        class_names: List[str],
    ):
        self.cfg         = cfg
        self.class_names = class_names

        # Componentes
        self.vad       = VADFilter(cfg)
        self.processor = BlockProcessor(model, class_names, cfg)
        self.det_log   = DetectionLogger(cfg.log_dir) if cfg.log_detections else None
        self.pg_writer = (
            PostgresWriter(cfg.db_dsn, cfg.site_id, cfg.model_id)
            if cfg.db_dsn else None
        )

        # Ring buffer y cola de bloques
        self._buf_samples = int(cfg.buffer_seconds * cfg.sample_rate)
        self._hop_samples = int((cfg.buffer_seconds - cfg.overlap_seconds) * cfg.sample_rate)
        self._ring        = np.zeros(self._buf_samples, dtype=np.float32)
        self._ring_ptr    = 0
        self._block_queue: queue.Queue = queue.Queue(maxsize=cfg.max_queue_size)

        # Control de hilos
        self._running   = threading.Event()
        self._workers:  List[threading.Thread] = []

        # Estadísticas
        self.stats: Dict = {
            "blocks_captured":  0,
            "blocks_silent":    0,
            "blocks_processed": 0,
            "detections":       0,
            "errors":           0,
            "start_time":       None,
        }

        # Prometheus opcional
        self._prom_counters = {}
        if cfg.enable_prometheus:
            self._setup_prometheus()

        # Señal SIGINT / SIGTERM para parada graceful
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ── Prometheus ─────────────────────────────────────────────────────────

    def _setup_prometheus(self) -> None:
        try:
            from prometheus_client import Counter, Gauge, start_http_server
            start_http_server(self.cfg.prometheus_port)
            self._prom_counters = {
                "detections": Counter("bioacoustics_detections_total",
                                      "Total detections", ["species"]),
                "blocks":     Counter("bioacoustics_blocks_total",
                                      "Total audio blocks processed"),
                "silent":     Counter("bioacoustics_silent_blocks_total",
                                      "Blocks discarded as silent"),
            }
            logger.info("Prometheus en puerto %d", self.cfg.prometheus_port)
        except ImportError:
            logger.warning("prometheus_client no instalado — métricas desactivadas.")

    # ── SIGNAL handler ──────────────────────────────────────────────────────

    def _signal_handler(self, signum, frame):
        logger.info("Señal %d recibida — deteniendo monitor ...", signum)
        self._running.clear()

    # ── Captura de audio ───────────────────────────────────────────────────

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback de PyAudio: convierte PCM a float32 y llena el ring buffer."""
        import pyaudio
        try:
            # PCM int16 → float32 [-1, 1]
            samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Copiar al ring buffer circular
            n = len(samples)
            end = self._ring_ptr + n
            if end <= self._buf_samples:
                self._ring[self._ring_ptr: end] = samples
            else:
                # Wrap-around
                first = self._buf_samples - self._ring_ptr
                self._ring[self._ring_ptr:] = samples[:first]
                self._ring[:n - first]       = samples[first:]

            self._ring_ptr = (self._ring_ptr + n) % self._buf_samples
            self.stats["blocks_captured"] += 1

            # Cada hop_samples → encolar un bloque para análisis
            if self.stats["blocks_captured"] % max(1, self._hop_samples // frame_count) == 0:
                block = np.roll(self._ring, -self._ring_ptr).copy()
                if not self._block_queue.full():
                    self._block_queue.put_nowait(block)

        except Exception as exc:
            logger.debug("Callback error: %s", exc)
            self.stats["errors"] += 1

        return (None, pyaudio.paContinue)

    # ── Hilo de procesamiento ──────────────────────────────────────────────

    def _worker_loop(self) -> None:
        logger.info("Worker iniciado (tid=%d)", threading.get_ident())
        sr = self.cfg.sample_rate

        while self._running.is_set():
            try:
                block = self._block_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # VAD
            active, vad_info = self.vad.is_active(block)
            if not active:
                self.stats["blocks_silent"] += 1
                if "silent" in self._prom_counters:
                    self._prom_counters["silent"].inc()
                continue

            # Inferencia
            pred = self.processor.process(block, sr)
            self.stats["blocks_processed"] += 1

            if "blocks" in self._prom_counters:
                self._prom_counters["blocks"].inc()

            if pred is None:
                continue

            self.stats["detections"] += 1
            species = pred["best"]["species"]

            if "detections" in self._prom_counters:
                self._prom_counters["detections"].labels(species=species).inc()

            # Logging JSONL
            record = {
                "monitor_ts":   datetime.now(timezone.utc).isoformat(),
                "site_id":      self.cfg.site_id,
                "model_id":     self.cfg.model_id,
                "vad":          vad_info,
                "prediction":   pred,
            }
            if self.det_log:
                self.det_log.write(record)

            # PostgreSQL
            if self.pg_writer:
                self.pg_writer.write(pred)

            # Consola
            ts = pred["timestamp"]
            logger.info(
                "[%s] 🔊 %s  (conf=%.3f | top_k=%s)",
                ts,
                species,
                pred["best"]["probability"],
                " | ".join(
                    f"{p['species']}:{p['probability']:.2f}"
                    for p in pred["top_k"]
                ),
            )

    # ── Arranque y parada ──────────────────────────────────────────────────

    def run(self) -> None:
        """Inicia la captura y el bucle de procesamiento. Bloqueante."""
        try:
            import pyaudio
        except ImportError:
            raise RuntimeError(
                "pyaudio no instalado. Instalar con: pip install pyaudio"
            )

        logger.info(
            "Iniciando AcousticMonitor — SR=%d Hz, buffer=%.1fs, device=%d",
            self.cfg.sample_rate, self.cfg.buffer_seconds, self.cfg.device_index,
        )

        self._running.set()
        self.stats["start_time"] = datetime.now(timezone.utc).isoformat()

        # Hilos worker
        for _ in range(self.cfg.n_worker_threads):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

        # Stream PyAudio
        pa  = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.cfg.channels,
                rate=self.cfg.sample_rate,
                input=True,
                input_device_index=self.cfg.device_index,
                frames_per_buffer=self.cfg.chunk_samples,
                stream_callback=self._audio_callback,
            )
            stream.start_stream()
            logger.info("Stream de audio iniciado. Ctrl+C para detener.")

            # Esperar hasta señal de parada
            while self._running.is_set() and stream.is_active():
                time.sleep(0.2)
                self._print_stats()

        except Exception as exc:
            logger.error("Error en stream PyAudio: %s", exc)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            self._running.clear()

            # Esperar workers
            for t in self._workers:
                t.join(timeout=3.0)

            # Cerrar recursos
            if self.det_log:
                self.det_log.close()
            if self.pg_writer:
                self.pg_writer.close()

            self._print_final_stats()

    def _print_stats(self) -> None:
        """Imprime estadísticas en consola cada 10 detecciones."""
        if self.stats["detections"] % 10 == 0 and self.stats["detections"] > 0:
            logger.info(
                "Stats | captured=%d | silent=%d | processed=%d | detections=%d | errors=%d",
                self.stats["blocks_captured"],
                self.stats["blocks_silent"],
                self.stats["blocks_processed"],
                self.stats["detections"],
                self.stats["errors"],
            )

    def _print_final_stats(self) -> None:
        logger.info("─" * 60)
        logger.info("Monitor detenido. Resumen final:")
        for k, v in self.stats.items():
            logger.info("  %-25s %s", k + ":", v)
        logger.info("─" * 60)

    # ── Modo archivo (sin micrófono) ───────────────────────────────────────

    def process_file(self, filepath: str) -> List[Dict]:
        """
        Procesa un archivo de audio en modo batch (sin captura en tiempo real).
        Útil para pruebas o procesamiento diferido.

        Returns lista de predicciones por bloque activo.
        """
        import soundfile as sf

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(path)

        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)

        results: List[Dict] = []
        hop  = self._hop_samples
        buf  = self._buf_samples

        for start in range(0, max(1, len(y) - buf), hop):
            block = y[start: start + buf]
            if len(block) < buf:
                block = np.pad(block, (0, buf - len(block)))

            active, vad_info = self.vad.is_active(block)
            if not active:
                continue

            pred = self.processor.process(block, sr)
            if pred:
                pred["start_s"] = start / sr
                pred["end_s"]   = (start + buf) / sr
                pred["vad"]     = vad_info
                results.append(pred)

                if self.det_log:
                    self.det_log.write(pred)

        logger.info("Archivo procesado: %d detecciones en %s", len(results), path.name)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _list_audio_devices() -> None:
    """Lista los dispositivos de audio disponibles en el sistema."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        print(f"\nDispositivos de audio disponibles ({pa.get_device_count()}):")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']} — SR={info['defaultSampleRate']:.0f}Hz")
        pa.terminate()
    except ImportError:
        print("pyaudio no instalado.")


def main() -> None:
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    parser = argparse.ArgumentParser(description="Monitor acústico pasivo bioacústico")
    sub = parser.add_subparsers(dest="command")

    # Comando: run
    run_p = sub.add_parser("run", help="Iniciar captura en tiempo real")
    run_p.add_argument("--checkpoint",   required=True)
    run_p.add_argument("--class-names",  required=True, help="JSON list o archivo .txt")
    run_p.add_argument("--device-index", type=int, default=0)
    run_p.add_argument("--sample-rate",  type=int, default=22_050)
    run_p.add_argument("--buffer",       type=float, default=3.0, dest="buffer_seconds")
    run_p.add_argument("--overlap",      type=float, default=0.5, dest="overlap_seconds")
    run_p.add_argument("--site-id",      default="")
    run_p.add_argument("--model-id",     default="")
    run_p.add_argument("--db-dsn",       default="")
    run_p.add_argument("--log-dir",      default="results/logs")
    run_p.add_argument("--model-device", default="cpu")
    run_p.add_argument("--confidence",   type=float, default=0.4, dest="confidence_min")
    run_p.add_argument("--top-k",        type=int, default=3)

    # Comando: file
    file_p = sub.add_parser("file", help="Procesar archivo de audio")
    file_p.add_argument("--checkpoint",  required=True)
    file_p.add_argument("--class-names", required=True)
    file_p.add_argument("--input",       required=True)
    file_p.add_argument("--sample-rate", type=int, default=22_050)
    file_p.add_argument("--output",      default=None, help="Guardar JSON de resultados")

    # Comando: devices
    sub.add_parser("devices", help="Listar dispositivos de audio")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    if args.command == "devices":
        _list_audio_devices()
        return

    # Cargar class_names
    def load_class_names(raw: str) -> List[str]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            p = Path(raw)
            if p.exists():
                return [line.strip() for line in p.read_text().splitlines() if line.strip()]
            return raw.split(",")

    # Cargar modelo
    from src.models.cnn_baseline import load_model
    device = getattr(args, "model_device", "cpu")
    model  = load_model(args.checkpoint, device)

    class_names = load_class_names(args.class_names)

    if args.command == "run":
        cfg = MonitorConfig(
            device_index   = args.device_index,
            sample_rate    = args.sample_rate,
            buffer_seconds = args.buffer_seconds,
            overlap_seconds= args.overlap_seconds,
            site_id        = args.site_id,
            model_id       = args.model_id,
            db_dsn         = args.db_dsn,
            log_dir        = args.log_dir,
            model_device   = args.model_device,
            confidence_min = args.confidence_min,
            top_k          = args.top_k,
        )
        monitor = AcousticMonitor(model, cfg, class_names)
        monitor.run()

    elif args.command == "file":
        cfg = MonitorConfig(
            sample_rate  = args.sample_rate,
            model_device = getattr(args, "model_device", "cpu"),
        )
        monitor = AcousticMonitor(model, cfg, class_names)
        results = monitor.process_file(args.input)

        print(f"\nResultados: {len(results)} detecciones")
        for r in results:
            print(f"  [{r['start_s']:.2f}s → {r['end_s']:.2f}s]  "
                  f"{r['best']['species']}  (conf={r['best']['probability']:.3f})")

        if args.output:
            Path(args.output).write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
