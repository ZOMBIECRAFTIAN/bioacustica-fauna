"""
scripts/export_model.py
─────────────────────────────────────────────────────────────────────────────
Exportación de modelos PyTorch a ONNX y TorchScript para deployment
en edge devices, servidores de inferencia (TensorRT, ONNX Runtime) y
entornos sin Python completo.

Formatos de salida
───────────────────
  TorchScript (.pt)   : serialización nativa PyTorch. Cargable sin código fuente.
  ONNX (.onnx)        : Open Neural Network Exchange. Compatible con ONNXRuntime,
                        TensorRT, OpenVINO, CoreML, etc.
  INT8 (ONNX)         : cuantización post-training a INT8 para inferencia rápida
                        en CPU (requiere onnxruntime + datos de calibración).

Benchmarking
─────────────
  Mide latencia media y throughput (muestras/s) de cada formato exportado
  y genera tabla comparativa para la tesis.

Validación
───────────
  Verifica que la salida ONNX/TorchScript difiera ≤ 1e-4 de la salida
  PyTorch original (tolerancia numérica por cuantización FP32→FP16).

Uso
───
    # Exportar todos los formatos
    python scripts/export_model.py \
        --checkpoint models/checkpoint_best.pth \
        --model-type efficientnet \
        --backbone efficientnet_b0 \
        --n-classes 17 \
        --output-dir models/exported \
        --formats torchscript onnx \
        --benchmark

    # Solo ONNX con cuantización INT8
    python scripts/export_model.py \
        --checkpoint models/checkpoint_best.pth \
        --model-type panns \
        --n-classes 17 \
        --output-dir models/exported \
        --formats onnx_int8 \
        --calibration-data data/processed/mammals

Autor: Ian
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE CARGA
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    model_type:      str,
    n_classes:       int,
    backbone:        str = "efficientnet_b0",
    device:          str = "cpu",
) -> nn.Module:
    """
    Carga el modelo desde checkpoint según el tipo.

    Parameters
    ----------
    checkpoint_path : path al archivo .pth
    model_type      : "cnn_baseline" | "efficientnet" | "panns"
    n_classes       : número de clases del head
    backbone        : para EfficientNet, ej. "efficientnet_b0"
    device          : "cpu" | "cuda"
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {path}")

    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    if model_type == "cnn_baseline":
        from src.models.cnn_baseline import BioAcousticCNN
        cfg = ckpt.get("config", {})
        model = BioAcousticCNN(
            n_classes  = n_classes,
            n_mels     = cfg.get("n_mels", 128),
            base_ch    = cfg.get("base_ch", 32),
            dropout    = 0.0,
        )

    elif model_type == "efficientnet":
        from src.models.efficientnet_classifier import EfficientNetBioAcoustic
        model = EfficientNetBioAcoustic(
            n_classes = n_classes,
            backbone  = backbone,
            pretrained= False,
            dropout   = 0.0,
        )

    elif model_type == "panns":
        from src.models.panns_classifier import PANNSCNN14BioAcoustic
        model = PANNSCNN14BioAcoustic(n_classes=n_classes, dropout=0.0)

    else:
        raise ValueError(f"model_type '{model_type}' no reconocido.")

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    logger.info("Modelo cargado: %s (%d clases)", model_type, n_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# FORMA DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def _dummy_input(
    model_type: str,
    n_mels:     int = 128,
    n_frames:   int = 256,
    device:     str = "cpu",
) -> torch.Tensor:
    """
    Tensor de entrada sintético (batch=1).

    Shape:
      cnn_baseline / efficientnet : (1, 1, n_mels, n_frames)
      panns                       : (1, 1, n_frames, n_mels)  ← formato PANNs
    """
    if model_type == "panns":
        return torch.randn(1, 1, n_frames, n_mels, device=device)
    return torch.randn(1, 1, n_mels, n_frames, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTACIÓN TORCHSCRIPT
# ─────────────────────────────────────────────────────────────────────────────

def export_torchscript(
    model:      nn.Module,
    dummy_in:   torch.Tensor,
    output_dir: Path,
    filename:   str = "model.pt",
) -> Path:
    """
    Exporta el modelo a TorchScript via torch.jit.trace.

    trace vs script
    ─────────────────
    trace  : registra el grafo siguiendo un input concreto. Más simple.
             Funciona para modelos sin flujo de control dinámico (if/for sobre Tensores).
    script : analiza el código Python estáticamente. Más robusto pero requiere
             anotaciones de tipo completas.

    BioAcousticCNN / EfficientNet / PANNs no tienen control flow dinámico
    sobre el tensor → trace es suficiente.
    """
    logger.info("Exportando TorchScript ...")
    model.eval()

    try:
        traced = torch.jit.trace(model, dummy_in, strict=False)
    except Exception as exc:
        logger.warning("trace falló, intentando script: %s", exc)
        traced = torch.jit.script(model)

    # Verificar reproducibilidad
    with torch.no_grad():
        out_orig   = model(dummy_in)
        out_traced = traced(dummy_in)

    diff = (out_orig - out_traced).abs().max().item()
    if diff > 1e-4:
        logger.warning("TorchScript: diferencia máxima = %.2e (umbral 1e-4)", diff)
    else:
        logger.info("TorchScript: diferencia máxima = %.2e ✓", diff)

    out_path = output_dir / filename
    traced.save(str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    logger.info("Guardado: %s (%.2f MB)", out_path, size_mb)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTACIÓN ONNX
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(
    model:         nn.Module,
    dummy_in:      torch.Tensor,
    output_dir:    Path,
    filename:      str = "model.onnx",
    opset_version: int = 17,
    dynamic_axes:  bool = True,
) -> Path:
    """
    Exporta a ONNX.

    opset_version: 17 es compatible con ONNXRuntime ≥ 1.15, TensorRT ≥ 8.6.
    dynamic_axes : permite batch size y longitud temporal variables en inferencia.
    """
    try:
        import onnx
    except ImportError:
        raise RuntimeError("pip install onnx onnxruntime")

    logger.info("Exportando ONNX (opset=%d) ...", opset_version)
    model.eval()

    out_path = output_dir / filename
    input_shape = dummy_in.shape   # (1, 1, H, W)

    dynamic = None
    if dynamic_axes:
        dynamic = {
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_in,
            str(out_path),
            opset_version      = opset_version,
            input_names        = ["input"],
            output_names       = ["output"],
            dynamic_axes       = dynamic,
            do_constant_folding= True,
            export_params      = True,
        )

    # Verificar modelo ONNX
    model_onnx = onnx.load(str(out_path))
    onnx.checker.check_model(model_onnx)
    logger.info("ONNX verificado: %d nodos", len(model_onnx.graph.node))

    # Validar numérica
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        inp  = {sess.get_inputs()[0].name: dummy_in.numpy()}
        out_onnx  = sess.run(None, inp)[0]
        out_torch = model(dummy_in).detach().numpy()
        diff = np.abs(out_onnx - out_torch).max()
        if diff > 1e-4:
            logger.warning("ONNX: diferencia máxima = %.2e", diff)
        else:
            logger.info("ONNX: diferencia máxima = %.2e ✓", diff)
    except ImportError:
        logger.info("onnxruntime no instalado — validación numérica omitida.")

    size_mb = out_path.stat().st_size / 1e6
    logger.info("Guardado: %s (%.2f MB)", out_path, size_mb)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CUANTIZACIÓN INT8
# ─────────────────────────────────────────────────────────────────────────────

def quantize_onnx_int8(
    onnx_path:        Path,
    output_dir:       Path,
    calibration_data: Optional[str] = None,
    filename:         str = "model_int8.onnx",
) -> Path:
    """
    Cuantización post-training a INT8 sobre el modelo ONNX.

    Usa onnxruntime.quantization con calibración estática si se proveen datos,
    o cuantización dinámica si no.

    Reducción de tamaño esperada: ~75% (FP32 → INT8).
    Pérdida de precisión típica: < 1% en accuracy top-1 para modelos bioacústicos.
    """
    try:
        from onnxruntime.quantization import (
            quantize_dynamic,
            quantize_static,
            CalibrationDataReader,
            QuantType,
        )
    except ImportError:
        raise RuntimeError("pip install onnxruntime")

    out_path = output_dir / filename

    if calibration_data:
        # Cuantización estática (más precisa)
        logger.info("Cuantización INT8 estática con calibración ...")

        class _CalibReader(CalibrationDataReader):
            def __init__(self, data_dir: str, n_samples: int = 100):
                self.data_dir = Path(data_dir)
                self.files    = list(self.data_dir.glob("**/*.npy"))[:n_samples]
                self._idx     = 0

            def get_next(self):
                if self._idx >= len(self.files):
                    return None
                arr = np.load(str(self.files[self._idx])).astype(np.float32)
                # Ajustar shape a (1,1,H,W)
                if arr.ndim == 2:
                    arr = arr[None, None, ...]
                elif arr.ndim == 3:
                    arr = arr[None, ...]
                self._idx += 1
                return {"input": arr}

        reader = _CalibReader(calibration_data)
        quantize_static(
            str(onnx_path),
            str(out_path),
            reader,
            quant_type=QuantType.QInt8,
        )
    else:
        # Cuantización dinámica (sin datos de calibración)
        logger.info("Cuantización INT8 dinámica ...")
        quantize_dynamic(
            str(onnx_path),
            str(out_path),
            weight_type=QuantType.QInt8,
        )

    size_orig = onnx_path.stat().st_size / 1e6
    size_int8 = out_path.stat().st_size / 1e6
    ratio     = (1 - size_int8 / size_orig) * 100
    logger.info(
        "INT8 guardado: %s (%.2f MB → %.2f MB, reducción %.1f%%)",
        out_path, size_orig, size_int8, ratio,
    )
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARKING
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_pytorch(
    model:    nn.Module,
    dummy_in: torch.Tensor,
    n_runs:   int = 100,
    warmup:   int = 10,
) -> Dict:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_in)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy_in)
            times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000  # ms
    return {
        "format":     "PyTorch (FP32)",
        "mean_ms":    float(np.mean(times)),
        "std_ms":     float(np.std(times)),
        "p50_ms":     float(np.percentile(times, 50)),
        "p95_ms":     float(np.percentile(times, 95)),
        "throughput": float(1000 / np.mean(times)),  # infer/s
    }


def benchmark_torchscript(
    ts_path:  Path,
    dummy_in: torch.Tensor,
    n_runs:   int = 100,
    warmup:   int = 10,
) -> Dict:
    model = torch.jit.load(str(ts_path))
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_in)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy_in)
            times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    return {
        "format":     "TorchScript (FP32)",
        "mean_ms":    float(np.mean(times)),
        "std_ms":     float(np.std(times)),
        "p50_ms":     float(np.percentile(times, 50)),
        "p95_ms":     float(np.percentile(times, 95)),
        "throughput": float(1000 / np.mean(times)),
    }


def benchmark_onnx(
    onnx_path: Path,
    dummy_in:  torch.Tensor,
    n_runs:    int = 100,
    warmup:    int = 10,
) -> Dict:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"format": str(onnx_path.name), "error": "onnxruntime no instalado"}

    sess    = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_np  = dummy_in.numpy()
    inp_key = sess.get_inputs()[0].name

    for _ in range(warmup):
        sess.run(None, {inp_key: inp_np})
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {inp_key: inp_np})
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    label = "ONNX FP32" if "int8" not in onnx_path.name else "ONNX INT8"
    return {
        "format":     label,
        "mean_ms":    float(np.mean(times)),
        "std_ms":     float(np.std(times)),
        "p50_ms":     float(np.percentile(times, 50)),
        "p95_ms":     float(np.percentile(times, 95)),
        "throughput": float(1000 / np.mean(times)),
    }


def print_benchmark_table(results: List[Dict]) -> None:
    """Imprime tabla comparativa de latencia."""
    SEP = "─" * 72
    FMT = "  {:<22} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print("\n" + SEP)
    print("  BENCHMARK DE LATENCIA (batch=1, CPU, n=100 runs)")
    print(SEP)
    print(FMT.format("Formato", "Mean(ms)", "Std(ms)", "P50(ms)", "P95(ms)", "inf/s"))
    print(SEP)
    for r in results:
        if "error" in r:
            print(f"  {r['format']:<22}  ERROR: {r['error']}")
            continue
        print(FMT.format(
            r["format"],
            f"{r['mean_ms']:.2f}",
            f"{r['std_ms']:.2f}",
            f"{r['p50_ms']:.2f}",
            f"{r['p95_ms']:.2f}",
            f"{r['throughput']:.1f}",
        ))
    print(SEP + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Exportar modelo bioacústico a ONNX / TorchScript"
    )
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--model-type",       required=True,
                        choices=["cnn_baseline", "efficientnet", "panns"])
    parser.add_argument("--n-classes",        type=int, required=True)
    parser.add_argument("--backbone",         default="efficientnet_b0")
    parser.add_argument("--output-dir",       default="models/exported")
    parser.add_argument("--device",           default="cpu")
    parser.add_argument("--n-mels",           type=int, default=128)
    parser.add_argument("--n-frames",         type=int, default=256)
    parser.add_argument("--opset",            type=int, default=17)
    parser.add_argument("--formats",          nargs="+",
                        default=["torchscript", "onnx"],
                        choices=["torchscript", "onnx", "onnx_int8"])
    parser.add_argument("--calibration-data", default=None,
                        help="Directorio .npy para cuantización INT8 estática")
    parser.add_argument("--benchmark",        action="store_true")
    parser.add_argument("--n-benchmark-runs", type=int, default=100)
    parser.add_argument("--class-names",      default=None,
                        help="JSON list o .txt con nombres de clase (para metadata)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar modelo
    model = load_model(
        args.checkpoint, args.model_type, args.n_classes,
        backbone=args.backbone, device=args.device,
    )

    dummy = _dummy_input(args.model_type, args.n_mels, args.n_frames, args.device)
    benchmark_results: List[Dict] = []

    # Benchmark PyTorch base
    if args.benchmark:
        logger.info("Benchmarking PyTorch ...")
        benchmark_results.append(
            benchmark_pytorch(model, dummy, n_runs=args.n_benchmark_runs)
        )

    exported_paths: Dict[str, Path] = {}

    # TorchScript
    if "torchscript" in args.formats:
        ts_path = export_torchscript(model, dummy, out_dir, "model.pt")
        exported_paths["torchscript"] = ts_path
        if args.benchmark:
            benchmark_results.append(
                benchmark_torchscript(ts_path, dummy, n_runs=args.n_benchmark_runs)
            )

    # ONNX FP32
    onnx_path = None
    if "onnx" in args.formats or "onnx_int8" in args.formats:
        onnx_path = export_onnx(
            model, dummy, out_dir, "model.onnx",
            opset_version=args.opset, dynamic_axes=True,
        )
        exported_paths["onnx"] = onnx_path
        if args.benchmark:
            benchmark_results.append(
                benchmark_onnx(onnx_path, dummy, n_runs=args.n_benchmark_runs)
            )

    # ONNX INT8
    if "onnx_int8" in args.formats and onnx_path is not None:
        int8_path = quantize_onnx_int8(
            onnx_path, out_dir,
            calibration_data=args.calibration_data,
        )
        exported_paths["onnx_int8"] = int8_path
        if args.benchmark:
            benchmark_results.append(
                benchmark_onnx(int8_path, dummy, n_runs=args.n_benchmark_runs)
            )

    # Tabla de benchmark
    if args.benchmark and benchmark_results:
        print_benchmark_table(benchmark_results)

    # Guardar metadata de exportación
    meta = {
        "model_type":   args.model_type,
        "backbone":     args.backbone,
        "n_classes":    args.n_classes,
        "n_mels":       args.n_mels,
        "n_frames":     args.n_frames,
        "checkpoint":   args.checkpoint,
        "exported":     {k: str(v) for k, v in exported_paths.items()},
        "benchmark":    benchmark_results,
        "input_shape":  list(dummy.shape),
    }
    if args.class_names:
        try:
            meta["class_names"] = json.loads(args.class_names)
        except Exception:
            p = Path(args.class_names)
            if p.exists():
                meta["class_names"] = [l.strip() for l in p.read_text().splitlines() if l.strip()]

    meta_path = out_dir / "export_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metadata guardada: %s", meta_path)

    print(f"\nModelos exportados en: {out_dir}")
    for fmt, path in exported_paths.items():
        size_mb = path.stat().st_size / 1e6
        print(f"  {fmt:<14} → {path.name}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
