"""
models/train.py
─────────────────────────────────────────────────────────────────────────────
Script de entrenamiento principal del sistema BioAcoustics AI.
Lee configuración desde configs/train_config.yaml y ejecuta el pipeline
completo: dataset → modelo → entrenamiento → evaluación → reporte.

Logging con MLflow para trazabilidad de experimentos.

Uso:
    python -m src.models.train
    python -m src.models.train --config configs/train_config.yaml
    python -m src.models.train --model panns --epochs 30

Autor: Ian
Versión: 1.0.0
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Windows scientific stacks can load two Intel OpenMP runtimes through deps.
# Set this before importing torch so local training can start.
if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────


def load_config(path: str | Path) -> dict:
    """Carga y valida la configuración YAML."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Validaciones mínimas
    required_keys = ["dataset", "model", "training", "audio", "output"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Claves faltantes en config: {missing}")

    return cfg


def resolve_device(cfg_device: str) -> str:
    """Resuelve el dispositivo de cómputo."""
    if cfg_device == "auto":
        if torch.cuda.is_available():
            dev = f"cuda:{torch.cuda.current_device()}"
            logger.info(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = cfg_device
    logger.info(f"Dispositivo de entrenamiento: {dev}")
    return dev


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTRUCCIÓN DEL DATASET Y DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────


def build_dataloaders(
    cfg: dict,
    device: str,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Construye train/val/test DataLoaders desde el directorio de espectrogramas.

    Estrategia de split: stratified por clase.
    Balanceo de clases: WeightedRandomSampler en train.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    from src.models.cnn_baseline import SpectrogramDataset

    dcfg = cfg["dataset"]
    root_dir = Path(dcfg["root_dir"])

    if not root_dir.exists():
        raise FileNotFoundError(
            f"Directorio de dataset no encontrado: {root_dir}\n"
            f"Ejecutar primero: python -m src.feature_extraction.batch_extractor"
        )

    # Dataset completo para obtener clases
    full_ds = SpectrogramDataset(
        root_dir=root_dir,
        target_size=tuple(dcfg["target_size"]),
        normalize=dcfg["normalize"],
    )

    class_names = full_ds.classes
    n = len(full_ds)
    logger.info(f"Dataset: {n} muestras | {len(class_names)} clases")

    # Filtrar clases con pocas muestras
    min_samples = dcfg.get("min_samples_per_class", 10)
    from collections import Counter

    label_counts = Counter(lbl for _, lbl in full_ds.samples)
    valid_classes = {cls for cls, c in label_counts.items() if c >= min_samples}
    removed = len(class_names) - len(valid_classes)
    if removed > 0:
        logger.warning(f"Eliminadas {removed} clases con < {min_samples} muestras")
        full_ds.samples = [(fp, lbl) for fp, lbl in full_ds.samples if lbl in valid_classes]
        # Re-mapear labels al rango [0, n_valid)
        old_to_new = {old: new for new, old in enumerate(sorted(valid_classes))}
        full_ds.samples = [(fp, old_to_new[lbl]) for fp, lbl in full_ds.samples]
        class_names = [class_names[i] for i in sorted(valid_classes)]
        full_ds.classes = class_names

    # ── Split estratificado ───────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    indices = list(range(len(full_ds.samples)))
    labels = [lbl for _, lbl in full_ds.samples]

    train_pct = dcfg.get("train_split", 0.70)
    val_pct = dcfg.get("val_split", 0.15)

    # Primer split: train vs rest
    idx_train, idx_rest, lbl_train, lbl_rest = train_test_split(
        indices,
        labels,
        test_size=1 - train_pct,
        stratify=labels,
        random_state=42,
    )
    # Segundo split: val vs test
    val_ratio = val_pct / (1 - train_pct)
    idx_val, idx_test = train_test_split(
        idx_rest,
        test_size=1 - val_ratio,
        stratify=lbl_rest,
        random_state=42,
    )

    def make_subset(ds, indices):
        """Crea un subconjunto del dataset por índices."""
        import copy

        sub = copy.copy(ds)
        sub.samples = [ds.samples[i] for i in indices]
        return sub

    train_ds = make_subset(full_ds, idx_train)
    val_ds = make_subset(full_ds, idx_val)
    test_ds = make_subset(full_ds, idx_test)

    logger.info(f"Split — train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")

    # ── Sampler balanceado para train ─────────────────────────────────────────
    train_labels = [lbl for _, lbl in train_ds.samples]
    class_weights = full_ds.get_class_weights()
    sample_weights = [float(class_weights[lbl]) for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    tcfg = cfg["training"]
    n_workers = tcfg.get("num_workers", 2)

    # En Windows, num_workers > 0 requiere __main__ guard — reducir si hay problemas
    if sys.platform == "win32" and n_workers > 0:
        logger.info("Windows detectado — usando num_workers=0 para DataLoader")
        n_workers = 0

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=(device.startswith("cuda")),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=n_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=n_workers,
    )

    return train_loader, val_loader, test_loader, class_names


# ─────────────────────────────────────────────────────────────────────────────
# 3. INSTANCIACIÓN DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────


def build_model(
    cfg: dict,
    n_classes: int,
    device: str,
):
    """Instancia el modelo según la configuración."""
    mcfg = cfg["model"]
    model_type = mcfg["type"].lower()

    if model_type == "cnn_baseline":
        from src.models.cnn_baseline import BioAcousticCNN

        model = BioAcousticCNN(
            n_classes=n_classes,
            in_channels=mcfg.get("in_channels", 1),
            dropout_rate=mcfg.get("dropout", 0.4),
        )
        logger.info(f"CNN Baseline — {model.count_parameters():,} params")
        return model, "baseline"

    elif model_type == "efficientnet":
        from src.models.efficientnet_classifier import EfficientNetBioAcoustic

        model = EfficientNetBioAcoustic(
            n_classes=n_classes,
            backbone=mcfg.get("backbone", "efficientnet_b0"),
            in_channels=mcfg.get("in_channels", 1),
            dropout_rate=mcfg.get("dropout", 0.35),
            pretrained=mcfg.get("pretrained", True),
            frozen_backbone=True,
        )
        logger.info(f"EfficientNet — backbone={mcfg['backbone']}")
        return model, "efficientnet"

    elif model_type in ("panns", "panns_cnn14"):
        from src.models.panns_classifier import PANNSCNN14BioAcoustic

        model = PANNSCNN14BioAcoustic(
            n_classes=n_classes,
            dropout_rate=mcfg.get("dropout", 0.5),
            freeze_backbone=True,
        )
        # Intentar cargar pesos preentrenados
        loaded = model.load_pretrained_panns(download_if_missing=False)
        if not loaded:
            logger.warning("PANNs sin pesos preentrenados. Entrenando desde cero.")
        return model, "panns"

    else:
        raise ValueError(f"Tipo de modelo no reconocido: {model_type}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────


def run_training(
    cfg: dict,
    model,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    class_weights: torch.Tensor | None = None,
    output_dir: Path = Path("models/trained"),
):
    """Selecciona el entrenador correcto y ejecuta el training."""
    tcfg = cfg["training"]

    if model_type == "baseline":
        from src.models.cnn_baseline import Trainer

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            lr=tcfg["lr_phase1"],
            weight_decay=tcfg.get("weight_decay", 1e-4),
            use_mixup=tcfg.get("use_mixup", True),
            use_specaugment=tcfg.get("use_specaugment", True),
            patience=tcfg.get("patience", 10),
            device=device,
        )
        total_epochs = tcfg["phase1_epochs"] + tcfg["phase2_epochs"] + tcfg["phase3_epochs"]
        history = trainer.fit(epochs=total_epochs)

    elif model_type == "efficientnet":
        from src.models.efficientnet_classifier import ProgressiveFinetuner

        trainer = ProgressiveFinetuner(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            epochs_phase1=tcfg["phase1_epochs"],
            epochs_phase2=tcfg["phase2_epochs"],
            epochs_phase3=tcfg["phase3_epochs"],
            device=device,
            class_weights=class_weights,
        )
        history = trainer.fit()

    elif model_type == "panns":
        from src.models.panns_classifier import PANNsTrainer

        trainer = PANNsTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            epochs_per_phase=(
                tcfg["phase1_epochs"],
                tcfg["phase2_epochs"],
                tcfg["phase3_epochs"],
            ),
            device=device,
            class_weights=class_weights,
        )
        history = trainer.fit()

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUACIÓN FINAL Y REPORTE
# ─────────────────────────────────────────────────────────────────────────────


def run_evaluation(
    model,
    test_loader: DataLoader,
    class_names: list[str],
    device: str,
    output_dir: Path,
    experiment_name: str,
) -> dict:
    """Evaluación completa sobre test set + guardado de reporte."""
    from src.models.cnn_baseline import evaluate

    logger.info("Evaluando sobre test set...")
    results = evaluate(model, test_loader, device=device, class_names=class_names)

    logger.info(f"{'='*50}")
    logger.info(f"RESULTADOS FINALES — {experiment_name}")
    logger.info(f"  Accuracy:         {results['accuracy']:.4f}")
    logger.info(f"  F1-macro:         {results['f1_macro']:.4f}")
    logger.info(f"  Precision-macro:  {results['precision_macro']:.4f}")
    logger.info(f"  Recall-macro:     {results['recall_macro']:.4f}")
    logger.info(f"{'='*50}")

    # Per-class reporte
    logger.info("\nF1 por clase:")
    for cls, f1 in sorted(results["per_class_f1"].items(), key=lambda x: -x[1]):
        bar = "█" * int(f1 * 20)
        logger.info(f"  {cls:<35} {bar:<20} {f1:.4f}")

    # Guardar reporte
    report_path = output_dir / f"{experiment_name}_test_results.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "experiment": experiment_name,
                "n_classes": len(class_names),
                "class_names": class_names,
                "metrics": {k: v for k, v in results.items() if k != "confusion_matrix"},
                "confusion_matrix": results["confusion_matrix"],
            },
            f,
            indent=2,
        )

    logger.info(f"Reporte guardado: {report_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. INTEGRACIÓN MLFLOW
# ─────────────────────────────────────────────────────────────────────────────


def log_to_mlflow(
    cfg: dict,
    history: dict,
    results: dict,
    model,
    model_type: str,
    class_names: list[str],
    duration_s: float,
):
    """Registra el experimento en MLflow."""
    try:
        import mlflow
        import mlflow.pytorch

        exp_name = cfg["output"].get("experiment_name", "bioacoustics")
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(run_name=f"{model_type}_{int(time.time())}"):
            # Parámetros
            flat_cfg = {
                "model.type": cfg["model"]["type"],
                "model.backbone": cfg["model"].get("backbone", "N/A"),
                "training.batch": cfg["training"]["batch_size"],
                "training.phase1": cfg["training"]["phase1_epochs"],
                "training.phase2": cfg["training"]["phase2_epochs"],
                "training.phase3": cfg["training"]["phase3_epochs"],
                "dataset.n_classes": len(class_names),
                "audio.sample_rate": cfg["audio"]["sample_rate"],
                "audio.n_mels": cfg["audio"]["n_mels"],
            }
            mlflow.log_params(flat_cfg)

            # Métricas finales
            mlflow.log_metrics(
                {
                    "test_accuracy": results["accuracy"],
                    "test_f1_macro": results["f1_macro"],
                    "test_precision": results["precision_macro"],
                    "test_recall": results["recall_macro"],
                    "train_duration_s": duration_s,
                }
            )

            # Curvas de entrenamiento epoch por epoch
            for i, (tl, vl, ta, va) in enumerate(
                zip(
                    history["train_loss"],
                    history["val_loss"],
                    history["train_acc"],
                    history["val_acc"],
                )
            ):
                mlflow.log_metrics(
                    {
                        "train_loss": tl,
                        "val_loss": vl,
                        "train_acc": ta,
                        "val_acc": va,
                    },
                    step=i,
                )

            # Modelo
            mlflow.pytorch.log_model(model, artifact_path="model")

        logger.info(f"Experimento registrado en MLflow: {exp_name}")

    except ImportError:
        logger.warning("MLflow no instalado — saltando logging de experimento")
    except Exception as e:
        logger.warning(f"Error en MLflow: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────


def main(config_path: str = "configs/train_config.yaml"):
    """Punto de entrada del pipeline de entrenamiento."""
    t0 = time.time()

    # ── Logging ───────────────────────────────────────────────────────────────
    cfg = load_config(config_path)
    log_level = cfg["output"].get("log_level", "INFO")
    output_dir = Path(cfg["output"]["model_dir"])
    results_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(cfg["output"]["results_dir"]) / "train.log",
                mode="a",
                encoding="utf-8",
            ),
        ],
    )

    exp_name = cfg["output"]["experiment_name"]

    logger.info(f"{'='*60}")
    logger.info(f"BioAcoustics AI — Entrenamiento: {exp_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"{'='*60}")

    # ── Dispositivo ───────────────────────────────────────────────────────────
    device = resolve_device(cfg["training"].get("device", "auto"))

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = build_dataloaders(cfg, device)
    n_classes = len(class_names)
    logger.info(f"Clases: {class_names}")

    # Pesos de clase para loss balanceado
    class_weights = None
    if cfg["training"].get("use_class_weights", True):
        train_ds = train_loader.dataset
        if hasattr(train_ds, "get_class_weights"):
            class_weights = train_ds.get_class_weights()

    # Guardar mapeo de clases
    with open(output_dir / "class_names.json", "w") as f:
        json.dump({"classes": class_names, "n_classes": n_classes}, f, indent=2)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model, model_type = build_model(cfg, n_classes, device)
    model.to(device)

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    logger.info(f"Iniciando entrenamiento ({model_type})...")
    history = run_training(
        cfg,
        model,
        model_type,
        train_loader,
        val_loader,
        device,
        class_weights,
        output_dir,
    )

    # ── Cargar mejor checkpoint para evaluación ───────────────────────────────
    best_ckpt_names = {
        "baseline": "best_model.pt",
        "efficientnet": "best_efficientnet.pt",
        "panns": "best_panns_cnn14.pt",
    }
    ckpt_path = output_dir / best_ckpt_names[model_type]
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state"])
        logger.info(f"Mejor checkpoint cargado: {ckpt_path}")
    model.eval()

    # ── Evaluación final ──────────────────────────────────────────────────────
    results = run_evaluation(
        model,
        test_loader,
        class_names,
        device,
        results_dir,
        exp_name,
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    duration_s = time.time() - t0
    log_to_mlflow(cfg, history, results, model, model_type, class_names, duration_s)

    logger.info(f"\nEntrenamiento completado en {duration_s/60:.1f} minutos")
    logger.info(f"F1-macro test: {results['f1_macro']:.4f}")
    logger.info(f"Accuracy test: {results['accuracy']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioAcoustics AI — Training Pipeline")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Ruta al archivo de configuración YAML",
    )
    parser.add_argument(
        "--model",
        choices=["cnn_baseline", "efficientnet", "panns"],
        help="Sobreescribe model.type en la configuración",
    )
    parser.add_argument(
        "--backbone",
        choices=["efficientnet_b0", "efficientnet_b4", "mobilenetv3_large_100"],
        help="Backbone de EfficientNet",
    )
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Épocas totales (distribuidas en 3 fases)")
    parser.add_argument("--device", help="Dispositivo: auto, cpu, cuda, cuda:0")
    args = parser.parse_args()

    # Cargar config y aplicar overrides de CLI
    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["type"] = args.model
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.device:
        cfg["training"]["device"] = args.device
    if args.epochs:
        total = args.epochs
        phase1 = 1 if total > 0 else 0
        phase2 = 0
        phase3 = 0
        if total >= 2:
            phase1 = max(1, total // 4)
            phase2 = max(1, total // 2)
            phase3 = max(0, total - phase1 - phase2)
        cfg["training"]["phase1_epochs"] = phase1
        cfg["training"]["phase2_epochs"] = phase2
        cfg["training"]["phase3_epochs"] = phase3

    # Guardar config efectiva
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir="configs", delete=False, prefix="effective_", encoding="utf-8"
    ) as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        effective_cfg_path = f.name

    main(config_path=effective_cfg_path)

    try:
        os.unlink(effective_cfg_path)
    except Exception:
        pass
