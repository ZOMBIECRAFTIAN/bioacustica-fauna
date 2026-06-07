"""
src/evaluation/evaluator.py
─────────────────────────────────────────────────────────────────────────────
Pipeline de evaluación completo para modelos de clasificación bioacústica.

Métricas implementadas
──────────────────────
- Accuracy, Balanced Accuracy
- Precision, Recall, F1 (micro / macro / weighted / per-class)
- ROC-AUC (macro OvR), Average Precision (PR-AUC)
- Confusion matrix (normalizada y absoluta)
- Top-K accuracy (K=1,3,5)
- Expected Calibration Error (ECE)

Outputs
───────
- Dict JSON serializable con todas las métricas
- Figuras: confusion matrix, ROC curves, PR curves, calibration plot
- Reporte de texto por especie (taxa-level)
- Integración opcional con MLflow

Uso
───
    from src.evaluation.evaluator import ModelEvaluator, EvaluationConfig

    cfg = EvaluationConfig(output_dir="results/eval_run1", log_to_mlflow=True)
    ev  = ModelEvaluator(model, cfg, device="cuda")
    report = ev.evaluate(test_loader, class_names)
    report.save()

Autor: Ian
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Visualización
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve

# scikit-learn métricas
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationConfig:
    output_dir: str = "results/evaluation"
    # Figuras
    save_confusion_matrix: bool = True
    save_roc_curves: bool = True
    save_pr_curves: bool = True
    save_calibration: bool = True
    save_per_class_report: bool = True
    fig_dpi: int = 150
    fig_format: str = "png"  # png | pdf | svg
    # MLflow
    log_to_mlflow: bool = False
    mlflow_run_id: str | None = None
    # Inferencia
    batch_size: int = 32
    tta_n_augments: int = 0  # 0 = sin Test-Time Augmentation
    # Calibración ECE
    ece_n_bins: int = 10
    # Top-K
    top_k_values: list[int] = field(default_factory=lambda: [1, 3, 5])


# ─────────────────────────────────────────────────────────────────────────────
# REPORTE
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationReport:
    """Contenedor de resultados serializable a JSON."""

    # Metadata
    timestamp: str = ""
    model_type: str = ""
    n_test_samples: int = 0
    n_classes: int = 0
    inference_time_s: float = 0.0
    throughput_samples_per_s: float = 0.0

    # Métricas globales
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    roc_auc_macro: float = 0.0
    pr_auc_macro: float = 0.0
    ece: float = 0.0  # Expected Calibration Error
    brier_score: float = 0.0

    # Top-K
    top_k: dict[str, float] = field(default_factory=dict)

    # Por clase
    per_class: dict[str, dict] = field(default_factory=dict)

    # Paths de figuras generadas
    figures: dict[str, str] = field(default_factory=dict)

    # Config usada
    config: dict = field(default_factory=dict)

    _output_dir: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_output_dir", None)
        return d

    def save(self) -> Path:
        """Guarda el reporte JSON en output_dir."""
        out = Path(self._output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "evaluation_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        logger.info("Reporte guardado: %s", path)
        return path

    def print_summary(self) -> None:
        sep = "─" * 60
        print(sep)
        print(f"  EVALUACIÓN — {self.model_type}")
        print(sep)
        print(f"  Muestras:          {self.n_test_samples:,}")
        print(f"  Clases:            {self.n_classes}")
        print(f"  Throughput:        {self.throughput_samples_per_s:.1f} muestras/s")
        print(sep)
        print(f"  Accuracy:          {self.accuracy:.4f}")
        print(f"  Balanced Acc:      {self.balanced_accuracy:.4f}")
        print(f"  Macro F1:          {self.macro_f1:.4f}")
        print(f"  Weighted F1:       {self.weighted_f1:.4f}")
        print(f"  ROC-AUC (macro):   {self.roc_auc_macro:.4f}")
        print(f"  PR-AUC  (macro):   {self.pr_auc_macro:.4f}")
        print(f"  ECE:               {self.ece:.4f}")
        for k, v in self.top_k.items():
            print(f"  Top-{k} Acc:        {v:.4f}")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUADOR
# ─────────────────────────────────────────────────────────────────────────────


class ModelEvaluator:
    """
    Evalúa cualquier modelo PyTorch con interfaz forward(x) → logits.

    Parameters
    ----------
    model      : nn.Module — modelo en eval mode (se pone automáticamente)
    cfg        : EvaluationConfig
    device     : str — "cpu" | "cuda" | "mps"
    model_type : str — etiqueta para el reporte (ej. "efficientnet_b0")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: EvaluationConfig,
        device: str = "cpu",
        model_type: str = "bioacoustic_model",
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.model_type = model_type
        self.model.eval()

        self._out = Path(cfg.output_dir)
        self._out.mkdir(parents=True, exist_ok=True)

    # ── Inferencia ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_inference(
        self,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns
        -------
        y_true   : (N,)    — labels enteros
        y_prob   : (N, C)  — probabilidades softmax
        y_pred   : (N,)    — predicciones argmax
        elapsed  : float   — segundos totales
        """
        all_labels: list[int] = []
        all_probs: list[np.ndarray] = []

        t0 = time.perf_counter()
        for batch in loader:
            # Soporta tuplas (x, y) o dicts {"spectrogram": x, "label": y}
            if isinstance(batch, list | tuple):
                x, y = batch[0], batch[1]
            else:
                x, y = batch["spectrogram"], batch["label"]

            x = x.to(self.device, non_blocking=True)

            logits = self.model(x)  # (B, C)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(y.numpy().tolist())

        elapsed = time.perf_counter() - t0
        y_true = np.array(all_labels, dtype=np.int64)
        y_prob = np.array(all_probs, dtype=np.float32)
        y_pred = np.argmax(y_prob, axis=1)
        return y_true, y_prob, y_pred, elapsed

    # ── ECE ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_ece(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error (ECE) multiclase."""
        confidences = y_prob.max(axis=1)
        correctness = (np.argmax(y_prob, axis=1) == y_true).astype(float)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            acc = correctness[mask].mean()
            conf = confidences[mask].mean()
            ece += mask.sum() / n * abs(acc - conf)
        return float(ece)

    # ── Figures ───────────────────────────────────────────────────────────────

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str],
    ) -> Path:
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        n = len(class_names)
        figsize = max(8, n * 0.5 + 2)
        fig, axes = plt.subplots(1, 2, figsize=(figsize * 2, figsize))

        for ax, data, title, fmt in zip(
            axes,
            [cm, cm_norm],
            ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
            ["d", ".2f"],
        ):
            sns.heatmap(
                data,
                annot=n <= 30,
                fmt=fmt,
                ax=ax,
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues",
                linewidths=0.3 if n <= 30 else 0,
            )
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            ax.tick_params(axis="both", labelsize=max(5, 9 - n // 10))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0)

        fig.tight_layout()
        path = self._out / f"confusion_matrix.{self.cfg.fig_format}"
        fig.savefig(path, dpi=self.cfg.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figura guardada: %s", path)
        return path

    def _plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: list[str],
    ) -> Path:
        from sklearn.metrics import auc, roc_curve

        n_classes = len(class_names)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        palette = plt.cm.tab20(np.linspace(0, 1, n_classes))

        fig, ax = plt.subplots(figsize=(9, 7))

        # Macro-average
        all_fpr = np.unique(
            np.concatenate([roc_curve(y_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)])
        )
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(
            all_fpr,
            mean_tpr,
            "k--",
            lw=2,
            label=f"Macro-avg (AUC={macro_auc:.3f})",
        )

        # Por clase (solo si ≤ 20 clases para legibilidad)
        if n_classes <= 20:
            for i, (name, color) in enumerate(zip(class_names, palette)):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                a = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=1.2, alpha=0.8, label=f"{name} (AUC={a:.3f})")

        ax.plot([0, 1], [0, 1], "gray", lw=1, linestyle=":")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — per class")
        ax.legend(loc="lower right", fontsize=7, ncol=max(1, n_classes // 10))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.3)

        fig.tight_layout()
        path = self._out / f"roc_curves.{self.cfg.fig_format}"
        fig.savefig(path, dpi=self.cfg.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_pr_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: list[str],
    ) -> Path:
        from sklearn.metrics import auc, precision_recall_curve

        n_classes = len(class_names)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        palette = plt.cm.tab20(np.linspace(0, 1, n_classes))

        fig, ax = plt.subplots(figsize=(9, 7))

        if n_classes <= 20:
            for i, (name, color) in enumerate(zip(class_names, palette)):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                a = auc(rec, prec)
                ax.plot(rec, prec, color=color, lw=1.2, alpha=0.8, label=f"{name} (AP={a:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves — per class")
        ax.legend(loc="upper right", fontsize=7, ncol=max(1, n_classes // 10))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.3)

        fig.tight_layout()
        path = self._out / f"pr_curves.{self.cfg.fig_format}"
        fig.savefig(path, dpi=self.cfg.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: list[str],
    ) -> Path:
        n_classes = len(class_names)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        palette = plt.cm.tab20(np.linspace(0, 1, min(n_classes, 20)))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Reliability diagram (promedio de todas las clases)
        ax = axes[0]
        for i in range(min(n_classes, 20)):
            prob_true, prob_pred = calibration_curve(
                y_bin[:, i], y_prob[:, i], n_bins=self.cfg.ece_n_bins
            )
            ax.plot(
                prob_pred,
                prob_true,
                "o-",
                color=palette[i],
                alpha=0.6,
                lw=1.2,
                ms=4,
                label=class_names[i],
            )

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Reliability Diagram")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

        # Histograma de confianzas
        ax2 = axes[1]
        confidences = y_prob.max(axis=1)
        ax2.hist(confidences, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        ax2.axvline(
            confidences.mean(),
            color="red",
            lw=2,
            linestyle="--",
            label=f"Mean = {confidences.mean():.3f}",
        )
        ax2.set_xlabel("Max predicted probability (confidence)")
        ax2.set_ylabel("Count")
        ax2.set_title("Confidence Distribution")
        ax2.legend()
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        path = self._out / f"calibration.{self.cfg.fig_format}"
        fig.savefig(path, dpi=self.cfg.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_per_class_bar(
        self,
        per_class: dict[str, dict],
        class_names: list[str],
    ) -> Path:
        names = list(per_class.keys())
        f1s = [per_class[n]["f1"] for n in names]
        precs = [per_class[n]["precision"] for n in names]
        recs = [per_class[n]["recall"] for n in names]

        x = np.arange(len(names))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 5))
        ax.bar(x - w, precs, w, label="Precision", color="#4C72B0", alpha=0.85)
        ax.bar(x, recs, w, label="Recall", color="#DD8452", alpha=0.85)
        ax.bar(x + w, f1s, w, label="F1", color="#55A868", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Metrics — Precision / Recall / F1")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(
            np.mean(f1s), color="green", lw=1.5, linestyle="--", label=f"Mean F1={np.mean(f1s):.3f}"
        )
        ax.legend()

        fig.tight_layout()
        path = self._out / f"per_class_metrics.{self.cfg.fig_format}"
        fig.savefig(path, dpi=self.cfg.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── Método principal ──────────────────────────────────────────────────────

    def evaluate(
        self,
        loader: DataLoader,
        class_names: list[str],
        mlflow_run_id: str | None = None,
    ) -> EvaluationReport:
        """
        Ejecuta la evaluación completa.

        Parameters
        ----------
        loader      : DataLoader de test (sin shuffle)
        class_names : lista de nombres de clase en orden de índice
        mlflow_run_id : ID de run MLflow para logging (opcional)

        Returns
        -------
        EvaluationReport con todas las métricas y paths de figuras.
        """
        import datetime

        logger.info("Iniciando evaluación — modelo: %s", self.model_type)

        # ── Inferencia ─────────────────────────────────────────────────────
        y_true, y_prob, y_pred, elapsed = self._run_inference(loader)
        n = len(y_true)
        n_cls = len(class_names)
        logger.info("Inferencia completa: %d muestras en %.2fs", n, elapsed)

        # ── Métricas globales ──────────────────────────────────────────────
        y_bin = label_binarize(y_true, classes=list(range(n_cls)))

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        mac_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        wgt_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mac_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        mac_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

        try:
            roc_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            roc_auc = float("nan")

        try:
            pr_auc = average_precision_score(y_bin, y_prob, average="macro")
        except ValueError:
            pr_auc = float("nan")

        ece = self._compute_ece(y_true, y_prob, self.cfg.ece_n_bins)

        # Brier Score (multiclase: media por clase)
        brier = float(np.mean([brier_score_loss(y_bin[:, i], y_prob[:, i]) for i in range(n_cls)]))

        # Top-K
        top_k_metrics: dict[str, float] = {}
        for k in self.cfg.top_k_values:
            if k <= n_cls:
                try:
                    top_k_metrics[str(k)] = float(top_k_accuracy_score(y_true, y_prob, k=k))
                except Exception:
                    top_k_metrics[str(k)] = float("nan")

        # ── Métricas por clase ─────────────────────────────────────────────
        per_class: dict[str, dict] = {}
        per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
        class_counts = np.bincount(y_true, minlength=n_cls)

        for i, name in enumerate(class_names):
            try:
                auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
            except ValueError:
                auc_i = float("nan")
            per_class[name] = {
                "f1": float(per_f1[i]),
                "precision": float(per_prec[i]),
                "recall": float(per_rec[i]),
                "roc_auc": float(auc_i),
                "support": int(class_counts[i]),
            }

        # ── Figuras ────────────────────────────────────────────────────────
        figures: dict[str, str] = {}

        if self.cfg.save_confusion_matrix:
            p = self._plot_confusion_matrix(y_true, y_pred, class_names)
            figures["confusion_matrix"] = str(p)

        if self.cfg.save_roc_curves:
            p = self._plot_roc_curves(y_true, y_prob, class_names)
            figures["roc_curves"] = str(p)

        if self.cfg.save_pr_curves:
            p = self._plot_pr_curves(y_true, y_prob, class_names)
            figures["pr_curves"] = str(p)

        if self.cfg.save_calibration:
            p = self._plot_calibration(y_true, y_prob, class_names)
            figures["calibration"] = str(p)

        if self.cfg.save_per_class_report:
            p = self._plot_per_class_bar(per_class, class_names)
            figures["per_class_bar"] = str(p)

        # ── Reporte ────────────────────────────────────────────────────────
        report = EvaluationReport(
            timestamp=datetime.datetime.now().isoformat(),
            model_type=self.model_type,
            n_test_samples=n,
            n_classes=n_cls,
            inference_time_s=elapsed,
            throughput_samples_per_s=n / elapsed if elapsed > 0 else 0.0,
            accuracy=float(acc),
            balanced_accuracy=float(bal_acc),
            macro_f1=float(mac_f1),
            weighted_f1=float(wgt_f1),
            macro_precision=float(mac_prec),
            macro_recall=float(mac_rec),
            roc_auc_macro=float(roc_auc),
            pr_auc_macro=float(pr_auc),
            ece=float(ece),
            brier_score=float(brier),
            top_k=top_k_metrics,
            per_class=per_class,
            figures=figures,
            config=asdict(self.cfg),
            _output_dir=str(self._out),
        )

        # ── MLflow ────────────────────────────────────────────────────────
        if self.cfg.log_to_mlflow:
            self._log_mlflow(report, figures)

        report.print_summary()
        logger.info("Evaluación completada.")
        return report

    # ── MLflow logging ────────────────────────────────────────────────────────

    def _log_mlflow(
        self,
        report: EvaluationReport,
        figures: dict[str, str],
    ) -> None:
        try:
            import mlflow

            with mlflow.start_run(
                run_id=self.cfg.mlflow_run_id,
                nested=self.cfg.mlflow_run_id is not None,
            ):
                # Métricas escalares
                mlflow.log_metrics(
                    {
                        "test/accuracy": report.accuracy,
                        "test/balanced_accuracy": report.balanced_accuracy,
                        "test/macro_f1": report.macro_f1,
                        "test/weighted_f1": report.weighted_f1,
                        "test/roc_auc_macro": report.roc_auc_macro,
                        "test/pr_auc_macro": report.pr_auc_macro,
                        "test/ece": report.ece,
                        "test/brier_score": report.brier_score,
                        "test/throughput": report.throughput_samples_per_s,
                        **{f"test/top_{k}_acc": v for k, v in report.top_k.items()},
                        **{f"test/f1_{n}": d["f1"] for n, d in report.per_class.items()},
                    }
                )

                # Artefactos
                for path in figures.values():
                    mlflow.log_artifact(path, artifact_path="eval_figures")

                json_path = Path(self.cfg.output_dir) / "evaluation_report.json"
                if json_path.exists():
                    mlflow.log_artifact(str(json_path), artifact_path="reports")

            logger.info("Métricas registradas en MLflow.")

        except ImportError:
            logger.warning("mlflow no instalado — omitiendo logging.")
        except Exception as exc:
            logger.warning("Error MLflow: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from torch.utils.data import DataLoader

    from src.models.cnn_baseline import SpectrogramDataset
    from src.models.cnn_baseline import load_model as load_cnn

    parser = argparse.ArgumentParser(description="Evaluar modelo bioacústico")
    parser.add_argument("--checkpoint", required=True, help="Path al .pth")
    parser.add_argument("--data-dir", required=True, help="Directorio test con subdirs por clase")
    parser.add_argument("--output-dir", default="results/evaluation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--model-type", default="cnn_baseline", choices=["cnn_baseline", "efficientnet", "panns"]
    )
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    # Dataset
    dataset = SpectrogramDataset(root_dir=args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    class_names = dataset.classes

    # Modelo
    device = args.device
    model = load_cnn(args.checkpoint, device)

    # Evaluación
    cfg = EvaluationConfig(
        output_dir=args.output_dir,
        log_to_mlflow=args.mlflow,
    )
    ev = ModelEvaluator(model, cfg, device=device, model_type=args.model_type)
    report = ev.evaluate(loader, class_names)
    report.save()


if __name__ == "__main__":
    main()
