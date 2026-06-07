"""
models/efficientnet_classifier.py
─────────────────────────────────────────────────────────────────────────────
Clasificador bioacústico con transfer learning usando EfficientNet-B0/B4
de la librería timm. Estrategia de fine-tuning progresivo (unfreezing gradual).

Dependencias:
    pip install timm torch torchvision

Modelos disponibles vía timm:
    efficientnet_b0  ~5.3 M params  — rápido, ideal para prototyping
    efficientnet_b4  ~19.3 M params — balance accuracy/velocidad
    tf_efficientnetv2_s — state-of-the-art, más pesado

Autor: Ian
Versión: 1.0.0
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ── Importación condicional de timm ──────────────────────────────────────────
try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm no instalado. Ejecutar: pip install timm")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODELO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────


class EfficientNetBioAcoustic(nn.Module):
    """
    EfficientNet (cualquier variante de timm) adaptado a clasificación bioacústica.

    Modificaciones sobre el backbone original:
      - Canal de entrada configurable (1 = Mel mono → replicado a 3ch internamente).
      - Head de clasificación reemplazado: GlobalPool → BN → Dropout → FC.
      - Soporte para extracción de embeddings (modo features).

    Estrategia de fine-tuning:
      Fase 1 (frozen_backbone=True):  Solo head — aprendizaje rápido sin destruir features.
      Fase 2 (frozen_backbone=False): Unfreeze layers superiores — ajuste fino.
      Fase 3 (unfreeze_all=True):     Todo el backbone — máximo ajuste con LR muy bajo.

    Args:
        n_classes:       Número de clases de salida.
        backbone:        Nombre del modelo en timm. Default: 'efficientnet_b0'.
        in_channels:     Canales de entrada (1 o 3).
        dropout_rate:    Dropout antes de la capa de clasificación.
        frozen_backbone: Si True, congela el backbone en la inicialización.
        pretrained:      Si True, carga pesos de ImageNet (recomendado).
    """

    def __init__(
        self,
        n_classes: int,
        backbone: str = "efficientnet_b0",
        in_channels: int = 1,
        dropout_rate: float = 0.3,
        frozen_backbone: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("Instalar timm: pip install timm")

        self.n_classes = n_classes
        self.backbone_name = backbone
        self.in_channels = in_channels

        # ── Cargar backbone preentrenado ──────────────────────────────────────
        # num_classes=0 elimina el head original y retorna features
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # no head → devuelve features
            global_pool="avg",  # global average pooling
            in_chans=3,  # siempre 3 internamente
        )

        # Dimensión de salida del backbone (feature dim)
        self.feat_dim = self.backbone.num_features

        # ── Head de clasificación personalizado ───────────────────────────────
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.feat_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, n_classes),
        )

        # ── Proyección de canales (1ch → 3ch) si es necesario ─────────────────
        if in_channels == 1:
            # Aprendemos una proyección lineal: 1ch → 3ch antes del backbone
            self.channel_proj = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        else:
            self.channel_proj = nn.Identity()

        # ── Congelar backbone si se solicita ──────────────────────────────────
        if frozen_backbone:
            self.freeze_backbone()

        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Control de congelamiento ──────────────────────────────────────────────

    def freeze_backbone(self):
        """Congela todos los pesos del backbone. Solo entrena el head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone congelado. Solo head entrena.")

    def unfreeze_last_n_blocks(self, n: int = 2):
        """
        Descongela los últimos n bloques del backbone (fine-tuning progresivo).
        Usar con LR muy bajo (1e-5 a 1e-4).
        """
        blocks = list(self.backbone.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Últimos {n} bloques descongelados. Params entrenables: {trainable:,}")

    def unfreeze_all(self):
        """Descongela todo el backbone para fine-tuning completo."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info(f"Backbone completo descongelado. Params: {self.count_parameters():,}")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x:               Tensor (B, C, H, W) — espectrograma normalizado.
            return_features: Si True retorna embeddings (B, feat_dim) sin clasificar.
        Returns:
            logits (B, n_classes) o features (B, feat_dim).
        """
        # Proyección de canal si es mono
        x = self.channel_proj(x)  # (B, 1, H, W) → (B, 3, H, W)

        features = self.backbone(x)  # (B, feat_dim)

        if return_features:
            return features

        return self.head(features)  # (B, n_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extrae embeddings L2-normalizados para búsqueda por similitud."""
        feats = self.forward(x, return_features=True)
        return F.normalize(feats, p=2, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ENTRENADOR CON FINE-TUNING PROGRESIVO
# ─────────────────────────────────────────────────────────────────────────────


class ProgressiveFinetuner:
    """
    Entrenamiento en tres fases con descongelamiento progresivo:

      Fase 1 │ backbone congelado │ LR=1e-3  │ epochs_phase1
      Fase 2 │ últimos N bloques  │ LR=1e-4  │ epochs_phase2
      Fase 3 │ backbone completo  │ LR=5e-5  │ epochs_phase3

    Cada fase usa scheduler OneCycleLR independiente.
    """

    PHASES = {
        1: {"lr": 1e-3, "unfreeze": "head"},
        2: {"lr": 1e-4, "unfreeze": "last_blocks", "n_blocks": 3},
        3: {"lr": 5e-5, "unfreeze": "all"},
    }

    def __init__(
        self,
        model: EfficientNetBioAcoustic,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str | Path = "models/trained",
        epochs_phase1: int = 15,
        epochs_phase2: int = 20,
        epochs_phase3: int = 15,
        device: str | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = {1: epochs_phase1, 2: epochs_phase2, 3: epochs_phase3}

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
            label_smoothing=0.1,
        )
        self.history: dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.best_val_loss = float("inf")

    def _make_optimizer(self, lr: float):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )

    def _make_scheduler(self, optimizer, epochs: int):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            steps_per_epoch=len(self.train_loader),
            epochs=epochs,
            pct_start=0.1,
            anneal_strategy="cos",
        )

    def _run_epoch(self, optimizer, scheduler=None, train: bool = True) -> tuple[float, float]:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for x, y in self.train_loader if train else self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()

                total_loss += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)

        return total_loss / total, correct / total

    def _run_phase(self, phase: int):
        cfg = self.PHASES[phase]
        epochs = self.epochs[phase]

        # Descongelar según fase
        if cfg["unfreeze"] == "head":
            self.model.freeze_backbone()
        elif cfg["unfreeze"] == "last_blocks":
            self.model.unfreeze_last_n_blocks(cfg["n_blocks"])
        elif cfg["unfreeze"] == "all":
            self.model.unfreeze_all()

        optimizer = self._make_optimizer(cfg["lr"])
        scheduler = self._make_scheduler(optimizer, epochs)
        best_phase = float("inf")

        logger.info(f"\n{'='*60}")
        logger.info(f"FASE {phase} | LR={cfg['lr']} | Epochs={epochs} | {cfg['unfreeze']}")
        logger.info(f"Params entrenables: {self.model.count_parameters():,}")
        logger.info(f"{'='*60}")

        patience, patience_cnt = max(5, epochs // 3), 0

        for ep in range(1, epochs + 1):
            tr_loss, tr_acc = self._run_epoch(optimizer, scheduler, train=True)
            vl_loss, vl_acc = self._run_epoch(optimizer, scheduler=None, train=False)

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(vl_loss)
            self.history["val_acc"].append(vl_acc)

            logger.info(
                f"P{phase} E{ep:03d}/{epochs} | "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
                f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}"
            )

            # Checkpoint global
            if vl_loss < self.best_val_loss:
                self.best_val_loss = vl_loss
                patience_cnt = 0
                self._save_checkpoint(ep, phase, vl_loss, vl_acc)
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    logger.info(f"Early stopping en Fase {phase}, epoch {ep}")
                    break

            if vl_loss < best_phase:
                best_phase = vl_loss

    def _save_checkpoint(self, epoch: int, phase: int, val_loss: float, val_acc: float):
        path = self.output_dir / "best_efficientnet.pt"
        torch.save(
            {
                "epoch": epoch,
                "phase": phase,
                "model_state": self.model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "n_classes": self.model.n_classes,
                "backbone": self.model.backbone_name,
                "in_channels": self.model.in_channels,
                "history": self.history,
            },
            path,
        )
        logger.info(f"  ✓ Checkpoint → {path} (val_loss={val_loss:.4f})")

    def fit(self) -> dict:
        """Ejecuta las 3 fases de entrenamiento."""
        for phase in [1, 2, 3]:
            self._run_phase(phase)

        with open(self.output_dir / "efficientnet_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# 3. CARGA DE CHECKPOINT Y UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────


def load_efficientnet(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> EfficientNetBioAcoustic:
    """Carga EfficientNetBioAcoustic desde checkpoint."""
    ck = torch.load(checkpoint_path, map_location=device)
    model = EfficientNetBioAcoustic(
        n_classes=ck["n_classes"],
        backbone=ck.get("backbone", "efficientnet_b0"),
        in_channels=ck.get("in_channels", 1),
        frozen_backbone=False,
    )
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model.to(device)


def compare_backbones(n_classes: int = 50) -> dict:
    """
    Compara parámetros y velocidad de inferencia entre variantes.
    Útil para selección de arquitectura según restricciones de hardware.
    """
    import time

    if not TIMM_AVAILABLE:
        return {"error": "timm no disponible"}

    results = {}
    backbones = ["efficientnet_b0", "efficientnet_b4", "mobilenetv3_large_100"]
    dummy = torch.randn(8, 1, 128, 128)

    for bb in backbones:
        try:
            m = EfficientNetBioAcoustic(n_classes=n_classes, backbone=bb, pretrained=False)
            m.unfreeze_all()
            m.eval()

            t0 = time.time()
            with torch.no_grad():
                for _ in range(20):
                    _ = m(dummy)
            elapsed = (time.time() - t0) / 20 * 1000  # ms por batch

            results[bb] = {
                "params": m.count_parameters(),
                "feat_dim": m.feat_dim,
                "ms_per_batch8": round(elapsed, 2),
            }
        except Exception as e:
            results[bb] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONFIGURACIÓN RECOMENDADA POR GRUPO TAXONÓMICO
# ─────────────────────────────────────────────────────────────────────────────

RECOMMENDED_CONFIG: dict[str, dict] = {
    "bats": {
        "backbone": "efficientnet_b0",
        "input_size": (64, 256),  # (n_mels, T) — muestreo ultrasónico
        "dropout": 0.3,
        "epochs": (10, 15, 10),
        "note": "Alta variabilidad intra-clase; usar MixUp fuerte (alpha=0.6)",
    },
    "frogs": {
        "backbone": "efficientnet_b4",
        "input_size": (128, 128),
        "dropout": 0.4,
        "epochs": (15, 20, 15),
        "note": "Alta especificidad taxonómica; buenos resultados con backbone grande",
    },
    "insects": {
        "backbone": "efficientnet_b0",
        "input_size": (128, 128),
        "dropout": 0.3,
        "epochs": (10, 15, 10),
        "note": "Señales periódicas; SpecAugment moderado para no destruir estructura",
    },
    "mammals_audible": {
        "backbone": "efficientnet_b4",
        "input_size": (128, 256),
        "dropout": 0.4,
        "epochs": (20, 25, 20),
        "note": "Máxima variabilidad intraespecífica; dataset grande requerido (>1000/clase)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN — Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not TIMM_AVAILABLE:
        print("Instalar timm: pip install timm")
        exit(1)

    N_CLASSES = 20
    model = EfficientNetBioAcoustic(
        n_classes=N_CLASSES,
        backbone="efficientnet_b0",
        in_channels=1,
        frozen_backbone=True,  # Fase 1: solo head
        pretrained=False,  # False para demo sin descarga
    )

    print(f"Backbone: {model.backbone_name}")
    print(f"Feature dim: {model.feat_dim}")
    print(f"Params entrenables (solo head): {model.count_parameters():,}")

    # Forward pass de prueba
    dummy = torch.randn(4, 1, 128, 128)
    with torch.no_grad():
        logits = model(dummy)
        probs = model.predict_proba(dummy)
        embeddings = model.get_embeddings(dummy)

    print(f"\nInput:      {dummy.shape}")
    print(f"Logits:     {logits.shape}")
    print(f"Probs:      {probs.shape}  |  sum={probs.sum(dim=1)}")
    print(f"Embeddings: {embeddings.shape}  |  L2 norm={embeddings.norm(dim=1)}")

    # Comparativa de backbones
    print("\n--- Comparativa de backbones ---")
    results = compare_backbones(n_classes=N_CLASSES)
    for name, info in results.items():
        print(f"  {name:35s}: {info}")
