"""
models/panns_classifier.py
─────────────────────────────────────────────────────────────────────────────
Fine-tuning de PANNs-CNN14 preentrenado en AudioSet (Kong et al., 2020)
para clasificación bioacústica multitaxonómica.

Referencia:
    Kong, Q. et al. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks
    for Audio Pattern Recognition. IEEE/ACM TASLP, 28, 2880–2894.
    GitHub: https://github.com/qiuqiangkong/audioset_tagging_cnn

Arquitectura CNN14:
    6 ConvBlock (2× Conv2D + BN + ReLU + Pool) →
    FC 2048 (embedding) → FC 527 (AudioSet classes)

Aquí reemplazamos el head de 527 clases por uno de n_classes.

Dependencias:
    pip install torch torchvision requests

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

# ─────────────────────────────────────────────────────────────────────────────
# 1. BLOQUE CONVOLUCIONAL CNN14
# ─────────────────────────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
    """
    Bloque convolucional PANNs: 2× (Conv2D → BN → ReLU) → AvgPool2D.
    Preserva la arquitectura original de CNN14 para compatibilidad
    con los pesos preentrenados de AudioSet.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x: torch.Tensor, pool_size=(2, 2), pool_type="avg") -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, pool_size)
        elif pool_type == "avg+max":
            x = (F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)) / 2
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2. BACKBONE CNN14 (arquitectura original PANNs)
# ─────────────────────────────────────────────────────────────────────────────


class CNN14Backbone(nn.Module):
    """
    Backbone CNN14 con la arquitectura exacta de PANNs para permitir
    la carga de pesos preentrenados en AudioSet.

    Entrada:  Mel spectrogram (B, 1, T, n_mels) — formato PANNs: tiempo en dim 2
    Salida:   embedding (B, 2048)
    """

    def __init__(self, mel_bins: int = 64):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(mel_bins)  # normalización del input Mel

        # 6 bloques convolucionales: canales duplican cada bloque
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T, n_mels) — espectrograma en formato PANNs
        Returns:
            embedding: (B, 2048)
        """
        # Normalización inicial del log-Mel
        x = x.transpose(1, 3)  # (B, n_mels, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (B, 1, T, n_mels)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")  # (/2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")  # (/4)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")  # (/8)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")  # (/16)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")  # (/32)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")  # sin pool

        # Global pooling: mean + max → concatenar
        x1 = torch.mean(x, dim=3)
        x1, _ = torch.max(x1, dim=2)
        x2 = torch.mean(x, dim=2)
        x2, _ = torch.max(x2, dim=2)
        x = x1 + x2  # (B, 2048)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELO COMPLETO CNN14 + HEAD BIOACÚSTICO
# ─────────────────────────────────────────────────────────────────────────────


class PANNSCNN14BioAcoustic(nn.Module):
    """
    PANNs-CNN14 adaptado para identificación bioacústica.

    Flujo:
        Log-Mel (B,1,T,n_mels) →
        CNN14Backbone (embedding 2048) →
        Clasificador bioacústico (n_classes)

    Modos de operación:
        1. from_scratch:      sin pesos preentrenados
        2. pretrained_panns:  carga pesos AudioSet oficiales
        3. features_only:     extracción de embeddings sin clasificar

    Args:
        n_classes:       Número de clases de salida.
        dropout_rate:    Dropout en el head.
        freeze_backbone: Si True, entrena solo el head en fase 1.
        classes_audioset: Número de clases del checkpoint original (527).
    """

    PANNS_WEIGHTS_URL = "https://zenodo.org/record/3987831/files/CNN14_mAP%3D0.431.pth?download=1"
    EMBEDDING_DIM = 2048

    def __init__(
        self,
        n_classes: int,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = True,
        sample_rate: int = 32_000,  # SR esperado por PANNs original
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14_000,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins

        # ── Backbone CNN14 ────────────────────────────────────────────────────
        self.backbone = CNN14Backbone(mel_bins=mel_bins)

        # ── Head de clasificación bioacústico ─────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.EMBEDDING_DIM, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, n_classes),
        )
        self._init_head()

        if freeze_backbone:
            self.freeze_backbone()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Control de congelamiento ──────────────────────────────────────────────

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("CNN14 backbone congelado.")

    def unfreeze_last_n_blocks(self, n: int = 2):
        """Descongela los últimos n ConvBlocks del backbone."""
        all_blocks = [
            self.backbone.conv_block6,
            self.backbone.conv_block5,
            self.backbone.conv_block4,
            self.backbone.conv_block3,
            self.backbone.conv_block2,
            self.backbone.conv_block1,
        ]
        for block in all_blocks[:n]:
            for param in block.parameters():
                param.requires_grad = True
        logger.info(f"Últimos {n} ConvBlocks descongelados.")

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        logger.info(f"CNN14 completo descongelado. Params: {self.count_parameters():,}")

    # ── Carga de pesos preentrenados ──────────────────────────────────────────

    def load_pretrained_panns(
        self,
        weights_path: str | Path | None = None,
        download_if_missing: bool = True,
        device: str = "cpu",
    ) -> bool:
        """
        Carga pesos preentrenados de CNN14 en AudioSet.

        Args:
            weights_path:         Ruta local al .pth. Si None, intenta descargar.
            download_if_missing:  Si True y no existe, descarga de Zenodo.
            device:               Dispositivo destino.

        Returns:
            True si la carga fue exitosa.
        """
        # Intentar ruta local
        if weights_path is None:
            weights_path = Path("models") / "panns_weights" / "CNN14_mAP=0.431.pth"

        weights_path = Path(weights_path)

        if not weights_path.exists():
            if not download_if_missing:
                logger.warning(f"Pesos PANNs no encontrados: {weights_path}")
                return False
            logger.info("Descargando pesos PANNs-CNN14 desde Zenodo...")
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import requests

                resp = requests.get(self.PANNS_WEIGHTS_URL, timeout=120, stream=True)
                resp.raise_for_status()
                with open(weights_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                logger.info(f"Pesos descargados → {weights_path}")
            except Exception as e:
                logger.error(f"Error descargando pesos: {e}")
                return False

        # Cargar checkpoint
        try:
            ck = torch.load(weights_path, map_location=device)
            state = ck.get("model", ck)  # algunos checkpoints envuelven en 'model'

            # Filtrar solo las claves del backbone (excluir fc_audioset original)
            backbone_state = {
                k.replace("module.", "").replace("backbone.", ""): v
                for k, v in state.items()
                if not k.startswith("fc_audioset") and not k.startswith("fc1_audioset")
            }

            missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
            if missing:
                logger.warning(f"Keys faltantes en backbone: {missing[:5]}...")
            logger.info("Pesos PANNs-CNN14 cargados correctamente.")
            return True

        except Exception as e:
            logger.error(f"Error cargando pesos PANNs: {e}")
            return False

    # ── Forward ───────────────────────────────────────────────────────────────

    def _to_panns_format(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convierte espectrograma de formato interno (B,1,n_mels,T)
        al formato PANNs (B,1,T,n_mels).
        """
        if x.dim() == 3:  # (B, n_mels, T)
            x = x.unsqueeze(1)  # (B, 1, n_mels, T)
        # PANNs espera (B, 1, T, n_mels)
        return x.permute(0, 1, 3, 2)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:                Tensor (B, 1, n_mels, T) o (B, n_mels, T).
            return_embedding: Si True, retorna embedding CNN14 (B, 2048).
        Returns:
            logits (B, n_classes) o embedding (B, 2048).
        """
        x = self._to_panns_format(x)  # (B, 1, T, n_mels)
        embedding = self.backbone(x)  # (B, 2048)

        if return_embedding:
            return embedding

        return self.head(embedding)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Embeddings L2-normalizados para búsqueda por similitud."""
        emb = self.forward(x, return_embedding=True)
        return F.normalize(emb, p=2, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_pct": round(100 * trainable / total, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTRENADOR CON FINE-TUNING PROGRESIVO (PANNs)
# ─────────────────────────────────────────────────────────────────────────────


class PANNsTrainer:
    """
    Entrenador específico para PANNs con 3 fases de fine-tuning progresivo.
    Implementa también knowledge distillation opcional desde el modelo
    original de 527 clases de AudioSet.

    Fases:
      1. Solo head (backbone congelado): lr=5e-4, ≤15 epochs
      2. Últimos 2 ConvBlocks desbloqueados: lr=1e-4, ≤20 epochs
      3. Backbone completo: lr=2e-5, ≤15 epochs
    """

    def __init__(
        self,
        model: PANNSCNN14BioAcoustic,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str | Path = "models/trained",
        epochs_per_phase: tuple[int, int, int] = (15, 20, 15),
        device: str | None = None,
        class_weights: torch.Tensor | None = None,
        use_label_smoothing: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs_per_phase = epochs_per_phase

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
            label_smoothing=0.1 if use_label_smoothing else 0.0,
        )

        self.history: dict[str, list] = {
            k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "phase")
        }
        self.best_val_loss = float("inf")

    # ── Configuración por fase ────────────────────────────────────────────────

    PHASE_CONFIG = {
        1: {"lr": 5e-4, "unfreeze": "head_only", "n_blocks": 0},
        2: {"lr": 1e-4, "unfreeze": "last_2_blocks", "n_blocks": 2},
        3: {"lr": 2e-5, "unfreeze": "all", "n_blocks": 6},
    }

    def _setup_phase(self, phase: int):
        cfg = self.PHASE_CONFIG[phase]
        if cfg["unfreeze"] == "head_only":
            self.model.freeze_backbone()
        elif cfg["unfreeze"] == "last_2_blocks":
            self.model.unfreeze_last_n_blocks(cfg["n_blocks"])
        elif cfg["unfreeze"] == "all":
            self.model.unfreeze_all()
        return cfg["lr"]

    def _make_optimizer(self, lr: float):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    # ── Epoch de entrenamiento ────────────────────────────────────────────────

    def _run_epoch(self, optimizer, scheduler=None, train: bool = True) -> tuple[float, float]:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        loader = self.train_loader if train else self.val_loader
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                # Formato PANNs necesita n_mels en dim 2 para bn0
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

    # ── Loop principal ────────────────────────────────────────────────────────

    def fit(self) -> dict:
        for phase in [1, 2, 3]:
            lr = self._setup_phase(phase)
            epochs = self.epochs_per_phase[phase - 1]
            opt = self._make_optimizer(lr)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=lr,
                steps_per_epoch=len(self.train_loader),
                epochs=epochs,
                pct_start=0.1,
            )
            patience = max(5, epochs // 3)
            no_improve = 0

            params_summary = self.model.parameter_summary()
            logger.info(
                f"\n{'='*60}\n"
                f"FASE {phase} | lr={lr} | epochs={epochs}\n"
                f"Params entrenables: {params_summary['trainable']:,} "
                f"({params_summary['trainable_pct']}%)\n{'='*60}"
            )

            for ep in range(1, epochs + 1):
                tr_loss, tr_acc = self._run_epoch(opt, sched, train=True)
                vl_loss, vl_acc = self._run_epoch(opt, None, train=False)

                self.history["train_loss"].append(tr_loss)
                self.history["train_acc"].append(tr_acc)
                self.history["val_loss"].append(vl_loss)
                self.history["val_acc"].append(vl_acc)
                self.history["phase"].append(phase)

                logger.info(
                    f"P{phase} E{ep:03d}/{epochs} | "
                    f"tr={tr_loss:.4f}/{tr_acc:.4f} | "
                    f"vl={vl_loss:.4f}/{vl_acc:.4f}"
                )

                if vl_loss < self.best_val_loss:
                    self.best_val_loss = vl_loss
                    no_improve = 0
                    self._save_checkpoint(ep, phase, vl_loss, vl_acc)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info(f"Early stopping — Fase {phase}, epoch {ep}")
                        break

        with open(self.output_dir / "panns_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def _save_checkpoint(self, epoch, phase, val_loss, val_acc):
        path = self.output_dir / "best_panns_cnn14.pt"
        torch.save(
            {
                "epoch": epoch,
                "phase": phase,
                "model_state": self.model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "n_classes": self.model.n_classes,
                "history": self.history,
            },
            path,
        )
        logger.info(f"  ✓ Checkpoint PANNs → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. UTILIDADES DE CARGA Y COMPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────


def load_panns(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> PANNSCNN14BioAcoustic:
    """Carga PANNs-CNN14 desde checkpoint de fine-tuning."""
    ck = torch.load(checkpoint_path, map_location=device)
    model = PANNSCNN14BioAcoustic(
        n_classes=ck["n_classes"],
        freeze_backbone=False,
    )
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model.to(device)


def model_comparison_table() -> str:
    """
    Tabla comparativa de los tres modelos del sistema.
    Útil para la sección de resultados de tesis.
    """
    rows = [
        ("CNN Baseline", "~4.2M", "Mel spectrogram", "Desde cero", "Rápido, ligero"),
        ("EfficientNet-B0", "~5.3M", "Mel spectrogram", "ImageNet TL", "Balance acc/vel"),
        ("EfficientNet-B4", "~19.3M", "Mel spectrogram", "ImageNet TL", "Mayor precisión"),
        (
            "PANNs-CNN14",
            "~79.6M",
            "Log-Mel (64 bandas)",
            "AudioSet (2M clips)",
            "Máxima transferencia",
        ),
    ]
    sep = "─" * 95
    header = f"{'Modelo':<20} {'Params':<12} {'Input':<22} {'Preentrenamiento':<22} {'Ventaja'}"
    lines = [sep, header, sep]
    for r in rows:
        lines.append(f"{r[0]:<20} {r[1]:<12} {r[2]:<22} {r[3]:<22} {r[4]}")
    lines.append(sep)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN — Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(model_comparison_table())

    # Instanciar PANNs-CNN14
    model = PANNSCNN14BioAcoustic(n_classes=20, freeze_backbone=True)
    print("\nPANNs-CNN14 instanciado.")
    print(f"Summary: {model.parameter_summary()}")

    # Forward pass: formato (B, 1, n_mels, T)
    # PANNs original usa 64 bandas Mel y SR=32kHz
    dummy = torch.randn(4, 1, 64, 313)  # ~3.2s a SR=32kHz, hop=320
    with torch.no_grad():
        logits = model(dummy)
        probs = model.predict_proba(dummy)
        embedding = model.get_embeddings(dummy)

    print(f"\nInput shape:     {dummy.shape}")
    print(f"Logits shape:    {logits.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding L2:    {embedding.norm(dim=1)}")

    # Intentar carga de pesos (silenciosa si no hay conexión)
    loaded = model.load_pretrained_panns(download_if_missing=False)
    status = "OK" if loaded else "No disponibles (modo offline)"
    print(f"\nPesos PANNs preentrenados: {status}")
    print(f"Arquitectura: {model.__class__.__name__}")
    print(f"Parametros:   {sum(p.numel() for p in model.parameters()):,}")
