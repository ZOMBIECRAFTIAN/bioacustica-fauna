"""
models/cnn_baseline.py
─────────────────────────────────────────────────────────────────────────────
CNN baseline desde cero para clasificación bioacústica.
Input: espectrograma Mel normalizado (1, n_mels, T) o (3, n_mels, T)
Output: distribución de probabilidad sobre N clases de especie/grupo.

Arquitectura: 4 bloques convolucionales residuales + BN + Dropout + FC head.

Dependencias:
    pip install torch torchvision

Autor: Ian
Versión: 1.0.0
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. BLOQUES ARQUITECTÓNICOS
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Bloque básico: Conv2D → BatchNorm → ReLU"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Bloque residual con skip-connection.
    Si in_ch != out_ch usa proyección 1×1 en el shortcut.

    Complejidad: O(H·W·C_in·C_out·k²)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Projection shortcut si dimensiones cambian
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class AttentionPool(nn.Module):
    """
    Atención temporal sobre el eje de tiempo del espectrograma.
    Aprende qué frames son más relevantes para la clasificación.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        weights = torch.softmax(self.attn(x).flatten(2), dim=-1)  # (B, 1, H*W)
        feats   = x.flatten(2)                                      # (B, C, H*W)
        return (feats * weights).sum(dim=-1)                        # (B, C)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODELO CNN BASELINE
# ─────────────────────────────────────────────────────────────────────────────

class BioAcousticCNN(nn.Module):
    """
    CNN bioacústica baseline con bloques residuales y atención temporal.

    Arquitectura:
        Stem (Conv 7×7, stride 2) →
        Bloque 1 (64 ch)  →
        Bloque 2 (128 ch, stride 2) →
        Bloque 3 (256 ch, stride 2) →
        Bloque 4 (512 ch, stride 2) →
        AttentionPool →
        FC head (512 → 256 → n_classes)

    Args:
        n_classes:    Número de clases de salida (especies o grupos).
        in_channels:  Canales de entrada (1 = Mel mono, 3 = Mel+chroma+contrast).
        dropout_rate: Dropout en el head de clasificación.

    Parámetros aproximados: ~4.2 M (n_classes=50)
    """

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 1,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.n_classes = n_classes

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── Bloques residuales ────────────────────────────────────────────────
        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        # ── Pooling con atención ──────────────────────────────────────────────
        self.attn_pool = AttentionPool(512)

        # ── Clasificador ─────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes),
        )

        # Inicialización de pesos
        self._init_weights()

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape (B, C, n_mels, T) — espectrograma Mel normalizado.
        Returns:
            logits: Tensor shape (B, n_classes) — logits sin softmax.
        """
        x = self.stem(x)          # (B, 64, H/4, W/4)
        x = self.layer1(x)        # (B, 64,  H/4,  W/4)
        x = self.layer2(x)        # (B, 128, H/8,  W/8)
        x = self.layer3(x)        # (B, 256, H/16, W/16)
        x = self.layer4(x)        # (B, 512, H/32, W/32)
        x = self.attn_pool(x)     # (B, 512)
        return self.classifier(x) # (B, n_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilidades softmax sobre las clases."""
        return F.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET PARA ESPECTROGRAMAS
# ─────────────────────────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """
    Dataset PyTorch para espectrogramas Mel guardados como .npy.

    Directorio esperado:
        data/spectrograms/
            ├── class_0/  ← nombre de especie o grupo
            │   ├── 0001.npy
            │   └── ...
            └── class_N/
                └── ...

    Args:
        root_dir:    Directorio raíz con subdirectorios por clase.
        transform:   Transformaciones adicionales (data augmentation).
        target_size: (height, width) para redimensionar el espectrograma.
        normalize:   Si True, normaliza a media=0, std=1 por canal.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform=None,
        target_size: Tuple[int, int] = (128, 128),
        normalize: bool = True,
    ):
        self.root_dir    = Path(root_dir)
        self.transform   = transform
        self.target_size = target_size
        self.normalize   = normalize

        # Construir mapeo class → int
        self.classes: List[str] = sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        # Listar todos los archivos con su etiqueta
        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for fp in cls_dir.glob("*.npy"):
                self.samples.append((fp, self.class_to_idx[cls]))

        if not self.samples:
            raise ValueError(f"No se encontraron archivos .npy en: {root_dir}")

        logger.info(f"Dataset cargado: {len(self.samples)} muestras, {len(self.classes)} clases")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fp, label = self.samples[idx]
        spec = np.load(fp).astype(np.float32)  # (n_mels, T) o (C, n_mels, T)

        # Asegurar 2D: (n_mels, T)
        if spec.ndim == 1:
            spec = spec[np.newaxis, :]
        if spec.ndim == 2:
            spec = spec[np.newaxis, :]         # → (1, n_mels, T)

        # Redimensionar con interpolación bilineal
        spec_t = torch.from_numpy(spec)
        spec_t = F.interpolate(
            spec_t.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        # Normalización por canal
        if self.normalize:
            for c in range(spec_t.shape[0]):
                mean = spec_t[c].mean()
                std  = spec_t[c].std() + 1e-8
                spec_t[c] = (spec_t[c] - mean) / std

        if self.transform:
            spec_t = self.transform(spec_t)

        return spec_t, label

    def get_class_weights(self) -> torch.Tensor:
        """Computa pesos inversamente proporcionales a la frecuencia de clase."""
        counts = torch.zeros(len(self.classes))
        for _, lbl in self.samples:
            counts[lbl] += 1
        weights = 1.0 / (counts + 1e-8)
        return weights / weights.sum() * len(self.classes)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA AUGMENTATION BIOACÚSTICO
# ─────────────────────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
    """
    SpecAugment para bioacústica (Park et al., 2019 adaptado).
    Aplica:
      - Frequency masking: enmascara bandas de frecuencia aleatorias.
      - Time masking:      enmascara segmentos temporales aleatorios.
      - Time warping:      distorsión temporal (opcional).

    Args:
        freq_mask_param: Máximo de bandas Mel a enmascarar.
        time_mask_param: Máximo de frames de tiempo a enmascarar.
        n_freq_masks:    Número de máscaras de frecuencia.
        n_time_masks:    Número de máscaras temporales.
    """

    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 30,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks    = n_freq_masks
        self.n_time_masks    = n_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Tensor (C, n_mels, T)
        Returns:
            spec augmentado in-place
        """
        _, n_mels, T = spec.shape
        spec = spec.clone()

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(n_mels - f, 1), (1,)).item()
            spec[:, f0:f0 + f, :] = 0.0

        # Time masking
        for _ in range(self.n_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            t0 = torch.randint(0, max(T - t, 1), (1,)).item()
            spec[:, :, t0:t0 + t] = 0.0

        return spec


class MixUp(nn.Module):
    """
    MixUp augmentation para bioacústica (Zhang et al., 2018).
    λ ~ Beta(α, α), mezcla pares de espectrogramas y sus etiquetas.
    Se aplica durante el training loop, no en el Dataset.
    """

    def __init__(self, alpha: float = 0.4):
        super().__init__()
        self.alpha = alpha

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            (x_mixed, y_a, y_b, lambda)
        """
        lam = float(np.random.beta(self.alpha, self.alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        x_mixed = lam * x + (1 - lam) * x[idx]
        return x_mixed, y, y[idx], lam


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Entrenador estándar para BioAcousticCNN.
    Soporta: class weights, MixUp, SpecAugment, early stopping, checkpoint.

    Ejemplo de uso:
        model   = BioAcousticCNN(n_classes=40)
        trainer = Trainer(model, train_dl, val_dl, "models/trained/")
        history = trainer.fit(epochs=50)
    """

    def __init__(
        self,
        model: BioAcousticCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str | Path = "models/trained",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_mixup: bool = True,
        use_specaugment: bool = True,
        patience: int = 10,
        device: Optional[str] = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.output_dir   = Path(output_dir)
        self.patience     = patience
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dispositivo
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Trainer: device={self.device}, params={model.count_parameters():,}")

        # Augmentación
        self.mixup       = MixUp(alpha=0.4) if use_mixup else None
        self.specaugment = SpecAugment()    if use_specaugment else None

        # Optimizador + scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Paso de entrenamiento ─────────────────────────────────────────────────

    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            # SpecAugment
            if self.specaugment:
                x = self.specaugment(x)

            # MixUp
            if self.mixup and torch.rand(1).item() > 0.5:
                x, y_a, y_b, lam = self.mixup(x, y)
                logits = self.model(x)
                loss   = lam * self.criterion(logits, y_a) + \
                         (1 - lam) * self.criterion(logits, y_b)
            else:
                logits = self.model(x)
                loss   = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += x.size(0)

        return total_loss / total, correct / total

    # ── Paso de validación ────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in self.val_loader:
            x, y   = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss   = self.criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += x.size(0)

        return total_loss / total, correct / total

    # ── Loop principal ────────────────────────────────────────────────────────

    def fit(self, epochs: int = 50) -> dict:
        """
        Entrena el modelo con early stopping basado en val_loss.

        Returns:
            history dict con listas de métricas por época.
        """
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }
        best_val_loss   = float("inf")
        patience_counter = 0
        best_path       = self.output_dir / "best_model.pt"

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._train_epoch()
            vl_loss, vl_acc = self._val_epoch()
            self.scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)

            lr_now = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} | lr={lr_now:.2e}"
            )

            # Checkpoint del mejor modelo
            if vl_loss < best_val_loss:
                best_val_loss    = vl_loss
                patience_counter = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "val_loss":    vl_loss,
                    "val_acc":     vl_acc,
                    "n_classes":   self.model.n_classes,
                    "history":     history,
                }, best_path)
                logger.info(f"  ✓ Checkpoint guardado → {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping en epoch {epoch} (patience={self.patience})")
                    break

        # Guardar historial
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: BioAcousticCNN,
    loader: DataLoader,
    device: str = "cpu",
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Evaluación completa sobre un DataLoader.

    Returns:
        {accuracy, f1_macro, precision_macro, recall_macro,
         confusion_matrix, per_class_f1}
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix,
    )

    model.eval()
    model.to(device)
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    results = {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class_f1":    {
            (class_names[i] if class_names else str(i)): float(per_class_f1[i])
            for i in range(len(per_class_f1))
        },
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. INFERENCIA INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str | Path, device: str = "cpu") -> BioAcousticCNN:
    """Carga un modelo desde checkpoint."""
    ck = torch.load(checkpoint_path, map_location=device)
    model = BioAcousticCNN(n_classes=ck["n_classes"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model.to(device)


def predict_single(
    model: BioAcousticCNN,
    spectrogram: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_k: int = 5,
    device: str = "cpu",
) -> List[dict]:
    """
    Inferencia sobre un único espectrograma.

    Args:
        spectrogram: Array (n_mels, T) o (1, n_mels, T).
        class_names: Lista de nombres de clase en orden.
        top_k:       Número de predicciones a retornar.

    Returns:
        Lista de {'class': str, 'probability': float, 'rank': int}
    """
    if spectrogram.ndim == 2:
        spectrogram = spectrogram[np.newaxis, :]  # (1, n_mels, T)
    if spectrogram.ndim == 3:
        spectrogram = spectrogram[np.newaxis, :]  # (1, 1, n_mels, T)

    x = torch.from_numpy(spectrogram.astype(np.float32)).to(device)

    with torch.no_grad():
        probs = model.predict_proba(x).squeeze(0).cpu().numpy()

    top_idx = np.argsort(probs)[::-1][:top_k]
    return [
        {
            "class":       class_names[i] if class_names else str(i),
            "probability": float(probs[i]),
            "rank":        rank + 1,
        }
        for rank, i in enumerate(top_idx)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN — Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ── Instanciar modelo ─────────────────────────────────────────────────────
    N_CLASSES = 10  # ajustar al dataset real
    model = BioAcousticCNN(n_classes=N_CLASSES, in_channels=1, dropout_rate=0.4)
    print(f"Parámetros entrenables: {model.count_parameters():,}")
    print(f"Arquitectura:\n{model}")

    # ── Forward pass de prueba ────────────────────────────────────────────────
    dummy_input = torch.randn(4, 1, 128, 128)  # batch=4, 1 canal, 128 Mel, 128 frames
    with torch.no_grad():
        logits = model(dummy_input)
        probs  = model.predict_proba(dummy_input)

    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape:  {probs.shape}")
    print(f"Suma probs (debe ser 1.0): {probs.sum(dim=-1)}")

    # ── SpecAugment test ──────────────────────────────────────────────────────
    augmenter = SpecAugment(freq_mask_param=20, time_mask_param=30)
    augmented = augmenter(dummy_input)
    print(f"\nSpecAugment output shape: {augmented.shape}")
    zeros_pct = (augmented == 0).float().mean().item()
    print(f"Fracción de ceros tras augment: {zeros_pct:.3f}")
