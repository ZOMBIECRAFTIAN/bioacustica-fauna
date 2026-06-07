"""
tests/test_models.py
─────────────────────────────────────────────────────────────────────────────
Tests unitarios para los modelos de clasificación bioacústica:
  - BioAcousticCNN (cnn_baseline.py)
  - EfficientNetBioAcoustic (efficientnet_classifier.py)
  - PANNSCNN14BioAcoustic (panns_classifier.py)

Ejecutar:
    pytest tests/test_models.py -v
    pytest tests/test_models.py -v -k "CNN"       # solo tests CNN
    pytest tests/test_models.py -v -k "EfficientNet"

Autor: Ian
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from src.models.cnn_baseline import (
    AttentionPool,
    BioAcousticCNN,
    ConvBNReLU,
    MixUp,
    ResidualBlock,
    SpecAugment,
    load_model,
    predict_single,
)
from src.models.panns_classifier import (
    ConvBlock,
    PANNSCNN14BioAcoustic,
    model_comparison_table,
)

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

N_CLASSES = 10
BATCH = 4
N_MELS = 128
TIME_STEPS = 128


@pytest.fixture
def dummy_mel():
    """Batch de espectrogramas Mel (B, 1, n_mels, T)."""
    return torch.randn(BATCH, 1, N_MELS, TIME_STEPS)


@pytest.fixture
def dummy_labels():
    return torch.randint(0, N_CLASSES, (BATCH,))


@pytest.fixture
def cnn_model():
    return BioAcousticCNN(n_classes=N_CLASSES, in_channels=1, dropout_rate=0.0)


@pytest.fixture
def panns_model():
    # PANNs bn0=BatchNorm2d(64) requires mel_bins=64 (PANNs original architecture)
    return PANNSCNN14BioAcoustic(n_classes=N_CLASSES, freeze_backbone=True)


@pytest.fixture
def panns_dummy_mel():
    """PANNs format: (B, 1, T, 64) -- tiempo en dim 2, mel en dim 3."""
    import numpy as np

    rng = np.random.default_rng(seed=1)
    arr = rng.uniform(-80, 0, (BATCH, 1, 64, TIME_STEPS)).astype(np.float32)
    return torch.from_numpy(arr)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BLOQUES ARQUITECTÓNICOS CNN
# ─────────────────────────────────────────────────────────────────────────────


class TestConvBNReLU:
    def test_output_shape(self):
        block = ConvBNReLU(1, 64)
        x = torch.randn(2, 1, 64, 64)
        out = block(x)
        assert out.shape == (2, 64, 64, 64)

    def test_output_nonnegative(self):
        """ReLU debe producir valores ≥ 0."""
        block = ConvBNReLU(1, 32)
        x = torch.randn(2, 1, 32, 32)
        out = block(x)
        assert out.min() >= 0


class TestResidualBlock:
    def test_same_channels_shortcut(self):
        block = ResidualBlock(64, 64, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_different_channels_projection(self):
        block = ResidualBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)

    def test_gradient_flows(self):
        block = ResidualBlock(32, 64, stride=1)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        loss = block(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestAttentionPool:
    def test_output_shape(self):
        pool = AttentionPool(512)
        x = torch.randn(4, 512, 4, 4)
        out = pool(x)
        assert out.shape == (4, 512)

    def test_weights_sum_to_one(self):
        pool = AttentionPool(64)
        x = torch.randn(2, 64, 8, 8)
        weights = torch.softmax(pool.attn(x).flatten(2), dim=-1)
        sums = weights.sum(dim=-1).squeeze()
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. BIOACOUSTIC CNN — FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────


class TestBioAcousticCNN:
    def test_output_shape(self, cnn_model, dummy_mel):
        with torch.no_grad():
            out = cnn_model(dummy_mel)
        assert out.shape == (BATCH, N_CLASSES)

    def test_predict_proba_sums_to_one(self, cnn_model, dummy_mel):
        with torch.no_grad():
            probs = cnn_model.predict_proba(dummy_mel)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_predict_proba_range(self, cnn_model, dummy_mel):
        with torch.no_grad():
            probs = cnn_model.predict_proba(dummy_mel)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_count_parameters_positive(self, cnn_model):
        assert cnn_model.count_parameters() > 0

    def test_count_parameters_reasonable(self, cnn_model):
        # CNN baseline debe tener entre 1M y 20M params
        n = cnn_model.count_parameters()
        assert 1_000_000 <= n <= 20_000_000

    def test_gradient_flows_through_model(self, cnn_model, dummy_mel, dummy_labels):
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)
        logits = cnn_model(dummy_mel)
        loss = nn.CrossEntropyLoss()(logits, dummy_labels)
        loss.backward()
        # Al menos una capa debe tener gradientes
        grads = [p.grad.abs().sum().item() for p in cnn_model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert any(g > 0 for g in grads)

    def test_multichannel_input(self):
        model = BioAcousticCNN(n_classes=5, in_channels=3)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 5)

    def test_different_n_classes(self):
        for n in [2, 10, 50, 200]:
            model = BioAcousticCNN(n_classes=n)
            x = torch.randn(2, 1, 64, 64)
            out = model(x)
            assert out.shape == (2, n)

    def test_eval_vs_train_deterministic(self, cnn_model, dummy_mel):
        """En modo eval, el mismo input debe producir el mismo output."""
        cnn_model.eval()
        with torch.no_grad():
            out1 = cnn_model(dummy_mel)
            out2 = cnn_model(dummy_mel)
        assert torch.allclose(out1, out2)

    def test_batch_size_one(self, cnn_model):
        cnn_model.eval()
        x = torch.randn(1, 1, 128, 128)
        out = cnn_model(x)
        assert out.shape == (1, N_CLASSES)

    def test_save_and_load(self, cnn_model, dummy_mel):
        """Verifica que guardar y cargar el modelo produce el mismo output."""
        cnn_model.eval()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        torch.save(
            {
                "model_state": cnn_model.state_dict(),
                "n_classes": cnn_model.n_classes,
            },
            ckpt_path,
        )

        loaded = load_model(ckpt_path, device="cpu")
        loaded.eval()

        with torch.no_grad():
            out_orig = cnn_model(dummy_mel)
            out_loaded = loaded(dummy_mel)

        assert torch.allclose(out_orig, out_loaded, atol=1e-5)

        import os

        os.unlink(ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────


class TestSpecAugment:
    def test_output_shape_preserved(self, dummy_mel):
        aug = SpecAugment()
        out = aug(dummy_mel[0])
        assert out.shape == dummy_mel[0].shape  # (1, N_MELS, T) -- 3D input

    def test_creates_zeros(self):
        aug = SpecAugment(freq_mask_param=30, time_mask_param=30, n_freq_masks=3, n_time_masks=3)
        x = torch.ones(1, 128, 128)  # 3D: (C, n_mels, T) -- formato esperado
        out = aug(x)
        # Debe haber ceros (máscaras aplicadas)
        assert (out == 0).any()

    def test_does_not_modify_original(self, dummy_mel):
        aug = SpecAugment()
        original = dummy_mel.clone()
        _ = aug(dummy_mel[0])
        assert torch.allclose(dummy_mel, original)

    def test_freq_mask_only_frequency_axis(self):
        aug = SpecAugment(freq_mask_param=20, time_mask_param=0, n_freq_masks=2, n_time_masks=0)
        x = torch.ones(1, 128, 128)  # 3D: (C, n_mels, T)
        out = aug(x)
        # Los ceros deben estar en filas (frecuencia), no en columnas (tiempo)
        zero_rows = (out[0].sum(dim=1) == 0).sum().item()
        assert zero_rows > 0

    def test_time_mask_only_time_axis(self):
        aug = SpecAugment(freq_mask_param=0, time_mask_param=20, n_freq_masks=0, n_time_masks=2)
        x = torch.ones(1, 128, 128)  # 3D: (C, n_mels, T)
        out = aug(x)
        zero_cols = (out[0].sum(dim=0) == 0).sum().item()
        assert zero_cols > 0


class TestMixUp:
    def test_output_shapes(self, dummy_mel, dummy_labels):
        mixup = MixUp(alpha=0.4)
        x_mixed, ya, yb, lam = mixup(dummy_mel, dummy_labels)
        assert x_mixed.shape == dummy_mel.shape
        assert ya.shape == dummy_labels.shape
        assert yb.shape == dummy_labels.shape
        assert 0.0 <= lam <= 1.0

    def test_lambda_in_range(self, dummy_mel, dummy_labels):
        mixup = MixUp(alpha=1.0)
        for _ in range(20):
            _, _, _, lam = mixup(dummy_mel, dummy_labels)
            assert 0.0 <= lam <= 1.0

    def test_mixed_output_bounded(self, dummy_mel, dummy_labels):
        """La mezcla lineal nunca debe superar los extremos de los inputs."""
        x = torch.rand(4, 1, 64, 64)  # valores en [0, 1]
        y = torch.zeros(4, dtype=torch.long)
        mixup = MixUp(alpha=0.5)
        x_mixed, *_ = mixup(x, y)
        assert x_mixed.min() >= 0.0 - 1e-5
        assert x_mixed.max() <= 1.0 + 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICT_SINGLE
# ─────────────────────────────────────────────────────────────────────────────


class TestPredictSingle:
    def setup_method(self):
        pass  # predict_single calls model.eval() internally

    def test_returns_list_of_dicts(self, cnn_model):
        spec = np.random.randn(N_MELS, TIME_STEPS).astype(np.float32)
        preds = predict_single(cnn_model, spec, top_k=3)
        assert isinstance(preds, list)
        assert len(preds) <= 3
        for p in preds:
            assert "class" in p
            assert "probability" in p
            assert "rank" in p

    def test_probabilities_descending(self, cnn_model):
        spec = np.random.randn(N_MELS, TIME_STEPS).astype(np.float32)
        preds = predict_single(cnn_model, spec, top_k=N_CLASSES)
        probs = [p["probability"] for p in preds]
        assert probs == sorted(probs, reverse=True)

    def test_rank_ascending(self, cnn_model):
        spec = np.random.randn(N_MELS, TIME_STEPS).astype(np.float32)
        preds = predict_single(cnn_model, spec, top_k=5)
        ranks = [p["rank"] for p in preds]
        assert ranks == list(range(1, len(preds) + 1))

    def test_class_names_used(self, cnn_model):
        names = [f"especie_{i}" for i in range(N_CLASSES)]
        spec = np.random.randn(N_MELS, TIME_STEPS).astype(np.float32)
        preds = predict_single(cnn_model, spec, class_names=names, top_k=3)
        for p in preds:
            assert p["class"] in names


# ─────────────────────────────────────────────────────────────────────────────
# 5. PANNS-CNN14
# ─────────────────────────────────────────────────────────────────────────────


class TestConvBlockPANNs:
    def test_avg_pool(self):
        block = ConvBlock(1, 64)
        x = torch.randn(2, 1, 64, 64)
        out = block(x, pool_size=(2, 2), pool_type="avg")
        assert out.shape == (2, 64, 32, 32)

    def test_max_pool(self):
        block = ConvBlock(64, 128)
        x = torch.randn(2, 64, 32, 32)
        out = block(x, pool_size=(2, 2), pool_type="max")
        assert out.shape == (2, 128, 16, 16)

    def test_no_pool(self):
        block = ConvBlock(64, 64)
        x = torch.randn(2, 64, 16, 16)
        out = block(x, pool_size=(1, 1), pool_type="avg")
        assert out.shape == (2, 64, 16, 16)


class TestPANNSModel:
    def test_output_shape(self, panns_model, panns_dummy_mel):
        with torch.no_grad():
            out = panns_model(panns_dummy_mel)
        assert out.shape == (BATCH, N_CLASSES)

    def test_embedding_shape(self, panns_model, panns_dummy_mel):
        with torch.no_grad():
            emb = panns_model(panns_dummy_mel, return_embedding=True)
        assert emb.shape == (BATCH, 2048)

    def test_get_embeddings_l2_normalized(self, panns_model, panns_dummy_mel):
        with torch.no_grad():
            emb = panns_model.get_embeddings(panns_dummy_mel)
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_predict_proba_sums_to_one(self, panns_model, panns_dummy_mel):
        with torch.no_grad():
            probs = panns_model.predict_proba(panns_dummy_mel)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_freeze_backbone(self, panns_model):
        panns_model.freeze_backbone()
        for name, param in panns_model.backbone.named_parameters():
            assert not param.requires_grad, f"{name} debería estar congelado"

    def test_unfreeze_all(self, panns_model):
        panns_model.freeze_backbone()
        panns_model.unfreeze_all()
        all_trainable = all(p.requires_grad for p in panns_model.parameters())
        assert all_trainable

    def test_parameter_summary_keys(self, panns_model):
        summary = panns_model.parameter_summary()
        assert "total" in summary
        assert "trainable" in summary
        assert "frozen" in summary
        assert summary["total"] == summary["trainable"] + summary["frozen"]

    def test_panns_format_conversion(self, panns_model):
        """Verifica que la conversión de formato no pierde información."""
        x = torch.randn(2, 1, N_MELS, TIME_STEPS)
        converted = panns_model._to_panns_format(x)
        assert converted.shape == (2, 1, TIME_STEPS, N_MELS)

    def test_gradient_head_only_when_frozen(self, panns_model, panns_dummy_mel, dummy_labels):
        panns_model.freeze_backbone()
        logits = panns_model(panns_dummy_mel)
        loss = nn.CrossEntropyLoss()(logits, dummy_labels)
        loss.backward()
        # Backbone no debe tener gradientes
        for name, param in panns_model.backbone.named_parameters():
            assert param.grad is None or param.grad.abs().sum().item() == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────


class TestUtilities:
    def test_model_comparison_table_returns_string(self):
        table = model_comparison_table()
        assert isinstance(table, str)
        assert "CNN14" in table
        assert "EfficientNet" in table

    def test_model_comparison_table_has_all_models(self):
        table = model_comparison_table()
        for model_name in ["CNN Baseline", "EfficientNet", "PANNs-CNN14"]:
            assert model_name in table


# ─────────────────────────────────────────────────────────────────────────────
# 7. TESTS DE INTEGRACIÓN (CNN → Augmentation → Forward)
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegration:
    def test_specaugment_then_forward(self, cnn_model, dummy_mel, dummy_labels):
        """Pipeline: SpecAugment → forward → loss (batch > 1 para BatchNorm)."""
        aug = SpecAugment(freq_mask_param=10, time_mask_param=10)
        # Augment each sample in batch independently, then stack
        augmented = torch.stack([aug(dummy_mel[i]) for i in range(len(dummy_mel))])
        cnn_model.eval()
        with torch.no_grad():
            logits = cnn_model(augmented)
        assert logits.shape == (len(dummy_mel), N_CLASSES)
        assert not torch.isnan(logits).any()

    def test_mixup_then_forward(self, cnn_model, dummy_mel, dummy_labels):
        """Pipeline: MixUp → forward → mixed loss."""
        mixup = MixUp(alpha=0.4)
        x_mix, ya, yb, lam = mixup(dummy_mel, dummy_labels)
        logits = cnn_model(x_mix)
        loss = lam * nn.CrossEntropyLoss()(logits, ya) + (1 - lam) * nn.CrossEntropyLoss()(
            logits, yb
        )
        loss.backward()
        assert not torch.isnan(loss)

    def test_both_augmentations_pipeline(self, cnn_model, dummy_mel, dummy_labels):
        """SpecAugment + MixUp + forward sin NaN."""
        aug = SpecAugment()
        mixup = MixUp(alpha=0.3)
        # Augment full batch, then mixup
        augmented = torch.stack([aug(dummy_mel[i]) for i in range(len(dummy_mel))])
        x_mix, ya, yb, lam = mixup(augmented, dummy_labels)
        cnn_model.eval()
        with torch.no_grad():
            logits = cnn_model(x_mix)
        loss = lam * nn.CrossEntropyLoss()(logits, ya) + (1 - lam) * nn.CrossEntropyLoss()(
            logits, yb
        )
        assert not torch.isnan(loss)
        assert loss.item() > 0
