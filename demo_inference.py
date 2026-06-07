"""
demo_inference.py -- Bioacustica Fauna
==============================================================
Minimal end-to-end demo: WAV file → bioacoustic preprocessing → mel spectrogram
→ model inference → top-5 predictions.

Works with random weights if no checkpoint is provided, to verify the pipeline.

Usage
-----
# No checkpoint — random weights (pipeline smoke test)
python demo_inference.py --audio path/to/recording.wav

# With a trained checkpoint
python demo_inference.py \\
    --audio     path/to/recording.wav \\
    --checkpoint models/trained/best_efficientnet.pt \\
    --model     efficientnet \\
    --preset    mammals

Available presets: bats | frogs | insects | mammals | reptiles
Available models:  cnn | efficientnet | panns
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# ── Try importing audio backend ───────────────────────────────────────────────
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# ── Taxon-specific preprocessing presets ─────────────────────────────────────
PRESETS: dict[str, dict] = {
    "bats": {
        "sample_rate": 192_000,
        "freq_min": 10_000,
        "freq_max": 96_000,
        "n_mels": 128,
        "fft_size": 1024,
        "hop_length": 256,
        "segment_duration": 0.5,   # bats: short echolocation pulses
        "description": "Ultrasonic — echolocation pulses 10–96 kHz (requires 192 kHz recorder)",
    },
    "frogs": {
        "sample_rate": 22_050,
        "freq_min": 100,
        "freq_max": 10_000,
        "n_mels": 128,
        "fft_size": 2048,
        "hop_length": 512,
        "segment_duration": 3.0,
        "description": "Advertisement calls — 100 Hz – 10 kHz",
    },
    "insects": {
        "sample_rate": 44_100,
        "freq_min": 200,
        "freq_max": 20_000,
        "n_mels": 128,
        "fft_size": 1024,
        "hop_length": 256,
        "segment_duration": 2.0,
        "description": "Stridulation — 200 Hz – 20 kHz",
    },
    "mammals": {
        "sample_rate": 44_100,
        "freq_min": 20,
        "freq_max": 20_000,
        "n_mels": 128,
        "fft_size": 2048,
        "hop_length": 512,
        "segment_duration": 3.0,
        "description": "Vocalizations / contact calls — 20 Hz – 20 kHz",
    },
    "reptiles": {
        "sample_rate": 22_050,
        "freq_min": 100,
        "freq_max": 5_000,
        "n_mels": 64,
        "fft_size": 2048,
        "hop_length": 512,
        "segment_duration": 3.0,
        "description": "Crocodilian rumbles / gecko calls — 100 Hz – 5 kHz",
    },
}

# ── Placeholder class names (replace with real taxonomy list) ─────────────────
PLACEHOLDER_CLASSES = [
    "Leopardus pardalis",
    "Nasua nasua",
    "Odocoileus virginianus",
    "Dasyprocta punctata",
    "Tamandua mexicana",
    "Procyon lotor",
    "Pecari tajacu",
    "Tapirus bairdii",
    "Panthera onca",
    "Canis latrans",
    "Didelphis virginiana",
    "Sylvilagus floridanus",
    "Sciurus aureogaster",
    "Myrmecophaga tridactyla",
    "Mazama temama",
    "Puma concolor",
    "Alouatta palliata",
]


# ── Lightweight model stubs ───────────────────────────────────────────────────
class _BioAcousticCNN(torch.nn.Module):
    """
    Stub of BioAcousticCNN with residual + attention blocks (~2M params).
    Replace with the real model import:
        from src.models.cnn_baseline import BioAcousticCNN
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((8, 8)),
            torch.nn.Flatten(),
        )
        self.head = torch.nn.Linear(64 * 8 * 8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class _EfficientNetBioAcoustic(torch.nn.Module):
    """
    Stub using timm EfficientNet-B0. Replace with:
        from src.models.efficientnet_classifier import EfficientNetBioAcoustic
    """

    def __init__(self, n_classes: int):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=False,
                in_chans=1,
                num_classes=n_classes,
            )
        except ImportError:
            warnings.warn("timm not installed — falling back to CNN stub")
            self.backbone = _BioAcousticCNN(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _PANNSCNN14BioAcoustic(torch.nn.Module):
    """
    Stub for PANNs-CNN14. In production:
        from src.models.panns_classifier import PANNSCNN14BioAcoustic
    Note: proper PANNs requires variable-length time axis; this stub uses AdaptivePool.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
        )
        self.head = torch.nn.Linear(128 * 4 * 4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


MODEL_REGISTRY = {
    "cnn": _BioAcousticCNN,
    "efficientnet": _EfficientNetBioAcoustic,
    "panns": _PANNSCNN14BioAcoustic,
}


# ── Audio loading + preprocessing ────────────────────────────────────────────
def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load WAV/FLAC, resample if needed. Returns (waveform_float32, sr)."""
    if not AUDIO_AVAILABLE:
        raise RuntimeError("librosa / soundfile not installed. Run: pip install librosa soundfile")
    waveform, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return waveform.astype(np.float32), sr


def segment_waveform(
    waveform: np.ndarray, sr: int, segment_duration: float, overlap: float = 0.5
) -> list[np.ndarray]:
    """Split waveform into overlapping fixed-length segments."""
    seg_len = int(segment_duration * sr)
    hop = int(seg_len * (1.0 - overlap))
    segments = []
    start = 0
    while start + seg_len <= len(waveform):
        segments.append(waveform[start : start + seg_len])
        start += hop
    if not segments:
        # Pad short recordings to one segment
        pad_len = seg_len - len(waveform)
        segments.append(np.pad(waveform, (0, pad_len), mode="constant"))
    return segments


def to_mel_spectrogram(
    segment: np.ndarray,
    sr: int,
    preset: dict,
    target_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Compute log-mel spectrogram from a waveform segment and resize."""
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_mels=preset["n_mels"],
        fmin=preset["freq_min"],
        fmax=min(preset["freq_max"], sr // 2),
        n_fft=preset["fft_size"],
        hop_length=preset["hop_length"],
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Resize to fixed target_size with bilinear interpolation
    t = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0)       # (1,1,H,W)
    t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()  # (1, H, W)


# ── Model loading ─────────────────────────────────────────────────────────────
def build_model(
    model_name: str,
    n_classes: int,
    checkpoint: Optional[Path],
    device: torch.device,
) -> torch.nn.Module:
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}")

    model = cls(n_classes).to(device)

    if checkpoint:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=device)
        # Support both raw state_dicts and dicts with a 'model_state_dict' key
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Checkpoint loaded: {checkpoint}")
    else:
        print("[WARN] No checkpoint provided — using random weights (pipeline smoke test only)")

    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(
    spectrograms: list[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """
    Run batch inference. Returns averaged softmax probabilities over all segments.
    Shape: (n_classes,)
    """
    batch = torch.tensor(
        np.stack(spectrograms, axis=0), dtype=torch.float32
    ).to(device)  # (N, 1, H, W)

    with torch.no_grad():
        logits = model(batch)             # (N, n_classes)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs.mean(axis=0)             # average across segments


# ── Result display ────────────────────────────────────────────────────────────
def print_results(
    averaged_probs: np.ndarray,
    class_names: list[str],
    top_k: int,
    audio_path: Path,
    preset_name: str,
    n_segments: int,
    elapsed_ms: float,
    random_weights: bool,
) -> None:
    top_indices = np.argsort(averaged_probs)[::-1][:top_k]

    print(f"\n[INFO] Audio: {audio_path.name}  |  segments: {n_segments}  |  preset: {preset_name}")
    print(f"[INFO] Inference complete — {elapsed_ms:.0f} ms")
    if random_weights:
        print("[WARN] *** RANDOM WEIGHTS — predictions are meaningless (pipeline test only) ***")

    print(f"\nTop-{top_k} Predictions (aggregated, {n_segments} segments):")
    print(f"  {'Rank':<6}{'Species':<32}{'Confidence':>12}")
    print("  " + "─" * 52)
    for rank, idx in enumerate(top_indices, 1):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        print(f"  #{rank:<5}{name:<32}{averaged_probs[idx]:>12.4f}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BioAcoustics demo inference — WAV → top-5 species predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio", type=Path, required=True, help="Input WAV/FLAC file")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to .pt model checkpoint (optional — random weights if omitted)"
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY), default="efficientnet",
        help="Model architecture (default: efficientnet)"
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS), default="mammals",
        help="Taxon-specific preprocessing preset (default: mammals)"
    )
    parser.add_argument(
        "--class-names", type=Path, default=None,
        help="Path to a .txt file with one class name per line (optional)"
    )
    parser.add_argument(
        "--n-classes", type=int, default=len(PLACEHOLDER_CLASSES),
        help=f"Number of output classes (default: {len(PLACEHOLDER_CLASSES)})"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top predictions to show (default: 5)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device: cpu | cuda | mps (default: cpu)"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5,
        help="Segment overlap fraction 0.0–0.9 (default: 0.5)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not args.audio.exists():
        print(f"[ERROR] Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    preset = PRESETS[args.preset]
    print(f"[INFO] Preset '{args.preset}': {preset['description']}")
    print(f"[INFO] Model:   {args.model}  |  device: {args.device}  |  n_classes: {args.n_classes}")

    # ── Class names ───────────────────────────────────────────────────────────
    if args.class_names and args.class_names.exists():
        class_names = args.class_names.read_text().strip().splitlines()
        print(f"[INFO] Loaded {len(class_names)} class names from {args.class_names}")
    else:
        class_names = PLACEHOLDER_CLASSES[:args.n_classes]
        while len(class_names) < args.n_classes:
            class_names.append(f"class_{len(class_names)}")
        if not args.class_names:
            print(f"[INFO] Using {len(class_names)} placeholder class names")

    # ── Load audio ────────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading audio: {args.audio}")
    waveform, sr = load_audio(args.audio, preset["sample_rate"])
    duration = len(waveform) / sr
    print(f"[INFO] Duration: {duration:.2f}s  |  sample rate: {sr} Hz")

    # ── Segment ───────────────────────────────────────────────────────────────
    segments = segment_waveform(waveform, sr, preset["segment_duration"], args.overlap)
    print(f"[INFO] Segments: {len(segments)} × {preset['segment_duration']}s (overlap={args.overlap:.0%})")

    # ── Mel spectrograms ──────────────────────────────────────────────────────
    print("[INFO] Extracting mel spectrograms...")
    spectrograms = [to_mel_spectrogram(seg, sr, preset) for seg in segments]

    # ── Build model ───────────────────────────────────────────────────────────
    device = torch.device(args.device)
    model = build_model(args.model, args.n_classes, args.checkpoint, device)

    # ── Inference ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    probs = run_inference(spectrograms, model, device)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ── Display ───────────────────────────────────────────────────────────────
    print_results(
        averaged_probs=probs,
        class_names=class_names,
        top_k=min(args.top_k, args.n_classes),
        audio_path=args.audio,
        preset_name=args.preset,
        n_segments=len(segments),
        elapsed_ms=elapsed_ms,
        random_weights=args.checkpoint is None,
    )


if __name__ == "__main__":
    main()
