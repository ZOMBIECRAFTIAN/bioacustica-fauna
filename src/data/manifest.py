"""
Dataset manifest utilities for scientifically defensible experiments.

The manifest links each generated spectrogram segment back to its original
audio file and metadata. Training can then split by original file/source/site
instead of randomly splitting highly correlated segments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"}
NEGATIVE_CLASSES = {
    "unknown",
    "unknown_biological",
    "rain",
    "wind",
    "human_voice",
    "traffic",
    "silence",
    "noise",
}

MANIFEST_FIELDS = [
    "segment_id",
    "spectrogram_path",
    "class_label",
    "scientific_name",
    "acoustic_group",
    "source",
    "source_url",
    "license",
    "country",
    "state",
    "site_id",
    "date",
    "recordist",
    "sample_rate",
    "duration_s",
    "original_format",
    "original_audio_path",
    "original_file_id",
    "segment_index",
    "split_group",
    "quality",
    "notes",
]


def audio_file_id(filename: str) -> str:
    """Return the same short hash used by batch_extractor.py."""
    return hashlib.sha256(Path(filename).name.encode()).hexdigest()[:8]


def parse_segment_name(path: str | Path) -> tuple[str, int | None]:
    """Extract original file id and segment index from '<hash>_s0001.npy'."""
    stem = Path(path).stem
    match = re.match(r"(?P<file_id>.+)_s(?P<idx>\d+)$", stem)
    if not match:
        return stem, None
    return match.group("file_id"), int(match.group("idx"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read metadata sidecar %s: %s", path, exc)
        return {}


def _read_class_catalog(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Read class-level metadata created by DatasetBuilder, if available."""
    manifest = _read_json(raw_dir / "dataset_manifest.json")
    catalog: dict[str, dict[str, Any]] = {}
    for row in manifest.get("classes", []):
        label = row.get("class_label")
        if label:
            catalog[label] = row
    return catalog


def _duration_to_seconds(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    text = str(value).strip()
    if not text:
        return ""
    parts = text.split(":")
    try:
        if len(parts) == 2:
            return f"{int(parts[0]) * 60 + float(parts[1]):.3f}"
        if len(parts) == 3:
            return f"{int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]):.3f}"
        return f"{float(text):.3f}"
    except ValueError:
        return text


def _probe_audio(path: Path) -> tuple[str, str]:
    """Best-effort sample-rate/duration extraction without making soundfile required."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return str(info.samplerate), f"{float(info.duration):.3f}"
    except Exception:
        return "", ""


def _first(meta: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = meta.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _site_id(meta: dict[str, Any]) -> str:
    explicit = _first(meta, "site_id", "site", "locality")
    if explicit:
        return explicit
    lat = _first(meta, "lat", "latitude", "decimalLatitude")
    lng = _first(meta, "lng", "lon", "longitude", "decimalLongitude")
    if lat and lng:
        return f"lat{lat}_lng{lng}"
    return ""


def _class_group(class_label: str, class_meta: dict[str, Any]) -> str:
    if class_label in NEGATIVE_CLASSES:
        return "negative"
    return str(class_meta.get("acoustic_group") or "unknown")


def _raw_audio_index(raw_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Index raw audio files by (class_label, original_file_id)."""
    catalog = _read_class_catalog(raw_dir)
    index: dict[tuple[str, str], dict[str, str]] = {}

    if not raw_dir.exists():
        return index

    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        class_label = class_dir.name
        class_meta = catalog.get(class_label, {})
        for audio_path in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS):
            sidecar = _read_json(audio_path.with_suffix(".json"))
            sr, duration = _probe_audio(audio_path)
            duration = duration or _duration_to_seconds(
                _first(sidecar, "duration", "duration_s", "length")
            )
            source = _first(sidecar, "source", "dataset", "publisher")
            site_id = _site_id(sidecar)
            file_id = audio_file_id(audio_path.name)
            split_group = site_id or _first(sidecar, "source_url") or f"{source}:{file_id}"

            index[(class_label, file_id)] = {
                "class_label": class_label,
                "scientific_name": _first(sidecar, "species", "scientific_name")
                or str(class_meta.get("scientific_name", "")),
                "acoustic_group": _class_group(class_label, class_meta),
                "source": source,
                "source_url": _first(sidecar, "source_url", "references", "download_url"),
                "license": _first(sidecar, "license", "license_code"),
                "country": _first(sidecar, "country"),
                "state": _first(sidecar, "state", "stateProvince"),
                "site_id": site_id,
                "date": _first(sidecar, "date", "observed_on", "event_date", "eventDate"),
                "recordist": _first(sidecar, "recordist", "author", "creator", "attribution"),
                "sample_rate": sr,
                "duration_s": duration,
                "original_format": audio_path.suffix.lower().lstrip("."),
                "original_audio_path": audio_path.as_posix(),
                "original_file_id": file_id,
                "split_group": split_group,
                "quality": _first(sidecar, "quality", "quality_grade"),
                "notes": "",
            }

    return index


def build_segment_manifest(
    raw_dir: str | Path,
    spectrogram_dir: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Build one CSV row per spectrogram segment."""
    raw_dir = Path(raw_dir)
    spectrogram_dir = Path(spectrogram_dir)
    output_csv = Path(output_csv) if output_csv else spectrogram_dir / "dataset_manifest.csv"

    raw_index = _raw_audio_index(raw_dir)
    rows: list[dict[str, str]] = []

    for class_dir in sorted(p for p in spectrogram_dir.iterdir() if p.is_dir()):
        class_label = class_dir.name
        for spec_path in sorted(class_dir.glob("*.npy")):
            file_id, segment_idx = parse_segment_name(spec_path)
            raw_meta = raw_index.get((class_label, file_id), {})
            acoustic_group = raw_meta.get(
                "acoustic_group",
                "negative" if class_label in NEGATIVE_CLASSES else "unknown",
            )
            row = dict.fromkeys(MANIFEST_FIELDS, "")
            row.update(raw_meta)
            row.update(
                {
                    "segment_id": spec_path.stem,
                    "spectrogram_path": spec_path.as_posix(),
                    "class_label": class_label,
                    "acoustic_group": acoustic_group,
                    "original_file_id": raw_meta.get("original_file_id", file_id),
                    "segment_index": "" if segment_idx is None else str(segment_idx),
                    "split_group": raw_meta.get("split_group") or f"{class_label}:{file_id}",
                }
            )
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    classes = sorted({row["class_label"] for row in rows})
    groups = sorted({row["acoustic_group"] for row in rows})
    summary = {
        "manifest_csv": output_csv.as_posix(),
        "n_segments": len(rows),
        "n_classes": len(classes),
        "classes": classes,
        "acoustic_groups": groups,
        "unmatched_segments": sum(1 for row in rows if not row["original_audio_path"]),
    }
    logger.info("Segment manifest written to %s", output_csv)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segment-level dataset_manifest.csv")
    parser.add_argument("--raw-dir", default="data/raw/multitaxon")
    parser.add_argument("--spectrogram-dir", default="data/spectrograms/multitaxon")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = build_segment_manifest(args.raw_dir, args.spectrogram_dir, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
