from pathlib import Path

import numpy as np
from src.data.manifest import audio_file_id, build_segment_manifest
from src.models.train import _segment_origin_id, _split_indices


class DummyDataset:
    def __init__(self, samples, classes):
        self.samples = samples
        self.classes = classes


def test_build_segment_manifest_links_segment_to_raw_audio(tmp_path):
    raw_dir = tmp_path / "raw"
    spec_dir = tmp_path / "spectrograms"
    class_dir = raw_dir / "smilisca_baudinii"
    spec_class_dir = spec_dir / "smilisca_baudinii"
    class_dir.mkdir(parents=True)
    spec_class_dir.mkdir(parents=True)

    (raw_dir / "dataset_manifest.json").write_text(
        """
        {
          "classes": [
            {
              "class_label": "smilisca_baudinii",
              "scientific_name": "Smilisca baudinii",
              "acoustic_group": "frogs"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    audio_path = class_dir / "call_001.wav"
    audio_path.write_bytes(b"not-real-audio")
    audio_path.with_suffix(".json").write_text(
        """
        {
          "source": "xeno-canto",
          "source_url": "https://xeno-canto.org/1",
          "license": "CC-BY-NC",
          "country": "Mexico",
          "date": "2026-01-01",
          "author": "field recordist"
        }
        """,
        encoding="utf-8",
    )

    file_id = audio_file_id(audio_path.name)
    np.save(spec_class_dir / f"{file_id}_s0000.npy", np.zeros((128, 64), dtype=np.float32))

    out_csv = tmp_path / "dataset_manifest.csv"
    summary = build_segment_manifest(raw_dir, spec_dir, out_csv)
    text = out_csv.read_text(encoding="utf-8")

    assert summary["n_segments"] == 1
    assert "smilisca_baudinii" in text
    assert "frogs" in text
    assert "CC-BY-NC" in text


def test_grouped_split_keeps_original_file_out_of_multiple_splits():
    samples = []
    manifest_rows = {}
    classes = ["class_a", "class_b"]

    for label_idx, class_name in enumerate(classes):
        for file_idx in range(4):
            origin_id = f"{class_name}_file_{file_idx}"
            for seg_idx in range(2):
                path = Path(f"/dataset/{class_name}/{origin_id}_s{seg_idx:04d}.npy")
                samples.append((path, label_idx))
                manifest_rows[f"{class_name}/{path.stem}"] = {
                    "original_file_id": origin_id,
                    "split_group": origin_id,
                }

    ds = DummyDataset(samples=samples, classes=classes)
    dcfg = {
        "split_strategy": "source_file",
        "train_split": 0.5,
        "val_split": 0.25,
        "test_split": 0.25,
        "random_state": 7,
    }

    idx_train, idx_val, idx_test = _split_indices(ds, dcfg, manifest_rows)

    def origins(indices):
        return {_segment_origin_id(samples[i][0]) for i in indices}

    train_origins = origins(idx_train)
    val_origins = origins(idx_val)
    test_origins = origins(idx_test)

    assert train_origins.isdisjoint(val_origins)
    assert train_origins.isdisjoint(test_origins)
    assert val_origins.isdisjoint(test_origins)
