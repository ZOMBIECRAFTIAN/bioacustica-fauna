"""
Print target species/classes for the master's multitaxon profile.

This script does not download data. It only summarizes the local project
configuration so the user knows what audio classes to collect.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "train_multitaxon.yaml"
MODEL_CLASSES_PATH = ROOT / "models" / "trained" / "multitaxon" / "class_names.json"


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_classes(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    classes = payload.get("classes") if isinstance(payload, dict) else payload
    return [str(item) for item in classes] if isinstance(classes, list) else []


def print_grouped(title: str, rows: list[tuple[str, str, str]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    current_group = None
    for label, scientific_name, group in sorted(rows, key=lambda row: (row[2], row[0])):
        if group != current_group:
            current_group = group
            print(f"\n[{group}]")
        print(f"  {label:36} {scientific_name}")


def load_profile_rows(profile: str) -> list[tuple[str, str, str]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.data.dataset_builder import get_targets_for_profile

    return [
        (target.class_label, target.scientific_name, target.acoustic_group)
        for target in get_targets_for_profile(profile)
    ]


def main() -> int:
    cfg = load_yaml(CONFIG_PATH)
    class_groups = cfg.get("dataset", {}).get("class_groups", {})
    negatives = cfg.get("dataset", {}).get("negative_classes", [])

    model_classes = load_model_classes(MODEL_CLASSES_PATH)
    if model_classes:
        model_rows = [
            (label, label.replace("_", " ").capitalize(), class_groups.get(label, "unknown"))
            for label in model_classes
        ]
        print_grouped("Modelo actual en disco", model_rows)
    else:
        print("\nModelo actual en disco")
        print("----------------------")
        print("No se encontro models/trained/multitaxon/class_names.json")

    profile_rows = load_profile_rows("mexico_multitaxon")
    print_grouped("Perfil objetivo mexico_multitaxon", profile_rows)

    print("\nClases negativas recomendadas")
    print("-----------------------------")
    for label in negatives:
        print(f"  {label}")

    print("\nDocumento detallado")
    print("-------------------")
    print("docs/metodologia/especies_objetivo_maestria.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
