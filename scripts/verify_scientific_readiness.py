"""
Quick scientific-readiness checks for the master's thesis project.

This script validates repository structure and configuration choices that matter
for a defensible experiment. It does not require the dataset to be downloaded.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str, failures: list[str]) -> None:
    print(f"[FAIL] {message}")
    failures.append(message)


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def require_file(path: str, failures: list[str]) -> None:
    full = ROOT / path
    if full.exists():
        ok(path)
    else:
        fail(f"Missing file: {path}", failures)


def load_yaml(path: str, failures: list[str]) -> dict:
    full = ROOT / path
    try:
        with open(full, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        ok(f"YAML parses: {path}")
        return data
    except Exception as exc:
        fail(f"YAML parse failed: {path} ({exc})", failures)
        return {}


def check_training_config(
    path: str, failures: list[str], expected_preset: str | None = None
) -> None:
    cfg = load_yaml(path, failures)
    dataset = cfg.get("dataset", {})
    audio = cfg.get("audio", {})
    evaluation = cfg.get("evaluation", {})

    if dataset.get("split_strategy") in {"source_file", "file", "source", "site"}:
        ok(f"{path}: leakage-safe split_strategy={dataset.get('split_strategy')}")
    else:
        fail(f"{path}: split_strategy should avoid segment leakage", failures)

    if dataset.get("manifest_csv"):
        ok(f"{path}: manifest_csv configured")
    else:
        fail(f"{path}: manifest_csv missing", failures)

    if expected_preset and audio.get("preset") != expected_preset:
        fail(f"{path}: expected audio.preset={expected_preset}", failures)
    elif audio.get("preset"):
        ok(f"{path}: audio.preset={audio.get('preset')}")

    if evaluation.get("group_metrics"):
        ok(f"{path}: group_metrics enabled")
    else:
        fail(f"{path}: group_metrics should be enabled", failures)


def main() -> int:
    failures: list[str] = []

    required_files = [
        "src/data/manifest.py",
        "src/api/static/data_entry.html",
        "scripts/list_target_species.py",
        "docs/metodologia/especies_objetivo_maestria.md",
        "tests/test_manifest_and_splits.py",
        "docs/metodologia/protocolo_dataset_multitaxon.md",
        "docs/metodologia/diseno_experimental_maestria.md",
        "docs/metodologia/clases_negativas.md",
        "docs/metodologia/validacion_campo.md",
        "results/templates/maestria_multitaxon/metrics_template.csv",
        "results/templates/maestria_multitaxon/per_group_metrics_template.csv",
        "configs/train_multitaxon.yaml",
        "configs/train_multitaxon_flat.yaml",
        "configs/train_multitaxon_stage1_group.yaml",
        "configs/train_multitaxon_stage2_group_template.yaml",
        ".gitattributes",
    ]
    for path in required_files:
        require_file(path, failures)

    check_training_config("configs/train_multitaxon.yaml", failures, expected_preset="adaptive")
    check_training_config(
        "configs/train_multitaxon_flat.yaml", failures, expected_preset="multitaxon"
    )
    check_training_config(
        "configs/train_multitaxon_stage1_group.yaml", failures, expected_preset="adaptive"
    )

    root_pdf = ROOT / "docs" / "marco_teorico_bioacustica_ia.pdf"
    nested_pdf = ROOT / "docs" / "marco_teorico" / "marco_teorico_bioacustica_ia.pdf"
    if root_pdf.exists():
        fail("Duplicate marco_teorico PDF still exists in docs/", failures)
    elif nested_pdf.exists():
        ok("Only organized marco_teorico PDF remains")
    else:
        warn("Marco teorico PDF not present; acceptable if generated later")

    checked_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8", errors="ignore")
        for path in [".env.example", "docker-compose.yml", "src/api/main.py"]
        if (ROOT / path).exists()
    )
    if "models/trained/mexico_birds/best_model.pt" in checked_text:
        fail("Old mexico_birds default model path remains", failures)
    else:
        ok("Default model paths point away from mexico_birds MVP")

    if failures:
        print("\nScientific readiness: FAILED")
        for item in failures:
            print(f" - {item}")
        return 1

    print("\nScientific readiness: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
