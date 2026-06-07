"""
scripts/verify_install.py
-----------------------------------------------------------------------------
Verificacion de dependencias criticas de Bioacustica Fauna.
Imprime un reporte de estado y retorna exit code 0 si todo OK, 1 si hay fallos.

Uso:
    python scripts/verify_install.py
    python scripts/verify_install.py --json   (salida JSON para CI)
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Windows scientific stacks can load duplicate Intel OpenMP runtimes via deps.
# This keeps the local diagnostics from crashing before they can report status.
if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Agregar la raiz del proyecto a sys.path para que 'src' sea importable
# independientemente de desde donde se ejecute el script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Checks individuales
# -----------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    ok: bool
    version: str = ""
    detail: str = ""
    required: bool = True


def check_python() -> CheckResult:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    ok = v >= (3, 10)
    return CheckResult(
        name="Python >= 3.10",
        ok=ok,
        version=version,
        detail="" if ok else f"Se requiere Python 3.10+, tienes {version}",
    )


def check_import(
    module: str,
    display_name: str | None = None,
    version_attr: str = "__version__",
    required: bool = True,
) -> CheckResult:
    name = display_name or module
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, version_attr, "?")
        return CheckResult(name=name, ok=True, version=str(version), required=required)
    except ImportError as e:
        return CheckResult(
            name=name,
            ok=False,
            detail=str(e),
            required=required,
        )


def check_torch() -> CheckResult:
    try:
        import torch

        cuda = torch.cuda.is_available()
        device = f"CUDA {torch.version.cuda}" if cuda else "CPU only"
        return CheckResult(
            name="PyTorch",
            ok=True,
            version=torch.__version__,
            detail=device,
        )
    except ImportError as e:
        return CheckResult(name="PyTorch", ok=False, detail=str(e))


def check_librosa_api() -> CheckResult:
    """Verifica que librosa >= 0.10 con API keyword-only funcione correctamente."""
    try:
        import librosa
        import numpy as np

        y = np.zeros(22050, dtype=np.float32)
        # Esta llamada falla en librosa < 0.10 si se pasa positional
        librosa.feature.spectral_centroid(y=y, sr=22050)
        return CheckResult(
            name="librosa API (keyword-only)",
            ok=True,
            version=librosa.__version__,
        )
    except TypeError as e:
        return CheckResult(
            name="librosa API (keyword-only)",
            ok=False,
            detail=f"API incompatible: {e}",
        )
    except ImportError as e:
        return CheckResult(name="librosa API (keyword-only)", ok=False, detail=str(e))


def check_soundfile() -> CheckResult:
    try:
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        # Escribe y lee un WAV sintetico
        y = np.zeros(1000, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, y, 22050)
        y2, sr = sf.read(tmp)
        os.unlink(tmp)
        ok = len(y2) == 1000 and sr == 22050
        return CheckResult(name="soundfile I/O", ok=ok, version=sf.__version__)
    except Exception as e:
        return CheckResult(name="soundfile I/O", ok=False, detail=str(e))


def check_psql() -> CheckResult:
    try:
        result = subprocess.run(
            ["psql", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = result.stdout.strip()
        return CheckResult(
            name="PostgreSQL client (psql)",
            ok=result.returncode == 0,
            version=version,
            required=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return CheckResult(
            name="PostgreSQL client (psql)",
            ok=False,
            detail="psql no encontrado en PATH (opcional para DB local)",
            required=False,
        )


def check_docker() -> CheckResult:
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = result.stdout.strip()
        return CheckResult(
            name="Docker",
            ok=result.returncode == 0,
            version=version,
            required=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return CheckResult(
            name="Docker",
            ok=False,
            detail="docker no encontrado (opcional)",
            required=False,
        )


def check_git() -> CheckResult:
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return CheckResult(
            name="Git",
            ok=result.returncode == 0,
            version=result.stdout.strip(),
            required=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return CheckResult(name="Git", ok=False, detail="git no encontrado", required=False)


def check_project_imports() -> CheckResult:
    """Verifica que los modulos del proyecto sean importables."""
    modules_to_test = [
        "src.audio_processing.preprocessor",
        "src.monitoring.acoustic_indices",
        "src.monitoring.soundscape_analyzer",
    ]
    failed = []
    for mod in modules_to_test:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            failed.append(f"{mod}: {e}")
        except Exception as e:
            failed.append(f"{mod}: {type(e).__name__}: {e}")

    if failed:
        return CheckResult(
            name="Project modules importable",
            ok=False,
            detail="; ".join(failed),
        )
    return CheckResult(
        name="Project modules importable",
        ok=True,
        version=f"{len(modules_to_test)} modules",
    )


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


def run_all_checks() -> list[CheckResult]:
    checks = [
        check_python(),
        check_torch(),
        check_import("librosa", "librosa"),
        check_librosa_api(),
        check_soundfile(),
        check_import("numpy", "numpy"),
        check_import("scipy", "scipy"),
        check_import("sklearn", "scikit-learn"),
        check_import("pandas", "pandas"),
        check_import("fastapi", "fastapi"),
        check_import("sqlalchemy", "sqlalchemy"),
        check_import("alembic", "alembic"),
        check_import("pydantic", "pydantic"),
        check_import("mlflow", "mlflow"),
        check_import("loguru", "loguru"),
        check_import("tqdm", "tqdm"),
        check_import("noisereduce", "noisereduce", required=True),
        check_import("timm", "timm"),
        # Optional
        check_import("pyaudio", "pyaudio", required=False),
        check_import("onnx", "onnx", required=False),
        check_import("onnxruntime", "onnxruntime", required=False),
        check_psql(),
        check_docker(),
        check_git(),
        check_project_imports(),
    ]
    return checks


def print_report(checks: list[CheckResult]) -> int:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Disable colors on Windows without ANSI support
    if platform.system() == "Windows":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            GREEN = RED = YELLOW = BOLD = RESET = ""

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Bioacustica Fauna -- Verificacion de Instalacion{RESET}")
    print(f"{BOLD}  Python {sys.version}{RESET}")
    print(f"{BOLD}  OS: {platform.system()} {platform.release()} ({platform.machine()}){RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    n_ok = n_fail = n_warn = 0

    for c in checks:
        if c.ok:
            icon = f"{GREEN}[OK]{RESET}"
            ver = f"  v{c.version}" if c.version else ""
            detail = f"  ({c.detail})" if c.detail else ""
            print(f"  {icon}  {c.name}{ver}{detail}")
            n_ok += 1
        elif not c.required:
            icon = f"{YELLOW}[--]{RESET}"
            detail = f"  {c.detail}" if c.detail else ""
            print(f"  {icon}  {c.name} (opcional){detail}")
            n_warn += 1
        else:
            icon = f"{RED}[FAIL]{RESET}"
            detail = f"  => {c.detail}" if c.detail else ""
            print(f"  {icon}  {c.name}{detail}")
            n_fail += 1

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(
        f"  Resultado: {GREEN}{n_ok} OK{RESET}  |  {RED}{n_fail} FAIL{RESET}  |  {YELLOW}{n_warn} opcional{RESET}"
    )
    print(f"{BOLD}{'='*60}{RESET}\n")

    if n_fail > 0:
        print(f"{RED}  Instalacion INCOMPLETA. Revisa los items FAIL.{RESET}\n")
        return 1
    else:
        print(f"{GREEN}  Instalacion COMPLETA. El sistema esta listo.{RESET}\n")
        return 0


def print_json(checks: list[CheckResult]) -> int:
    data = {
        "python": sys.version,
        "os": f"{platform.system()} {platform.release()}",
        "checks": [
            {
                "name": c.name,
                "ok": c.ok,
                "version": c.version,
                "detail": c.detail,
                "required": c.required,
            }
            for c in checks
        ],
        "summary": {
            "ok": sum(1 for c in checks if c.ok),
            "fail": sum(1 for c in checks if not c.ok and c.required),
            "optional_missing": sum(1 for c in checks if not c.ok and not c.required),
        },
    }
    print(json.dumps(data, indent=2))
    return 0 if data["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    use_json = "--json" in sys.argv
    checks = run_all_checks()
    if use_json:
        exit_code = print_json(checks)
    else:
        exit_code = print_report(checks)
    sys.exit(exit_code)
