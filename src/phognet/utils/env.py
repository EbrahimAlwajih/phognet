from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def collect_versions(extra_packages: list[str] | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }

    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass

    try:
        import torchvision  # type: ignore

        info["torchvision"] = torchvision.__version__
    except Exception:
        pass

    try:
        import numpy  # type: ignore

        info["numpy"] = numpy.__version__
    except Exception:
        pass

    try:
        import medmnist  # type: ignore

        info["medmnist"] = medmnist.__version__
    except Exception:
        pass

    if extra_packages:
        for pkg in extra_packages:
            try:
                info[f"pkg_{pkg}"] = metadata.version(pkg)
            except Exception:
                info[f"pkg_{pkg}"] = None

    return info


def write_versions_json(run_dir: str | Path, filename: str = "versions.json") -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    info = collect_versions()
    (run_dir / filename).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def write_pip_freeze(run_dir: str | Path, filename: str = "pip_freeze.txt") -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as e:
        freeze = f"# pip freeze failed: {e}\n"
    (run_dir / filename).write_text(freeze, encoding="utf-8")
