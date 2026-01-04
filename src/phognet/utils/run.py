from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def make_run_id(project: str, dataset: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_project = project.replace(" ", "-").lower()
    safe_dataset = dataset.replace(" ", "-").lower()
    return f"{ts}_{safe_project}_{safe_dataset}_seed{seed}"


def ensure_run_dir(base_dir: str | Path, run_id: str) -> Path:
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_yaml(
    run_dir: str | Path, config: dict[str, Any], filename: str = "config.yaml"
) -> None:
    run_dir = Path(run_dir)
    (run_dir / filename).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def save_meta_json(run_dir: str | Path, meta: dict[str, Any], filename: str = "meta.json") -> None:
    run_dir = Path(run_dir)
    (run_dir / filename).write_text(json.dumps(meta, indent=2), encoding="utf-8")
