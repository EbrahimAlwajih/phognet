# PHOG-Net: An Interpretable and Efficient Hybrid CNN Framework for Medical Image Classification Using Gradient Histogram Integration

This repository contains the **PHOG-Net** model and training utilities (including ablation support), organized as a reusable Python package.

## What you get
- `phognet.models`: PHOG-Net architecture + PHOG descriptor layer
- `phognet.data`: dataloaders for MedMNIST + torchvision datasets, with optional 3/7-channel inputs
- `phognet.training`: metrics (ACC/AUC via MedMNIST evaluator) + early stopping
- `scripts/`: runnable entrypoints (run tracking + version logging)
- Automatic run lineage: every run logs `versions.json`, `pip_freeze.txt`, and config snapshot.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick run (creates a run folder and logs environment)
```bash
python scripts/train.py --config configs/train.yaml
```

## Ablation / sweep runner
```bash
python scripts/run_ablation.py --dataset pathmnist --epochs 50 --batch_size 128 --img_size 128 --n_channel 3
```

Outputs are written under `runs/<run_id>/` (checkpoints + metrics + environment snapshots).

## Citation
See `CITATION.cff`.

<<<<<<< HEAD
If you use this code in academic work, please cite the project as described in:

```
CITATION.cff
```

---

## Releases

Release artifacts and source archives are available at:

[https://github.com/EbrahimAlwajih/phognet/releases](https://github.com/EbrahimAlwajih/phognet/releases)

---
=======
## GitHub Releases
See `docs/RELEASES.md` (GitHub Releases only; no PyPI upload).
>>>>>>> parent of 692e6b0 (update readme file)
