from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from phognet.data.dataloaders import get_dataloaders
from phognet.models.phognet import PHOGNet, PHOGProcessingBlock, PHOGNetAblation
from phognet.training.early_stopping import DualMetricEarlyStopping
from phognet.training.metrics import calc_accuracy
from phognet.utils.env import write_pip_freeze, write_versions_json
from phognet.utils.run import ensure_run_dir, make_run_id, save_config_yaml, save_meta_json
from phognet.utils.seed import seed_everything


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def get_optimizer(name: str, model: torch.nn.Module, lr: float, weight_decay: float, momentum: float):
    name = name.lower()
    params = filter(lambda p: p.requires_grad, model.parameters())
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "radam":
        return torch.optim.RAdam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    project = cfg.get("project", "phog-net")
    dataset = cfg.get("dataset", "pathmnist")
    seed = int(cfg.get("seed", 42))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    run_base = cfg.get("run_dir", "./runs")

    seed_everything(seed)

    run_id = make_run_id(project=project, dataset=dataset, seed=seed)
    run_dir = ensure_run_dir(run_base, run_id)

    # Run lineage
    save_config_yaml(run_dir, cfg)
    save_meta_json(run_dir, {"project": project, "dataset": dataset, "seed": seed, "device": device, "run_id": run_id})
    write_versions_json(run_dir)
    write_pip_freeze(run_dir)

    # Data
    batch_size = int(cfg.get("batch_size", 128))
    img_size = int(cfg.get("img_size", 32))
    n_channel = int(cfg.get("n_channel", 3))
    train_loader, test_loader, nInputPlane, num_classes, task = get_dataloaders(dataset, img_size=img_size, batch_size=batch_size, n_channel=n_channel)

    # Model
    model_cfg = cfg.get("model", {})
    bins = int(model_cfg.get("bins", 20))
    levels = int(model_cfg.get("levels", 1))
    num_blocks = model_cfg.get("num_blocks", [2, 2, 2, 2])
    ablation_case = model_cfg.get("ablation_case", None)

    if ablation_case:
        model = PHOGNetAblation(PHOGProcessingBlock, num_blocks, num_classes=num_classes, bins=bins, levels=levels, nInputPlane=nInputPlane, ablation_case=ablation_case)
    else:
        model = PHOGNet(PHOGProcessingBlock, num_blocks, num_classes=num_classes, bins=bins, levels=levels, nInputPlane=nInputPlane)

    model.to(device)

    # Loss
    criterion = nn.BCEWithLogitsLoss() if dataset == "chestmnist" else nn.CrossEntropyLoss()

    # Optimizer/scheduler
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer_name = str(train_cfg.get("optimizer", "Adam"))
    momentum = float(train_cfg.get("momentum", 0.9))
    patience = int(train_cfg.get("patience", 10))
    accumulation_steps = int(train_cfg.get("accumulation_steps", 4))

    optimizer = get_optimizer(optimizer_name, model, lr=lr, weight_decay=weight_decay, momentum=momentum)
    milestones = train_cfg.get("milestones", [50, 75])
    gamma = float(train_cfg.get("gamma", 0.1))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # Early stopping
    early = DualMetricEarlyStopping(patience=patience, verbose=True, path=Path(run_dir) / "best.pt")

    last_ckpt = Path(run_dir) / "last.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                loss = criterion(outputs, labels.float()) / accumulation_steps
            else:
                loss = criterion(outputs, labels) / accumulation_steps

            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad()

        # Eval
        acc_test, loss_test, auc_test = calc_accuracy(model, loader=test_loader, device=device, num_classes=num_classes, criterion=criterion, task=task)

        # Save last checkpoint
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": loss_test},
            last_ckpt,
        )

        print(f"Epoch {epoch}: acc={acc_test:.4f} loss={loss_test:.4f} auc={auc_test:.4f}")

        early(acc_test, auc_test, model, epoch)
        if early.early_stop:
            print("Early stopping triggered.")
            break

        scheduler.step()

    print(f"[OK] Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
