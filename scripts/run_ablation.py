from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn as nn

from phognet.data.dataloaders import get_dataloaders
from phognet.models.phognet import PHOGNetAblation, PHOGProcessingBlock
from phognet.training.early_stopping import DualMetricEarlyStopping
from phognet.training.metrics import calc_accuracy
from phognet.utils.env import write_pip_freeze, write_versions_json
from phognet.utils.run import ensure_run_dir, make_run_id, save_meta_json
from phognet.utils.seed import seed_everything


def get_optimizer(
    name: str, model: torch.nn.Module, lr: float, weight_decay: float, momentum: float
):
    name = name.lower()
    params = filter(lambda p: p.requires_grad, model.parameters())
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "radam":
        return torch.optim.RAdam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def train_one(
    run_dir: Path,
    dataset: str,
    bins: int,
    levels: int,
    n_channel: int,
    img_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    momentum: float,
    patience: int,
    ablation_case: str | None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, nInputPlane, num_classes, task = get_dataloaders(
        dataset, img_size=img_size, batch_size=batch_size, n_channel=n_channel
    )

    model = PHOGNetAblation(
        PHOGProcessingBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        bins=bins,
        levels=levels,
        nInputPlane=nInputPlane,
        ablation_case=ablation_case,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss() if dataset == "chestmnist" else nn.CrossEntropyLoss()
    optimizer = get_optimizer(
        optimizer_name, model, lr=lr, weight_decay=weight_decay, momentum=momentum
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    early = DualMetricEarlyStopping(patience=patience, verbose=True, path=run_dir / "best.pt")

    last_ckpt = run_dir / "last.pt"
    accumulation_steps = 4

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        for step, (inputs, labels) in enumerate(train_loader):
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

        acc_test, loss_test, auc_test = calc_accuracy(
            model,
            loader=test_loader,
            device=device,
            num_classes=num_classes,
            criterion=criterion,
            task=task,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_test,
            },
            last_ckpt,
        )

        print(
            f"[{run_dir.name}] Epoch {epoch}: acc={acc_test:.4f} loss={loss_test:.4f} auc={auc_test:.4f}"
        )

        early(acc_test, auc_test, model, epoch)
        if early.early_stop:
            print(f"[{run_dir.name}] Early stopping.")
            break

        scheduler.step()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="phog-net")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_dir", default="./runs")

    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--levels", type=int, default=1)
    p.add_argument("--dataset", default="pathmnist")
    p.add_argument("--n_channel", type=int, default=3)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=128)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", default="Adam")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--patience", type=int, default=10)

    p.add_argument("--ablation_cases", nargs="*", default=[None])
    p.add_argument("--datasets", nargs="*", default=None)
    args = p.parse_args()

    seed_everything(args.seed)

    bins_values = [args.bins]
    levels_values = [args.levels]
    datasets_values = args.datasets if args.datasets else [args.dataset]
    ablation_cases = args.ablation_cases

    combos = list(itertools.product(bins_values, levels_values, datasets_values, ablation_cases))
    for bins, levels, dataset, ablation_case in combos:
        run_id = make_run_id(project=args.project, dataset=dataset, seed=args.seed)
        suffix = f"{ablation_case}" if ablation_case else "baseline"
        run_dir = ensure_run_dir(args.run_dir, f"{run_id}_{suffix}")

        # lineage
        save_meta_json(
            run_dir,
            {
                "project": args.project,
                "dataset": dataset,
                "seed": args.seed,
                "bins": bins,
                "levels": levels,
                "n_channel": args.n_channel,
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "ablation_case": ablation_case,
            },
        )
        write_versions_json(run_dir)
        write_pip_freeze(run_dir)

        train_one(
            run_dir=run_dir,
            dataset=dataset,
            bins=bins,
            levels=levels,
            n_channel=args.n_channel,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
            momentum=args.momentum,
            patience=args.patience,
            ablation_case=ablation_case,
        )


if __name__ == "__main__":
    main()
