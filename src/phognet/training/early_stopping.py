from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class DualMetricEarlyStopping:
    """Early stopping prioritizing ACC, breaking ties with AUC."""

    patience: int = 10
    min_delta: float = 0.0
    verbose: bool = False
    path: str | Path = "best.pt"

    counter: int = 0
    early_stop: bool = False
    val_acc_max: float = float("-inf")
    val_auc_max: float = float("-inf")

    def __call__(self, val_acc: float, val_auc: float, model: torch.nn.Module, epoch: int) -> None:
        improved = False

        if val_acc > self.val_acc_max + self.min_delta:
            self.val_acc_max = val_acc
            self.val_auc_max = val_auc
            improved = True
            self.counter = 0
        elif val_acc == self.val_acc_max and val_auc > self.val_auc_max + self.min_delta:
            self.val_auc_max = val_auc
            improved = True
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        if improved:
            self.save_checkpoint(val_acc, val_auc, model, epoch)

    def save_checkpoint(
        self, val_acc: float, val_auc: float, model: torch.nn.Module, epoch: int
    ) -> None:
        if self.verbose:
            print(f"Epoch {epoch}: Saving model with accuracy {val_acc:.4f} and AUC {val_auc:.4f}.")

        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": float(val_acc),
                "val_auc": float(val_auc),
            },
            path,
        )
