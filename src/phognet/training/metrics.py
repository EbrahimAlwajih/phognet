from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from medmnist.evaluator import getACC, getAUC
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def calc_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None,
    num_classes: int,
    task: str,
) -> tuple[float, float]:
    """Metrics consistent with MedMNIST evaluator."""
    auc = float(getAUC(y_true, y_pred_proba, task))
    acc = float(getACC(y_true, y_pred_proba, task))
    return auc, acc


def calc_accuracy(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    verbose: bool = False,
    num_classes: int = 10,
    criterion: nn.Module | None = None,
    task: str = "multi-label, binary-class",
) -> Tuple[float, float, float]:
    """Return (accuracy, loss, auc)."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    outputs_full = []
    labels_full = []
    epoch_loss = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Eval", disable=not verbose):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, labels)

            epoch_loss.append(float(loss.item()))
            outputs_full.append(outputs.detach())
            labels_full.append(labels.detach())

    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)

    # Probabilities for AUC/ACC in MedMNIST evaluator
    outputs_proba = outputs_full.softmax(dim=1).detach().cpu().numpy()
    labels_np = labels_full.detach().cpu().numpy()

    # Pred labels (for optional additional metrics)
    y_pred = outputs_full.argmax(dim=1).detach().cpu().numpy()

    auc, acc = calc_metrics(labels_np, outputs_proba, y_pred, num_classes=num_classes, task=task)
    loss_value = float(np.mean(epoch_loss)) if epoch_loss else 0.0
    return acc, loss_value, auc


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    criterion: nn.Module | None = None,
) -> tuple[float, float]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item())

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(1, len(data_loader))
    acc = float(accuracy_score(all_labels, all_preds))
    return acc, avg_loss
