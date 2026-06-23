from __future__ import annotations
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


@dataclass
class RunConfig:
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.0
    amp: bool = True
    max_batches_per_epoch: Optional[int] = None


def _iter_limited(loader, max_batches: Optional[int]):
    if max_batches is None:
        yield from loader
    else:
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            yield batch


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())

    acc = accuracy_score(y_true, y_pred)

    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    rec_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    report_df = None
    if class_names is not None:
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).T

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
        "f1_micro": f1_micro,
        "report_df": report_df,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    cfg: RunConfig,
    device: torch.device,
) -> Tuple[List[float], List[float], dict, int]:
    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(cfg.epochs):
        #train
        model.train()
        running_loss, n = 0.0, 0

        for images, labels in _iter_limited(train_loader, cfg.max_batches_per_epoch):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += float(loss.detach().cpu().item()) * bs
            n += bs

        train_loss = running_loss / max(n, 1)
        train_losses.append(train_loss)

        #val loss
        model.eval()
        running_loss, n = 0.0, 0
        with torch.no_grad():
            for images, labels in _iter_limited(val_loader, cfg.max_batches_per_epoch):
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                bs = labels.size(0)
                running_loss += float(loss.detach().cpu().item()) * bs
                n += bs

        val_loss = running_loss / max(n, 1)
        val_losses.append(val_loss)

        #best epoch by DEV
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        val_metrics = evaluate_model(model, val_loader, device)
        print(
            f"Epoch {epoch+1:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} "
            f"val_f1_micro={val_metrics['f1_micro']:.4f}"
        )

    return train_losses, val_losses, best_state, best_epoch



def plot_losses(group_name: str, results: Dict[str, Dict[str, List[float]]]) -> None:
    plt.figure()
    for name, r in results.items():
        plt.plot(r["train_loss"], linestyle="--", label=f"{name} Train")
        plt.plot(r["val_loss"], label=f"{name} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train/Validation Loss | {group_name}")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

