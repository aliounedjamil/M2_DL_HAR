"""Boucles d'entraînement / évaluation (inspirées des TP : train_epoch, evaluate).

Responsable : DIOP Alioune — relecture : FALL Madjiguène.
"""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from .metrics import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(name: str, params, lr: float, weight_decay: float = 0.0, momentum: float = 0.9):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name in ("sgd_momentum", "momentum"):
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"optimiseur inconnu : {name}")


def train_epoch(model, loader, criterion, optimizer, device=DEVICE) -> dict:
    model.train()
    total_loss, y_true, y_pred = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        y_true.append(y.cpu()); y_pred.append(logits.argmax(1).cpu())
    y_true = torch.cat(y_true).numpy(); y_pred = torch.cat(y_pred).numpy()
    return {"loss": total_loss / len(y_true), **compute_metrics(y_true, y_pred)}


@torch.no_grad()
def evaluate(model, loader, criterion, device=DEVICE, return_preds: bool = False) -> dict:
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        y_true.append(y.cpu()); y_pred.append(logits.argmax(1).cpu())
    y_true = torch.cat(y_true).numpy(); y_pred = torch.cat(y_pred).numpy()
    out = {"loss": total_loss / len(y_true), **compute_metrics(y_true, y_pred)}
    if return_preds:
        out["y_true"], out["y_pred"] = y_true, y_pred
    return out


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs: int = 50,
    criterion=None,
    patience: int | None = None,
    monitor: str = "f1_macro",
    device=DEVICE,
    verbose: bool = False,
) -> dict:
    """Entraîne et renvoie l'historique par époque. Early stopping si `patience` est donné
    (restaure les meilleurs poids selon `monitor` en validation)."""
    criterion = criterion or nn.CrossEntropyLoss()
    model.to(device)
    hist = {k: [] for k in ["train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"]}
    best, best_state, best_epoch, wait = -np.inf, None, 0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, criterion, optimizer, device)
        va = evaluate(model, val_loader, criterion, device)
        for split, m in (("train", tr), ("val", va)):
            hist[f"{split}_loss"].append(m["loss"])
            hist[f"{split}_acc"].append(m["accuracy"])
            hist[f"{split}_f1"].append(m["f1_macro"])
        score = va[monitor] if monitor != "loss" else -va["loss"]
        if score > best:
            best, best_epoch, wait = score, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if verbose:
            print(f"ep {ep:3d} | train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} "
                  f"| val loss {va['loss']:.4f} acc {va['accuracy']:.4f} f1 {va['f1_macro']:.4f}")
        if patience is not None and wait >= patience:
            break
    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch
    hist["train_time_s"] = time.time() - t0
    return hist


def run_seeds(build_fn: Callable[[int], tuple], seeds=(0, 1, 2), **fit_kwargs) -> dict:
    """Répète l'entraînement sur plusieurs graines (Partie 2.3).

    build_fn(seed) doit renvoyer (model, train_loader, val_loader, optimizer).
    Renvoie les historiques + la moyenne ± écart-type des scores de validation finaux.
    """
    runs = []
    for s in seeds:
        set_seed(s)
        model, tl, vl, opt = build_fn(s)
        h = fit(model, tl, vl, opt, **fit_kwargs)
        i = h["best_epoch"] - 1 if fit_kwargs.get("patience") else -1
        runs.append({"seed": s, "history": h,
                     "val_acc": h["val_acc"][i], "val_f1": h["val_f1"][i]})
    summary = {}
    for k in ("val_acc", "val_f1"):
        v = np.array([r[k] for r in runs])
        summary[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return {"runs": runs, "summary": summary}


def lr_range_test(model, loader, optimizer, lr_min=1e-6, lr_max=1, n_steps=100, device=DEVICE):
    """LR range test (Smith) : augmente le LR exponentiellement, renvoie (lrs, losses)."""
    criterion = nn.CrossEntropyLoss()
    model.to(device).train()
    gamma = (lr_max / lr_min) ** (1 / n_steps)
    for g in optimizer.param_groups:
        g["lr"] = lr_min
    lrs, losses, it = [], [], iter(loader)
    for _ in range(n_steps):
        try:
            X, y = next(it)
        except StopIteration:
            it = iter(loader); X, y = next(it)
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward(); optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"]); losses.append(loss.item())
        if not np.isfinite(loss.item()) or loss.item() > 4 * min(losses):
            break
        for g in optimizer.param_groups:
            g["lr"] *= gamma
    return lrs, losses


def save_json(obj, path: str | Path) -> None:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=conv))
