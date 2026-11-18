"""Métriques de classification.

Responsable : FALL Madjiguène — relecture : DIOP Alioune.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import config


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def per_class_f1(y_true, y_pred) -> dict:
    scores = f1_score(y_true, y_pred, average=None, labels=range(config.N_CLASSES), zero_division=0)
    return {c: float(s) for c, s in zip(config.CLASSES, scores)}


def confusion(y_true, y_pred, normalize: str | None = None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=range(config.N_CLASSES), normalize=normalize)


def mean_std(values) -> str:
    v = np.asarray(values, dtype=float)
    std = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.4f} ± {std:.4f}"
