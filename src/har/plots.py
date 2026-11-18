"""Figures : courbes d'apprentissage, matrices de confusion, projections 2D.

Responsable : FALL Madjiguène — relecture : DIOP Alioune.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import config


def _save(fig, name: str | None):
    if name:
        config.FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


def learning_curves(hist: dict, title: str = "", save: str | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, lab in zip(axes, ["loss", "acc", "f1"], ["Loss", "Accuracy", "F1 macro"]):
        ax.plot(hist[f"train_{key}"], label="train")
        ax.plot(hist[f"val_{key}"], label="validation")
        ax.set_xlabel("époque"); ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(title); fig.tight_layout(); _save(fig, save)
    return fig


def compare_curves(histories: dict[str, dict], key: str = "val_f1", save: str | None = None):
    """Superpose une métrique pour plusieurs configurations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in histories.items():
        ax.plot(h[key], label=name)
    ax.set_xlabel("époque"); ax.set_ylabel(key); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); _save(fig, save)
    return fig


def confusion_heatmap(cm: np.ndarray, title: str = "", normalize: bool = True, save: str | None = None):
    import seaborn as sns
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=config.CLASSES, yticklabels=config.CLASSES, ax=ax)
    ax.set_xlabel("prédit"); ax.set_ylabel("vrai"); ax.set_title(title)
    fig.tight_layout(); _save(fig, save)
    return fig


def scatter_2d(Z: np.ndarray, y: np.ndarray, title: str = "", ax=None, save: str | None = None):
    """Nuage t-SNE / UMAP coloré par activité."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(config.CLASSES):
        m = y == i
        ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.6, label=c)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=4, fontsize=8)
    if fig is not None:
        fig.tight_layout(); _save(fig, save)
    return ax
