"""Interprétabilité : t-SNE / UMAP sur les entrées brutes et la dernière couche cachée (Partie 6).

Responsable : DIOP Alioune — mis au point sur le modèle V1, réutilisé tel quel dans la V2.
Utilisation dans n'importe quel notebook :

    from har import interpret
    interpret.compare_spaces(model, X_va, y_va, method="tsne", save="p6_tsne_v2")
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import plots


def hidden_activations(model, X: np.ndarray, device=None) -> np.ndarray:
    """Activations de la dernière couche cachée (model.embed)."""
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        H = model.embed(torch.as_tensor(X, dtype=torch.float32, device=device))
    return H.cpu().numpy()


def project(Z: np.ndarray, method: str = "tsne", seed: int = 0, **kwargs) -> np.ndarray:
    """Projection 2D. kwargs : perplexity (t-SNE), n_neighbors / min_dist (UMAP)."""
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed, init="pca", **kwargs).fit_transform(Z)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed, **kwargs).fit_transform(Z)
    raise ValueError(method)


def compare_spaces(model, X, y, method="tsne", seed=0, save=None, **kwargs):
    """Côte à côte : entrées brutes vs dernière couche cachée."""
    Z_in = project(X, method, seed, **kwargs)
    Z_hid = project(hidden_activations(model, X), method, seed, **kwargs)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    plots.scatter_2d(Z_in, y, f"{method.upper()} — entrées brutes", ax[0])
    plots.scatter_2d(Z_hid, y, f"{method.upper()} — dernière couche cachée", ax[1])
    fig.tight_layout(); plots._save(fig, save)
    return fig


def perplexity_grid(Z, y, perplexities=(5, 30, 50, 100), seed=0, save=None):
    """Sensibilité de la t-SNE à la perplexité (limites, P6.3)."""
    fig, axes = plt.subplots(1, len(perplexities), figsize=(5 * len(perplexities), 5))
    for ax, p in zip(axes, perplexities):
        plots.scatter_2d(project(Z, "tsne", seed, perplexity=p), y, f"perplexité {p}", ax)
    fig.tight_layout(); plots._save(fig, save)
    return fig
