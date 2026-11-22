"""Interprétabilité : t-SNE / UMAP sur les entrées brutes et la dernière couche cachée (Partie 6).

Responsable : DIOP Alioune — mis au point sur le modèle V1, réutilisable tel quel dans la V2 :

    from har import interpret
    interpret.compare_spaces(model, X_va, y_va, method="tsne", save="v2_tsne")
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from . import config, plots


# --------------------------------------------------------------------------- représentations
def hidden_activations(model, X: np.ndarray, device=None) -> np.ndarray:
    """Activations de la dernière couche cachée (model.embed), en mode évaluation."""
    import torch
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        H = model.embed(torch.as_tensor(X, dtype=torch.float32, device=device))
    return H.cpu().numpy()


def umap_available() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except ImportError:
        return False


def project(Z: np.ndarray, method: str = "tsne", seed: int = 0, **kwargs) -> np.ndarray:
    """Projection 2D. kwargs : perplexity (t-SNE) ; n_neighbors, min_dist (UMAP)."""
    if method == "tsne":
        from sklearn.manifold import TSNE
        kwargs.setdefault("perplexity", 30)
        return TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", **kwargs).fit_transform(Z)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed, **kwargs).fit_transform(Z)
    raise ValueError(method)


# --------------------------------------------------------------------------- figures
def compare_spaces(model, X, y, method="tsne", seed=0, save=None, **kwargs):
    """Côte à côte : entrées brutes vs dernière couche cachée. Renvoie (fig, Z_in, Z_hid)."""
    Z_in = project(X, method, seed, **kwargs)
    Z_hid = project(hidden_activations(model, X), method, seed, **kwargs)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    plots.scatter_2d(Z_in, y, f"{method.upper()} — entrées brutes ({X.shape[1]} dim.)", ax[0])
    plots.scatter_2d(Z_hid, y, f"{method.upper()} — dernière couche cachée", ax[1])
    fig.tight_layout(); plots._save(fig, save)
    return fig, Z_in, Z_hid


def perplexity_grid(Z, y, perplexities=(5, 30, 50, 100), seed=0, save=None):
    """Sensibilité de la t-SNE à la perplexité (limite n°1)."""
    fig, axes = plt.subplots(1, len(perplexities), figsize=(5 * len(perplexities), 5))
    for ax, p in zip(axes, perplexities):
        plots.scatter_2d(project(Z, "tsne", seed, perplexity=p), y, f"perplexité {p}", ax)
    fig.tight_layout(); plots._save(fig, save)
    return fig


def seed_grid(Z, y, seeds=(0, 1, 2), method="tsne", save=None, **kwargs):
    """Variabilité d'une exécution à l'autre (limite n°2)."""
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 5))
    for ax, s in zip(axes, seeds):
        plots.scatter_2d(project(Z, method, s, **kwargs), y, f"{method} — graine {s}", ax)
    fig.tight_layout(); plots._save(fig, save)
    return fig


# --------------------------------------------------------------------------- mesures quantitatives
def separation_scores(spaces: dict[str, np.ndarray], y, k: int = 10) -> pd.DataFrame:
    """Pour chaque espace : silhouette (−1…1, plus haut = classes mieux séparées) et
    pureté des k plus proches voisins (part des voisins de même classe)."""
    rows = {}
    for name, Z in spaces.items():
        rows[name] = {"silhouette": float(silhouette_score(Z, y)),
                      f"pureté {k}-NN": float(np.diag(neighbor_mixing(Z, y, k)).mean())}
    return pd.DataFrame(rows).T


def neighbor_mixing(Z, y, k: int = 10) -> np.ndarray:
    """Matrice M[i, j] = part des k plus proches voisins des points de classe i qui sont de classe j.
    Hors diagonale : chevauchement entre classes — à comparer à la matrice de confusion."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    M = np.zeros((config.N_CLASSES, config.N_CLASSES))
    for i in range(config.N_CLASSES):
        neigh = y[idx[y == i]].ravel()
        M[i] = np.bincount(neigh, minlength=config.N_CLASSES) / max(len(neigh), 1)
    return M


def mixing_vs_confusion(M: np.ndarray, cm: np.ndarray) -> pd.DataFrame:
    """Met côte à côte, pour chaque paire de classes (i ≠ j), le chevauchement dans la projection
    et le taux de confusion du modèle ; trié par confusion décroissante."""
    cmn = cm / cm.sum(axis=1, keepdims=True)
    rows = []
    for i in range(config.N_CLASSES):
        for j in range(config.N_CLASSES):
            if i != j:
                rows.append({"vraie classe": config.CLASSES[i], "confondue avec": config.CLASSES[j],
                             "taux de confusion": cmn[i, j], "chevauchement k-NN": M[i, j]})
    return pd.DataFrame(rows).sort_values("taux de confusion", ascending=False).reset_index(drop=True)


def mixing_heatmaps(M: np.ndarray, cm: np.ndarray, title_left="chevauchement k-NN (projection)", save=None):
    import seaborn as sns
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for a, mat, t in [(ax[0], M, title_left), (ax[1], cmn, "matrice de confusion (normalisée)")]:
        off = mat.copy(); np.fill_diagonal(off, np.nan)
        sns.heatmap(off, annot=True, fmt=".2f", cmap="Reds", ax=a, vmin=0,
                    xticklabels=config.CLASSES, yticklabels=config.CLASSES)
        a.set_title(t + " — hors diagonale")
    fig.tight_layout(); plots._save(fig, save)
    return fig
