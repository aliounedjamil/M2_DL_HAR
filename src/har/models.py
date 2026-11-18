"""MLP configurable.

Responsable : FALL Madjiguène — relecture : DIOP Alioune.
"""
from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    """MLP : [Linear -> (BatchNorm) -> activation -> (Dropout)] x n -> Linear.

    hidden=[]           -> régression logistique (aucune couche cachée)
    hidden=[64]         -> Version 1
    hidden=[256,128,64] -> Version 2
    activation : un nom, ou une liste (un par couche cachée) pour mélanger les activations (P3.2).
    norm : None | "batch" | "layer" (LayerNorm : bonus P7).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        n_classes: int = 6,
        activation: str | list[str] = "relu",
        dropout: float = 0.0,
        norm: str | None = None,
    ):
        super().__init__()
        acts = [activation] * len(hidden) if isinstance(activation, str) else list(activation)
        assert len(acts) == len(hidden), "une activation par couche cachée"

        layers: list[nn.Module] = []
        prev = in_dim
        for h, a in zip(hidden, acts):
            layers.append(nn.Linear(prev, h))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(ACTIVATIONS[a]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.features = nn.Sequential(*layers)   # sortie = dernière couche cachée (P6)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Activations de la dernière couche cachée (pour t-SNE / UMAP)."""
        return self.features(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
