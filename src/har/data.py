"""Chargement des données, découpage train/validation et DataLoaders.

Responsable : DIOP Alioune — relecture : FALL Madjiguène.

Règle : `load_test()` n'est appelé qu'une seule fois, pour l'évaluation finale (Partie 5.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config


# --------------------------------------------------------------------------- chargement
def load_train() -> pd.DataFrame:
    return pd.read_csv(config.TRAIN_CSV)


def load_test() -> pd.DataFrame:
    """À N'UTILISER QU'UNE FOIS (Partie 5.2)."""
    return pd.read_csv(config.TEST_CSV)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Toutes les colonnes sauf la cible et l'identifiant du volontaire."""
    return [c for c in df.columns if c not in (config.TARGET_COL, config.SUBJECT_COL)]


def encode_labels(y: pd.Series) -> np.ndarray:
    """Activity (str) -> entier 0..5 selon l'ordre de config.CLASSES."""
    mapping = {c: i for i, c in enumerate(config.CLASSES)}
    return y.map(mapping).to_numpy(dtype=np.int64)


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_columns(df)].to_numpy(dtype=np.float32)
    y = encode_labels(df[config.TARGET_COL])
    return X, y


# --------------------------------------------------------------------------- découpages
def split_by_subject(
    df: pd.DataFrame,
    n_val_subjects: int = config.N_VAL_SUBJECTS,
    seed: int = config.SPLIT_SEED,
    val_subjects: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Réserve des volontaires ENTIERS pour la validation.

    Les fenêtres consécutives d'un même volontaire se chevauchent : un découpage par lignes
    met des fenêtres quasi identiques des deux côtés et surestime la performance.
    """
    subjects = np.sort(df[config.SUBJECT_COL].unique())
    if val_subjects is None:
        rng = np.random.default_rng(seed)
        val_subjects = sorted(rng.choice(subjects, size=n_val_subjects, replace=False).tolist())
    mask = df[config.SUBJECT_COL].isin(val_subjects)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True), list(val_subjects)


def split_random_rows(
    df: pd.DataFrame, val_frac: float = 0.2, seed: int = config.SPLIT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpage naïf par lignes (pour la comparaison demandée en 1.3)."""
    tr, va = train_test_split(
        df, test_size=val_frac, random_state=seed, stratify=df[config.TARGET_COL]
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True)


# --------------------------------------------------------------------------- mise à l'échelle
def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Ajusté sur l'apprentissage UNIQUEMENT, puis appliqué à val/test."""
    return StandardScaler().fit(X_train)


# --------------------------------------------------------------------------- PyTorch
def make_loaders(
    X_train, y_train, X_val, y_val, batch_size: int = config.BATCH_SIZE, seed: int = 0
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(seed)
    tr = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train))
    va = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(y_val))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g),
        DataLoader(va, batch_size=512, shuffle=False),
    )


def prepare(df_train: pd.DataFrame, df_val: pd.DataFrame, scale: bool = True):
    """Raccourci : DataFrames -> (X_tr, y_tr, X_va, y_va, scaler)."""
    X_tr, y_tr = to_xy(df_train)
    X_va, y_va = to_xy(df_val)
    scaler = None
    if scale:
        scaler = fit_scaler(X_tr)
        X_tr = scaler.transform(X_tr).astype(np.float32)
        X_va = scaler.transform(X_va).astype(np.float32)
    return X_tr, y_tr, X_va, y_va, scaler
