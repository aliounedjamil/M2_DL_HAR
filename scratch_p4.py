import sys
sys.path.append("src")

import numpy as np
import pandas as pd

from har import config, data, metrics, training
from har.models import MLP, count_params

HIDDEN = [256, 128, 64]
ACT = ["relu", "relu", "tanh"]
LR = 3e-4
EPOCHS = 30
#

CONFIGS = {
    "base (aucune)":             dict(),
    "dropout 0.1":               dict(dropout=0.1),
    "dropout 0.3":               dict(dropout=0.3),
    "dropout 0.5":               dict(dropout=0.5),
    "weight decay 1e-5":         dict(wd=1e-5),
    "weight decay 1e-4":         dict(wd=1e-4),
    "weight decay 1e-3":         dict(wd=1e-3),
    "batchnorm":                 dict(norm="batch"),
    "early stopping (p=5)":      dict(patience=5),
    "dropout 0.3 + wd 1e-4":     dict(dropout=0.3, wd=1e-4),
    "batchnorm + dropout 0.3":   dict(norm="batch", dropout=0.3),
    "bn + drop 0.3 + wd + ES":   dict(norm="batch", dropout=0.3, wd=1e-4, patience=5),
}

df = data.load_train()
df_tr, df_va, _ = data.split_by_subject(df)
X_tr, y_tr, X_va, y_va, _ = data.prepare(df_tr, df_va)
in_dim = X_tr.shape[1]
print("paramètres :", count_params(MLP(in_dim=in_dim, hidden=HIDDEN, activation=ACT)))

rows, logs = [], {}
for name, cfg in CONFIGS.items():
    dropout = cfg.get("dropout", 0.0)
    wd = cfg.get("wd", 0.0)
    norm = cfg.get("norm")
    patience = cfg.get("patience")

    def build(seed, dropout=dropout, wd=wd, norm=norm):
        tl, vl = data.make_loaders(X_tr, y_tr, X_va, y_va, seed=seed)
        model = MLP(in_dim=in_dim, hidden=HIDDEN, activation=ACT, dropout=dropout, norm=norm)
        opt = training.make_optimizer("adam", model.parameters(), lr=LR, weight_decay=wd)
        return model, tl, vl, opt

    res = training.run_seeds(build, seeds=config.SEEDS, epochs=EPOCHS, patience=patience)
    runs = res["runs"]
    idx = [(r["history"]["best_epoch"] - 1) if patience else -1 for r in runs]
    val_f1 = [r["val_f1"] for r in runs]
    val_acc = [r["val_acc"] for r in runs]
    train_acc = [r["history"]["train_acc"][i] for r, i in zip(runs, idx)]
    val_loss = [r["history"]["val_loss"][i] for r, i in zip(runs, idx)]
    rows.append({
        "config": name,
        "val_f1": metrics.mean_std(val_f1),
        "val_acc": metrics.mean_std(val_acc),
        "val_loss": round(float(np.mean(val_loss)), 4),
        "train_acc": round(float(np.mean(train_acc)), 4),
        "gap": round(float(np.mean(train_acc) - np.mean(val_acc)), 4),
        "best_epoch": round(float(np.mean([r["history"]["best_epoch"] for r in runs])), 1),
        "time_s": round(float(np.mean([r["history"]["train_time_s"] for r in runs])), 1),
    })
    logs[name] = res
    print("fait :", name, flush=True)

table = pd.DataFrame(rows)
print("\n=== Régularisation (moyenne ± écart-type sur 3 graines) ===")
print(table.to_string(index=False))

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(config.RESULTS_DIR / "p4_regularisation.csv", index=False)
training.save_json(logs, config.LOG_DIR / "p4_regularisation.json")