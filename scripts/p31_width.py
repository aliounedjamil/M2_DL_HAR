import sys
sys.path.append("src")

import numpy as np
import pandas as pd

from har import config, data, metrics, training
from har.models import MLP, count_params

WIDTHS = [64, 128, 256]
LRS = [1e-4, 3e-4, 1e-3, 3e-3]
EPOCHS = 30

df = data.load_train()
df_tr, df_va, _ = data.split_by_subject(df)
X_tr, y_tr, X_va, y_va, _ = data.prepare(df_tr, df_va)
in_dim = X_tr.shape[1]

rows, logs = [], {}
for width in WIDTHS:
    for lr in LRS:
        def build(seed, width=width, lr=lr):
            tl, vl = data.make_loaders(X_tr, y_tr, X_va, y_va, seed=seed)
            model = MLP(in_dim=in_dim, hidden=[width])
            opt = training.make_optimizer("adam", model.parameters(), lr=lr)
            return model, tl, vl, opt

        res = training.run_seeds(build, seeds=config.SEEDS, epochs=EPOCHS)
        runs = res["runs"]
        val_f1 = [r["val_f1"] for r in runs]
        val_acc = [r["val_acc"] for r in runs]
        train_acc = [r["history"]["train_acc"][-1] for r in runs]
        times = [r["history"]["train_time_s"] for r in runs]
        rows.append({
            "width": width,
            "params": count_params(MLP(in_dim=in_dim, hidden=[width])),
            "lr": lr,
            "val_f1": metrics.mean_std(val_f1),
            "val_acc": metrics.mean_std(val_acc),
            "train_acc": round(float(np.mean(train_acc)), 4),
            "gap": round(float(np.mean(train_acc) - np.mean(val_acc)), 4),
            "time_s": round(float(np.mean(times)), 1),
            "_f1": float(np.mean(val_f1)),
        })
        logs[f"w{width}_lr{lr}"] = res
        print("fait :", width, lr, flush=True)

table = pd.DataFrame(rows)
print("\n=== Grille complète ===")
print(table.drop(columns="_f1").to_string(index=False))
best = table.loc[table.groupby("width")["_f1"].idxmax()]
print("\n=== Meilleur LR par largeur ===")
print(best.drop(columns="_f1").to_string(index=False))

training.save_json(logs, config.LOG_DIR / "p31_width_lr_grid.json")