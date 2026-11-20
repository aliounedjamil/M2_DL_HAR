import sys
sys.path.append("src")

import numpy as np
import pandas as pd

from har import config, data, metrics, training
from har.models import MLP, count_params

ARCHS = {
    "2 couches": [128, 64],
    "3 couches": [256, 128, 64],
    "4 couches": [256, 128, 64, 32],
}
ACTS = {
    "relu": lambda d: "relu",
    "tanh": lambda d: "tanh",
    "sigmoid": lambda d: "sigmoid",
    "relu+tanh": lambda d: ["relu"] * (d - 1) + ["tanh"],   # tanh sur la dernière couche cachée
}
LRS = [1e-4, 3e-4, 1e-3]
EPOCHS = 30

df = data.load_train()
df_tr, df_va, _ = data.split_by_subject(df)
X_tr, y_tr, X_va, y_va, _ = data.prepare(df_tr, df_va)
in_dim = X_tr.shape[1]

rows, logs = [], {}
for arch, hidden in ARCHS.items():
    for act_name, act_fn in ACTS.items():
        act = act_fn(len(hidden))
        n_params = count_params(MLP(in_dim=in_dim, hidden=hidden, activation=act))
        for lr in LRS:
            def build(seed, hidden=hidden, act=act, lr=lr):
                tl, vl = data.make_loaders(X_tr, y_tr, X_va, y_va, seed=seed)
                model = MLP(in_dim=in_dim, hidden=hidden, activation=act)
                opt = training.make_optimizer("adam", model.parameters(), lr=lr)
                return model, tl, vl, opt

            res = training.run_seeds(build, seeds=config.SEEDS, epochs=EPOCHS)
            runs = res["runs"]
            val_f1 = [r["val_f1"] for r in runs]
            val_acc = [r["val_acc"] for r in runs]
            train_acc = [r["history"]["train_acc"][-1] for r in runs]
            times = [r["history"]["train_time_s"] for r in runs]
            rows.append({
                "arch": arch, "activation": act_name, "params": n_params, "lr": lr,
                "val_f1": metrics.mean_std(val_f1),
                "val_acc": metrics.mean_std(val_acc),
                "train_acc": round(float(np.mean(train_acc)), 4),
                "gap": round(float(np.mean(train_acc) - np.mean(val_acc)), 4),
                "time_s": round(float(np.mean(times)), 1),
                "_f1": float(np.mean(val_f1)),
            })
            logs[f"{arch}_{act_name}_lr{lr}"] = res
            print("fait :", arch, act_name, lr, flush=True)

table = pd.DataFrame(rows)
best = table.loc[table.groupby(["arch", "activation"])["_f1"].idxmax()]
best = best.sort_values("_f1", ascending=False)
print("\n=== Meilleur LR par (profondeur, activation), classé par F1 ===")
print(best.drop(columns="_f1").to_string(index=False))

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
table.drop(columns="_f1").to_csv(config.RESULTS_DIR / "p32_depth_activation.csv", index=False)
training.save_json(logs, config.LOG_DIR / "p32_depth_activation.json")