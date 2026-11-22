import sys
sys.path.append("src")

import numpy as np
import pandas as pd

from har import config, data, metrics, training
from har.models import MLP, count_params

N_TRIALS = 30
SEARCH_SEED = 100          # graine de la recherche, différente de config.SEEDS
rng = np.random.default_rng(123)

df = data.load_train()
df_tr, df_va, _ = data.split_by_subject(df)
X_tr, y_tr, X_va, y_va, _ = data.prepare(df_tr, df_va)
in_dim = X_tr.shape[1]

ARCHS = {"[64]": [64], "[256]": [256], "[128,64]": [128, 64], "[256,128,64]": [256, 128, 64]}


def sample():
    return dict(
        arch=str(rng.choice(list(ARCHS))),
        act=str(rng.choice(["relu", "tanh", "relu+tanh"])),
        lr=float(10 ** rng.uniform(-4, np.log10(3e-3))),
        dropout=float(rng.choice([0.0, 0.1, 0.3, 0.5])),
        wd=float(rng.choice([0.0, 1e-5, 1e-4, 1e-3])),
        norm=str(rng.choice(["none", "batch"])),
        batch_size=int(rng.choice([32, 64, 128])),
    )


def activation(cfg):
    d = len(ARCHS[cfg["arch"]])
    return ["relu"] * (d - 1) + ["tanh"] if cfg["act"] == "relu+tanh" else cfg["act"]


def make_build(cfg):
    def build(seed):
        tl, vl = data.make_loaders(X_tr, y_tr, X_va, y_va, batch_size=cfg["batch_size"], seed=seed)
        model = MLP(in_dim=in_dim, hidden=ARCHS[cfg["arch"]], activation=activation(cfg),
                    dropout=cfg["dropout"], norm=None if cfg["norm"] == "none" else cfg["norm"])
        opt = training.make_optimizer("adam", model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        return model, tl, vl, opt
    return build


# 1) recherche : une seule graine par essai, early stopping (patience 5)
trials = []
for i in range(N_TRIALS):
    cfg = sample()
    r = training.run_seeds(make_build(cfg), seeds=[SEARCH_SEED], epochs=30, patience=5)["runs"][0]
    trials.append({**cfg, "val_f1": r["val_f1"], "val_acc": r["val_acc"],
                   "best_epoch": r["history"]["best_epoch"],
                   "time_s": round(r["history"]["train_time_s"], 1)})
    print(f"essai {i + 1}/{N_TRIALS} : F1 {r['val_f1']:.4f}", flush=True)

table = pd.DataFrame(trials).sort_values("val_f1", ascending=False).reset_index(drop=True)
print("\n=== Top 10 de la recherche (1 graine) ===")
print(table.head(10).to_string())

# 2) ré-évaluation des 3 meilleures sur les graines habituelles (0, 1, 2)
final = []
for _, row in table.head(3).iterrows():
    cfg = {k: row[k] for k in ("arch", "act", "lr", "dropout", "wd", "norm", "batch_size")}
    cfg["batch_size"] = int(cfg["batch_size"])
    res = training.run_seeds(make_build(cfg), seeds=config.SEEDS, epochs=30, patience=5)
    final.append({**cfg,
                  "params": count_params(make_build(cfg)(0)[0]),
                  "val_f1": metrics.mean_std([r["val_f1"] for r in res["runs"]]),
                  "val_acc": metrics.mean_std([r["val_acc"] for r in res["runs"]])})
final = pd.DataFrame(final)
print("\n=== Top 3 ré-évalués sur 3 graines (comparer à combo : 0.9516 ± 0.0052) ===")
print(final.to_string(index=False))

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(config.RESULTS_DIR / "p7_random_search.csv", index=False)
final.to_csv(config.RESULTS_DIR / "p7_top3_reeval.csv", index=False)