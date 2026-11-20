import sys
sys.path.append("src")

import json
import matplotlib.pyplot as plt

from har import config, plots, training

logs = json.loads((config.LOG_DIR / "p4_regularisation.json").read_text())

COMMON = dict(hidden=[256, 128, 64], activation=["relu", "relu", "tanh"],
              optimizer="adam", lr=3e-4, epochs=30, seeds=config.SEEDS)

# nom court -> (clé dans le log P4, réglages de régularisation)
SELECTED = {
    "combo":    ("bn + drop 0.3 + wd + ES", dict(norm="batch", dropout=0.3, weight_decay=1e-4, patience=5)),
    "es":       ("early stopping (p=5)",    dict(norm=None, dropout=0.0, weight_decay=0.0, patience=5)),
    "dropout5": ("dropout 0.5",             dict(norm=None, dropout=0.5, weight_decay=0.0, patience=None)),
}

out = {}
for short, (key, reg) in SELECTED.items():
    out[short] = {**COMMON, **reg, "p4_key": key, "val_summary": logs[key]["summary"]}
training.save_json(out, config.RESULTS_DIR / "selected_configs.json")

histories = {}
for short, (key, _) in SELECTED.items():
    h = logs[key]["runs"][0]["history"]            # graine 0
    histories[short] = h
    plots.learning_curves(h, title=f"{short} (graine 0, meilleure époque = {h['best_epoch']})",
                          save=f"p51_{short}")

plots.compare_curves(histories, key="val_f1", save="p51_compare_val_f1")
plots.compare_curves(histories, key="val_loss", save="p51_compare_val_loss")
plt.show()