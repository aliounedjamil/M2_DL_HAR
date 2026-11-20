import sys
sys.path.append("src")

import json
import numpy as np
from har import config

logs = json.loads((config.LOG_DIR / "p4_regularisation.json").read_text())
KEYS = {"base": "base (aucune)", "dropout5": "dropout 0.5",
        "es": "early stopping (p=5)", "combo": "bn + drop 0.3 + wd + ES"}

print(f"{'config':10s} {'ép. min loss':>13s} {'min val_loss':>13s} {'std F1 (ép>=5)':>15s}")
for short, key in KEYS.items():
    rows = []
    for r in logs[key]["runs"]:
        vl = np.array(r["history"]["val_loss"])
        vf = np.array(r["history"]["val_f1"])
        rows.append((vl.argmin() + 1, vl.min(), vf[5:].std(ddof=1)))
    m = np.mean(rows, axis=0)
    print(f"{short:10s} {m[0]:13.1f} {m[1]:13.4f} {m[2]:15.4f}")