import sys
sys.path.append("src")

import matplotlib.pyplot as plt
import torch

from har import config, data, metrics, plots, training
from har.models import MLP, count_params

df = data.load_train()
print("train :", df.shape, "| volontaires :", df[config.SUBJECT_COL].nunique())

df_tr, df_va, val_subjects = data.split_by_subject(df)
print("volontaires de validation :", val_subjects)
print("lignes train / val :", len(df_tr), len(df_va))
assert not set(df_tr[config.SUBJECT_COL]) & set(df_va[config.SUBJECT_COL]), "volontaire commun !"

X_tr, y_tr, X_va, y_va, scaler = data.prepare(df_tr, df_va)
print("X_tr :", X_tr.shape)

training.set_seed(0)
train_loader, val_loader = data.make_loaders(X_tr, y_tr, X_va, y_va, seed=0)
model = MLP(in_dim=X_tr.shape[1], hidden=[64])
print("paramètres :", count_params(model))

crit = torch.nn.CrossEntropyLoss()
print("perte initiale (val) :", training.evaluate(model, val_loader, crit)["loss"], "| ln 6 =", 1.79)

opt = training.make_optimizer("adam", model.parameters(), lr=1e-3)
hist = training.fit(model, train_loader, val_loader, opt, epochs=20, verbose=True)

out = training.evaluate(model, val_loader, crit, return_preds=True)
print(metrics.compute_metrics(out["y_true"], out["y_pred"]))
print(metrics.per_class_f1(out["y_true"], out["y_pred"]))

plots.learning_curves(hist, title="check P0 : MLP [64]")
plt.show()