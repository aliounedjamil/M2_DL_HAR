#!/usr/bin/env bash
# Construit l'archive de rendu Moodle (sujet, partie 8) : M2_DL_2026_DIOP_FALL.zip
#   1. le rapport (PDF, 6 pages max) ;
#   2. le code des 2 versions (notebooks V1 et V2, avec le code et les résultats dont la V2 a besoin).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=M2_DL_2026_DIOP_FALL
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/code/notebooks" "$OUT/code/scripts" "$OUT/code/results" "$OUT/code/data/HAR"
cp report/main.pdf "$OUT/rapport_DIOP_FALL.pdf"
cp notebooks/V1_one_hidden_layer.ipynb notebooks/V2_deep_network.ipynb "$OUT/code/notebooks/"
cp -r src "$OUT/code/"
cp scripts/p*.py "$OUT/code/scripts/"
cp -r results/logs results/*.csv results/*.json "$OUT/code/results/"
cp requirements.txt "$OUT/code/"
cat > "$OUT/code/LISEZMOI.md" <<'TXT'
# Code — Projet HAR (Alioune DIOP, Madjiguène FALL)

- `notebooks/V1_one_hidden_layer.ipynb` : version 1, réseau à une couche cachée.
- `notebooks/V2_deep_network.ipynb` : version 2, réseau profond, régularisation, interprétabilité, recherche aléatoire.
- `src/har/` : fonctions communes utilisées par la V2 (données, modèle, `train_epoch`, `evaluate`, métriques).
- `scripts/` : expériences longues de la V2 ; leurs résultats sont dans `results/` et relus par le notebook.

Pour réexécuter : `pip install -r requirements.txt`, placer `train.csv` et `test.csv` dans `data/HAR/`,
puis *Restart & Run All* sur chaque notebook (graines fixées).
TXT
find "$OUT" -name "__pycache__" -prune -exec rm -rf {} +
zip -rq "$OUT.zip" "$OUT" && rm -rf "$OUT"
echo "Créé : $OUT.zip"
