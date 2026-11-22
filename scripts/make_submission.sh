#!/usr/bin/env bash
# Construit l'archive de rendu Moodle : M2_DL_2026_DIOP_FALL.zip
# (rapport PDF + notebooks exécutés + code + scripts + résultats ; sans les données)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=M2_DL_2026_DIOP_FALL
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/code/notebooks"
cp report/main.pdf "$OUT/rapport_DIOP_FALL.pdf"
cp notebooks/00_exploration.ipynb notebooks/V1_one_hidden_layer.ipynb \
   notebooks/06_interpretability.ipynb notebooks/V2_deep_network.ipynb "$OUT/code/notebooks/"
cp -r src scripts results requirements.txt README.md "$OUT/code/"
cp report/AI_USAGE.md "$OUT/code/"
mkdir -p "$OUT/code/data/HAR"          # placer train.csv et test.csv ici pour réexécuter
find "$OUT" -name "__pycache__" -prune -exec rm -rf {} +
zip -rq "$OUT.zip" "$OUT" && rm -rf "$OUT"
echo "Créé : $OUT.zip"
