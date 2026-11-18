#!/usr/bin/env bash
# Construit l'archive de rendu Moodle : M2_DL_2026_DIOP_FALL.zip
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=M2_DL_2026_DIOP_FALL
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/code"
cp report/main.pdf "$OUT/rapport_DIOP_FALL.pdf"
cp notebooks/V1_one_hidden_layer.ipynb notebooks/V2_deep_network.ipynb "$OUT/code/"
cp -r src requirements.txt README.md "$OUT/code/"
find "$OUT" -name "__pycache__" -prune -exec rm -rf {} +
zip -rq "$OUT.zip" "$OUT" && rm -rf "$OUT"
echo "Créé : $OUT.zip"
