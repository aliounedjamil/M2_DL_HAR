#!/usr/bin/env bash
# Construit l'archive de rendu Moodle (sujet, partie 8) : M2_DL_2026_DIOP_FALL.zip
#   1. le rapport (PDF, 6 pages max) ;
#   2. le code des 2 versions : deux notebooks autonomes (V1, V2), exécutés.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=M2_DL_2026_DIOP_FALL
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT"
cp report/main.pdf "$OUT/rapport_DIOP_FALL.pdf"
cp notebooks/V1_one_hidden_layer.ipynb notebooks/V2_deep_network.ipynb "$OUT/"
zip -rq "$OUT.zip" "$OUT" && rm -rf "$OUT"
echo "Créé : $OUT.zip"
