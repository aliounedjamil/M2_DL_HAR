#!/usr/bin/env bash
# Crée les labels, jalons et issues de la répartition (TASKS.md) sur GitHub.
# Prérequis : GitHub CLI connecté (`gh auth login`) et exécuté depuis le dépôt cloné.
# Usage : ./scripts/create_github_issues.sh <login_github_alioune> <login_github_madjiguene>
set -euo pipefail

ALIOUNE="${1:?login GitHub de Alioune}"
MADJI="${2:?login GitHub de Madjiguène}"

# ---------- labels
for l in "partie-0:ededed" "partie-1:0e8a16" "partie-2:1d76db" "partie-3:5319e7" \
         "partie-4:fbca04" "partie-5:d93f0b" "partie-6:b60205" "bonus:c5def5" \
         "rapport:006b75" "notebook-rendu:000000" "resp:alioune:0052cc" "resp:madjiguene:e99695"; do
  gh label create "${l%:*}" --color "${l##*:}" --force >/dev/null
done

# ---------- jalons (milestones)
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
for m in S1 S2 S3 S4 S5 S6; do
  gh api "repos/$REPO/milestones" -f title="$m" >/dev/null 2>&1 || true
done

issue () {  # titre, assigné, labels, jalon, corps
  local who; [ "$2" = "$ALIOUNE" ] && who="resp:alioune" || who="resp:madjiguene"
  if gh issue create --title "$1" --assignee "$2" --label "$3,$who" --milestone "$4" --body "$5" >/dev/null 2>&1; then
    echo "  ✓ $1  ($2)"
  else  # assigné pas encore collaborateur : on crée sans assignation, l'étiquette resp:* indique le responsable
    gh issue create --title "$1" --label "$3,$who" --milestone "$4" --body "$5" >/dev/null
    echo "  ✓ $1  (non assignée — étiquette $who)"
  fi
}

echo "Création des issues…"
# ---------------- Parcours A — Alioune
issue "A · P0 · data.py + training.py sur les vraies données" "$ALIOUNE" partie-0 "S1" \
"- [ ] chargement, découpage par volontaire (SPLIT_SEED figé), DataLoaders
- [ ] train_epoch / evaluate / fit / run_seeds / lr_range_test"
issue "A · P1.2 · Exploration des données" "$ALIOUNE" partie-1 "S1" \
"- [ ] tailles, volontaires, distribution des classes, stats descriptives
- [ ] histogrammes, boxplots, corrélations ; discussion prétraitement ; subject en entrée ?"
issue "A · P1.3 · Validation par volontaire vs lignes aléatoires" "$ALIOUNE" partie-1 "S1" \
"- [ ] justification (fenêtres qui se chevauchent) ; comparaison des deux découpages"
issue "A · P1.4 · Référence : régression logistique" "$ALIOUNE" partie-1 "S1" ""
issue "A · P2.1 · MLP [64] : perte initiale + surapprentissage d'un petit lot" "$ALIOUNE" partie-2 "S2" ""
issue "A · P2.2 · Activations × optimiseurs (LR réglé) + accuracy vs F1" "$ALIOUNE" partie-2 "S2" ""
issue "A · P2.3 · Variabilité : ≥ 2 configs × 3 graines" "$ALIOUNE" partie-2 "S3" ""
issue "A · P6 · interpret.py + t-SNE / UMAP (modèle V1, validation)" "$ALIOUNE" partie-6 "S4" \
"- [ ] entrées brutes vs dernière couche cachée ; lien avec la matrice de confusion de validation
- [ ] perplexité, variabilité entre graines, limites"
issue "A · Notebook V1 + test unique (référence + MLP V1)" "$ALIOUNE" notebook-rendu "S5" ""
issue "A · Rapport : intro, données, référence, P2, interprétabilité" "$ALIOUNE" rapport "S5" ""
# ---------------- Parcours B — Madjiguène
issue "B · P0 · models.py + metrics.py + plots.py" "$MADJI" partie-0 "S1" ""
issue "B · P3.1 · Largeur 64 / 128 / 256 + nb de paramètres" "$MADJI" partie-3 "S1" ""
issue "B · P3.2 · Réseaux 2 à 4 couches, combinaisons d'activations" "$MADJI" partie-3 "S2" ""
issue "B · P4 · Régularisation + sélection de 2–3 configs" "$MADJI" partie-4 "S3" ""
issue "B · P5.1 · Courbes d'apprentissage des configs retenues" "$MADJI" partie-5 "S4" ""
issue "B · P7 · Bonus : recherche automatique d'hyperparamètres" "$MADJI" bonus "S4" ""
issue "B · P5.2 · Test unique V2 + matrices de confusion" "$MADJI" partie-5 "S5" ""
issue "B · Notebook V2 (+ appel à interpret.py)" "$MADJI" notebook-rendu "S5" ""
issue "B · Rapport : architectures, P3, P4, évaluation finale, discussion, mise en page" "$MADJI" rapport "S5" ""
# ---------------- Ensemble
issue "Ensemble · Rapport final, relecture croisée, usage IA, ZIP Moodle" "$ALIOUNE" rapport "S6" ""

echo "Terminé."
