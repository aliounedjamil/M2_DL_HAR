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
for m in "S1 - Socle + Partie 1" "S2-S3 - Parties 2 et 3" "S4 - Partie 4 + selection" \
         "S5 - Test final + interpretabilite" "S6 - Rapport + rendu"; do
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
issue "P0 · data.py : chargement + découpage par volontaire + DataLoaders" "$ALIOUNE" partie-0 "S1 - Socle + Partie 1" \
"- [ ] load_train / load_test / to_xy
- [ ] split_by_subject (volontaires disjoints) et split_random_rows
- [ ] StandardScaler ajusté sur train seulement
- [ ] make_loaders avec graine"
issue "P0 · training.py : train_epoch, evaluate, fit, run_seeds, LR range test" "$ALIOUNE" partie-0 "S1 - Socle + Partie 1" \
"- [ ] historique par époque (loss/acc/F1, train/val)
- [ ] early stopping avec restauration des meilleurs poids
- [ ] run_seeds -> moyenne ± écart-type"
issue "P0 · models.py : MLP configurable" "$MADJI" partie-0 "S1 - Socle + Partie 1" \
"- [ ] largeur / profondeur / activations par couche
- [ ] dropout, BatchNorm, LayerNorm
- [ ] embed() = dernière couche cachée ; count_params"
issue "P0 · metrics.py + plots.py" "$MADJI" partie-0 "S1 - Socle + Partie 1" \
"- [ ] accuracy, F1 macro, F1 par classe, matrice de confusion
- [ ] courbes d'apprentissage, heatmap de confusion, nuage 2D"

issue "P1.2 · Exploration des données" "$MADJI" partie-1 "S1 - Socle + Partie 1" \
"- [ ] nb d'observations / variables / volontaires (train, test)
- [ ] distribution des classes
- [ ] stats descriptives ; variables déjà mises à l'échelle ?
- [ ] histogrammes, boxplots, corrélations
- [ ] discussion prétraitement ; subject en entrée ?"
issue "P1.3 · Validation par volontaire vs lignes aléatoires" "$ALIOUNE" partie-1 "S1 - Socle + Partie 1" \
"- [ ] 5 volontaires entiers en validation
- [ ] justifier (fenêtres qui se chevauchent)
- [ ] comparer les deux découpages pour un même modèle"
issue "P1.4 · Référence : régression logistique" "$ALIOUNE" partie-1 "S1 - Socle + Partie 1" \
"- [ ] accuracy, F1 macro, F1 par classe en validation"

issue "P2.1 · MLP 1 couche cachée : perte initiale + surapprentissage d'un petit lot" "$ALIOUNE" partie-2 "S2-S3 - Parties 2 et 3" \
"- [ ] perte initiale ≈ ln(6) ≈ 1,79
- [ ] perte → 0 sur un petit lot"
issue "P2.2 · Activations × optimiseurs (LR réglé) + accuracy vs F1" "$ALIOUNE" partie-2 "S2-S3 - Parties 2 et 3" \
"- [ ] ReLU / Sigmoid / Tanh
- [ ] SGD / SGD+momentum / Adam, LR réglé chacun
- [ ] courbes train/val par époque
- [ ] discussion accuracy vs F1 macro / par classe"
issue "P2.3 · Variabilité : ≥ 2 configs × ≥ 3 graines" "$ALIOUNE" partie-2 "S2-S3 - Parties 2 et 3" \
"- [ ] tableau moyenne ± écart-type"

issue "P3.1 · Largeur 64 / 128 / 256 + nb de paramètres" "$MADJI" partie-3 "S2-S3 - Parties 2 et 3" \
"- [ ] impact sur performance et overfitting, 3 graines"
issue "P3.2 · Réseaux 2 à 4 couches, combinaisons d'activations" "$MADJI" partie-3 "S2-S3 - Parties 2 et 3" \
"- [ ] accuracy, F1, nb de paramètres, temps d'entraînement"

issue "P4 · Régularisation (Dropout, weight decay, BatchNorm, early stopping)" "$ALIOUNE" partie-4 "S4 - Partie 4 + selection" \
"- [ ] chaque variante sur 3 graines
- [ ] retenir 2–3 configs -> results/selected_configs.json"

issue "P5.1 · Courbes d'apprentissage des configs retenues" "$MADJI" partie-5 "S5 - Test final + interpretabilite" \
"- [ ] loss + accuracy/F1, train/val"
issue "P5.2 · ÉVALUATION UNIQUE SUR LE TEST + matrices de confusion (ensemble)" "$MADJI" partie-5 "S5 - Test final + interpretabilite" \
"⚠️ À faire une seule fois, à deux.
- [ ] 2–3 configs + référence ; val vs test
- [ ] matrices de confusion ; classes difficiles et explication capteurs
- [ ] conclure sur 1.3 : quel découpage de validation prédit le mieux le test ?"
issue "P6 · t-SNE / UMAP : couche cachée vs entrées brutes" "$MADJI" partie-6 "S5 - Test final + interpretabilite" \
"- [ ] synthèse de lecture
- [ ] projections + lien avec la matrice de confusion
- [ ] perplexité, distances entre groupes, variabilité"

issue "P7 · Bonus : recherche automatique d'hyperparamètres" "$ALIOUNE" bonus "S5 - Test final + interpretabilite" \
"- [ ] random search / Optuna, en validation uniquement"

issue "Notebook V1 (1 couche cachée) — assemblage" "$ALIOUNE" notebook-rendu "S6 - Rapport + rendu" \
"- [ ] 7 sections imposées ; Restart & Run All OK"
issue "Notebook V2 (réseau profond) — assemblage" "$MADJI" notebook-rendu "S6 - Rapport + rendu" \
"- [ ] 7 sections imposées ; Restart & Run All OK"

issue "Rapport · intro, données/découpage, résultats P1.4-P2-P4" "$ALIOUNE" rapport "S6 - Rapport + rendu" ""
issue "Rapport · architectures, résultats P3-P5-P6, discussion, mise en page" "$MADJI" rapport "S6 - Rapport + rendu" ""
issue "Rapport · conclusion + relecture croisée + usage IA + ZIP Moodle" "$ALIOUNE" rapport "S6 - Rapport + rendu" \
"À faire ensemble. Assigner aussi $MADJI."

echo "Terminé."
