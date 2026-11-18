# Répartition des tâches — deux parcours indépendants

Principe : le projet est coupé en **deux parcours qui ne dépendent pas l'un de l'autre**.
Chacun peut avancer à son rythme sans attendre l'autre ; Alioune démarre en premier.

- **Parcours A — DIOP Alioune** : données, référence, réseau à 1 couche cachée (**Version 1**), interprétabilité.
- **Parcours B — FALL Madjiguène** : réseaux larges et profonds, régularisation, bonus (**Version 2**).

Charge estimée : **A ≈ 10,5 j · B ≈ 10,5 j**.

## Ce qui rend les parcours indépendants

| Risque de dépendance | Règle qui le supprime |
|---|---|
| Le découpage train/val | Figé dans le code : `data.split_by_subject(df)` avec `config.SPLIT_SEED = 42` → les **mêmes 5 volontaires** pour tout le monde, sans attendre la P1.3. |
| Le code commun (`src/har/`) | Déjà écrit. On ne fait que des **ajouts** (nouvelles fonctions / nouveaux arguments optionnels), jamais de changement de signature. Tout changement passe par une PR. |
| Le pas d'apprentissage | B ne réutilise pas les résultats de la P2 : elle règle son propre LR (petite grille) sur ses réseaux profonds. |
| L'évaluation finale sur le test | Chacun évalue **ses** modèles **une seule fois** dans la section 7 de **son** notebook : A → référence + MLP 1 couche (V1) ; B → 2–3 configs profondes (V2). |
| t-SNE / UMAP sur le réseau final | A écrit `src/har/interpret.py` et le met au point sur son modèle V1. B n'a qu'**un appel de fonction** à ajouter dans la V2, en toute fin de projet. |

Seuls points de rencontre : la relecture des PR (facultative pour avancer) et le rapport final (semaine 6).

---

## Parcours A — DIOP Alioune (commence en premier)

| # | Tâche | Charge |
|---|---|---|
| P0 | Vérifier / compléter `data.py` et `training.py` sur les vraies données | 1 j |
| P1.2 | Exploration des données (`00_exploration.ipynb`) : tailles, volontaires, classes, stats, histogrammes, boxplots, corrélations, discussion prétraitement, `subject` en entrée ? | 1 j |
| P1.3 | Validation par volontaires entiers vs lignes aléatoires (justification + comparaison) | 1 j |
| P1.4 | Référence : régression logistique (accuracy, F1 macro, F1 par classe) | 0,5 j |
| P2.1 | MLP [64] : perte initiale ≈ ln 6, surapprentissage d'un petit lot | 0,5 j |
| P2.2 | Activations (ReLU / Sigmoid / Tanh) × optimiseurs (SGD / momentum / Adam), LR réglé ; accuracy vs F1 | 2 j |
| P2.3 | ≥ 2 configs × 3 graines, moyenne ± écart-type | 0,5 j |
| P6 | `src/har/interpret.py` + `06_interpretability.ipynb` : t-SNE / UMAP sur entrées brutes vs dernière couche cachée (modèle V1), lien avec la matrice de confusion **de validation**, perplexité, variabilité | 2 j |
| V1 | Assemblage `V1_one_hidden_layer.ipynb` + **section 7 : test une seule fois** (référence + MLP V1, matrice de confusion, réponse finale à la P1.3) | 0,5 j |
| Rapport | Introduction · Données et découpage · Référence · Résultats P2 · Interprétabilité | 1,5 j |

## Parcours B — FALL Madjiguène

| # | Tâche | Charge |
|---|---|---|
| P0 | Vérifier / compléter `models.py`, `metrics.py`, `plots.py` | 1 j |
| P3.1 | Largeur 64 / 128 / 256 : nb de paramètres, performance, overfitting (3 graines) | 1 j |
| P3.2 | 2 à 4 couches cachées, combinaisons d'activations : acc, F1, paramètres, temps | 1,5 j |
| P4 | Régularisation sur son meilleur réseau : Dropout 0,1/0,3/0,5, weight decay, BatchNorm, early stopping ; retenir 2–3 configs → `results/selected_configs.json` | 2 j |
| P5.1 | Courbes d'apprentissage des configs retenues | 0,5 j |
| P7 | Bonus : recherche automatique d'hyperparamètres (random search / Optuna, validation seulement) | 1 j |
| P5.2 | **Section 7 de la V2 : test une seule fois** sur les 2–3 configs, matrices de confusion, classes difficiles (point de vue capteurs) | 1 j |
| V2 | Assemblage `V2_deep_network.ipynb` (+ un appel à `interpret.py` pour la t-SNE/UMAP du modèle final) | 0,5 j |
| Rapport | Architectures (+ paramètres) · Résultats P3, P4 · Évaluation finale · Discussion · Mise en page | 2 j |

## Ensemble (semaine 6)

- [ ] Tableau final comparant référence, V1 et V2 (validation vs test) dans le rapport.
- [ ] Conclusion, relecture croisée, section « usage d'outils d'IA » (`report/AI_USAGE.md`).
- [ ] `scripts/make_submission.sh` → `M2_DL_2026_DIOP_FALL.zip` sur Moodle.
- [ ] Bonus optionnel : CNN 1D / RNN sur signaux bruts.

---

## Calendrier indicatif

| Semaine | Alioune (A) | Madjiguène (B) |
|---|---|---|
| S1 | P0 · P1.2 · P1.3 · P1.4 | P0 · P3.1 |
| S2 | P2.1 · P2.2 | P3.2 |
| S3 | P2.3 · rédaction P1–P2 | P4 |
| S4 | P6 (t-SNE / UMAP) | P5.1 · P7 bonus |
| S5 | V1 + test unique · rédaction | P5.2 test unique · V2 · rédaction |
| S6 | **Ensemble** : rapport final, relecture, ZIP | **Ensemble** |
