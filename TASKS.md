# Répartition des tâches

Principe : chacun **porte** des parties de bout en bout (code, expériences, figures, section du rapport)
et **relit** les parties de l'autre via les Pull Requests. La charge est équilibrée : Alioune a plus
d'expériences de réglage (Parties 2, 4), Madjiguène plus d'analyse et de visualisation (Parties 3, 5, 6) ;
chacun assemble un des deux notebooks de rendu.

Estimation de charge (jours-personne indicatifs) : **Alioune ≈ 9,5 j · Madjiguène ≈ 9,5 j**.

---

## Étape 0 — Socle commun (semaine 1) · à faire en premier, ensemble

| Tâche | Qui | Fichier | Fini quand… |
|---|---|---|---|
| Créer le dépôt, protéger `main`, inviter le binôme | Alioune | GitHub | Madjiguène peut pousser une branche |
| Chargement des CSV, découpage par volontaire, DataLoaders | Alioune | `src/har/data.py` | `split_by_subject` renvoie des volontaires disjoints |
| MLP configurable | Madjiguène | `src/har/models.py` | `MLP(561, [64], 6)` et `MLP(561, [256,128,64], 6, dropout=0.3, batch_norm=True)` fonctionnent |
| `train_epoch`, `evaluate`, `fit`, `run_seeds` | Alioune | `src/har/training.py` | un historique par époque est renvoyé (loss/acc/F1, train/val) |
| Métriques + figures | Madjiguène | `src/har/metrics.py`, `plots.py` | courbes et matrice de confusion s'affichent |

Charge : 1 j chacun.

---

## DIOP Alioune

### P1.3 — Découpage apprentissage / validation (1 j)
- [ ] Réserver 5 volontaires entiers de `train.csv` pour la validation (`config.VAL_SUBJECTS`).
- [ ] Expliquer pourquoi (fenêtres consécutives qui se chevauchent → fuite d'information).
- [ ] Pour un même modèle (régression logistique **et** MLP 64), comparer validation par lignes
      aléatoires vs par volontaire. Lequel est le plus proche du test ? *(à vérifier en P5.2)*

### P1.4 — Référence (0,5 j)
- [ ] Régression logistique multiclasse (scikit-learn) : accuracy, F1 macro, F1 par classe en validation.

### P2 — MLP à une couche cachée (3 j)
- [ ] 2.1 : MLP 64 neurones, CrossEntropyLoss ; vérifier la perte initiale ≈ ln(6) ≈ 1,79 ;
      surapprentissage d'un petit lot (perte → 0).
- [ ] 2.2 : activations ReLU / Sigmoid / Tanh.
- [ ] 2.2 : optimiseurs SGD / SGD+momentum / Adam, **LR réglé pour chacun** (LR range test ou grille log).
- [ ] 2.2 : suivi accuracy + loss train/val à chaque époque ; discussion accuracy vs F1 (macro, par classe)
      en lien avec la distribution des classes (P1.2).
- [ ] 2.3 : ≥ 2 configurations × ≥ 3 graines → moyenne ± écart-type.

### P4 — Régularisation (2 j)
- [ ] Sur le meilleur réseau profond de P3 : Dropout (0,1 / 0,3 / 0,5), weight decay, BatchNorm, early stopping.
- [ ] Chaque variante sur 3 graines ; tableau mean ± std.
- [ ] Retenir les **2–3 meilleures configurations** (val) → les noter dans `results/selected_configs.json`.

### P7 — Bonus : recherche automatique d'hyperparamètres (1 j)
- [ ] Random search (ou Optuna) sur largeur, profondeur, LR, dropout, weight decay — en validation uniquement.

### Assemblage notebook V1 (0,5 j)
- [ ] `notebooks/V1_one_hidden_layer.ipynb` : structure imposée en 7 sections, Restart & Run All OK.

### Rapport (1,5 j)
- [ ] Introduction (motivation, choix méthodologiques).
- [ ] Données, découpage, prétraitements (avec Madjiguène pour l'EDA).
- [ ] Résultats P1.4, P2, P4 (tableaux mean ± std, figures).

**Total ≈ 9,5 j**

---

## FALL Madjiguène

### P1.2 — Exploration des données (1 j)
- [ ] Nombre d'observations, de variables, de volontaires (train et test).
- [ ] Distribution des classes (train, test, par volontaire).
- [ ] Statistiques descriptives (moyenne, écart-type, min, max) ; les variables sont-elles déjà dans [-1, 1] ?
- [ ] Histogrammes, boxplots, carte de corrélations.
- [ ] Discussion prétraitement : normalisation, encodage de `Activity`, retrait de `subject` des entrées (pourquoi).

### P3 — Extension de l'architecture (2,5 j)
- [ ] 3.1 : largeur 64 → 128 → 256 ; nombre de paramètres ; impact sur l'overfitting.
- [ ] 3.2 : 2 à 4 couches cachées ; combinaisons d'activations ; accuracy, F1, nb de paramètres, temps d'entraînement.
- [ ] 3 graines pour les comparaisons importantes.

### P5 — Visualisation et évaluation finale (1,5 j)
- [ ] 5.1 : courbes loss + accuracy/F1 train/val pour les configurations retenues.
- [ ] 5.2 : **évaluation unique sur le test** (à faire ensemble) des 2–3 configs + la référence ; comparaison val vs test.
- [ ] Matrices de confusion test ; discussion des classes difficiles (SITTING/STANDING…) du point de vue capteurs.

### P6 — t-SNE / UMAP (2 j)
- [ ] 6.1 : courte synthèse de lecture (principe, limites).
- [ ] 6.2 : projections des activations de la dernière couche cachée vs des entrées brutes.
- [ ] 6.3 : lien avec la matrice de confusion ; effet de la perplexité ; variabilité entre exécutions.

### Assemblage notebook V2 (0,5 j)
- [ ] `notebooks/V2_deep_network.ipynb` : réseau profond + régularisations + visualisations + bonus, Restart & Run All OK.

### Rapport (2 j)
- [ ] Description des architectures (+ nb de paramètres).
- [ ] Résultats P3, P5, P6 (figures).
- [ ] Discussion critique (performances, généralisation, limites).
- [ ] Mise en page finale (6 pages max).

**Total ≈ 9,5 j**

---

## Ensemble

- [ ] Conclusion du rapport.
- [ ] Relecture croisée complète du rapport.
- [ ] `report/AI_USAGE.md` → section dans le rapport sur l'usage d'outils d'IA.
- [ ] Bonus optionnel : CNN 1D / RNN sur signaux bruts UCI.
- [ ] `scripts/make_submission.sh` → `M2_DL_2026_DIOP_FALL.zip` déposé sur Moodle.

---

## Calendrier indicatif (à ajuster à la date de rendu)

| Semaine | Alioune | Madjiguène |
|---|---|---|
| S1 | Socle `data.py`, `training.py` · P1.3 · P1.4 | Socle `models.py`, `metrics.py`, `plots.py` · P1.2 |
| S2 | P2 (activations, optimiseurs, LR) | P3.1 (largeur) |
| S3 | P2.3 (graines) · assemblage V1 | P3.2 (profondeur) |
| S4 | P4 (régularisation) · sélection des configs | P5.1 courbes · P6 t-SNE/UMAP |
| S5 | P7 bonus · rapport | **P5.2 test (ensemble)** · P6.3 · assemblage V2 · rapport |
| S6 | Relecture croisée, conclusion, ZIP | Relecture croisée, mise en page, ZIP |

Dépendances clés : P3 attend le socle ; P4 attend la meilleure architecture de P3 ; P5.2 et P6 attendent
`results/selected_configs.json` (fin P4).
