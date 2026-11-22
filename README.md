# Projet HAR — Étude progressive des réseaux de neurones multicouches

**M2 MALIA — Deep Learning — Université Lumière Lyon 2 — 2026-2027** · Enseignant : Walid Bechkit
**Binôme :** Alioune DIOP · Madjiguène FALL

Reconnaissance de 6 activités (marche, montée, descente, assis, debout, allongé) à partir des 561 variables
calculées sur l'accéléromètre et le gyroscope d'un smartphone, avec des MLP de complexité croissante.

## Répartition

| Alioune DIOP — Version 1 | Madjiguène FALL — Version 2 |
|---|---|
| Exploration et prétraitements (1.2) | Largeur et profondeur (3) |
| Découpage par volontaire vs aléatoire (1.3) | Régularisation (4) |
| Référence : régression logistique (1.4) | Courbes, évaluation finale, confusions (5) |
| MLP à 1 couche : contrôles, pas, activations, optimiseurs, 3 graines (2) | Recherche aléatoire d'hyperparamètres (7) |
| t-SNE / UMAP de la couche cachée (6) | |

## Résultats clés (F1 macro ; moyenne ± écart-type sur 3 graines)

| Modèle | Validation | Test |
|---|---|---|
| Régression logistique (référence) | 0,946 | **0,935** |
| V1 — MLP [64], ReLU, SGD + momentum | 0,950 ± 0,003 | 0,932 ± 0,001 |
| V2 — MLP [256, 128, 64] + BatchNorm + dropout 0,3 + weight decay + arrêt anticipé | 0,952 ± 0,008 | 0,934 ± 0,003 |

- Le découpage aléatoire surestime l'accuracy de test de 3 à 4 points ; le découpage par volontaire de moins de 2.
- Largeur, profondeur et régularisations isolées n'améliorent pas la validation au-delà de la variabilité.
- Aucun MLP ne dépasse la régression logistique au test ; la confusion principale oppose « assis » et « debout ».

Détails : `report/main.pdf` (6 pages).

## Relancer le projet

```bash
pip install -r requirements.txt
# placer train.csv et test.csv (Kaggle, « Human Activity Recognition with Smartphones ») dans data/HAR/
jupyter notebook
```

Dans `notebooks/`, *Restart & Run All*, dans cet ordre (≈ 45 min sur CPU au total) :

1. `00_exploration.ipynb` — exploration (1.2)
2. `V1_one_hidden_layer.ipynb` — découpage, référence, MLP à une couche, **test de la V1**
3. `06_interpretability.ipynb` — t-SNE / UMAP (lit la configuration retenue par la V1)
4. `V2_deep_network.ipynb` — réseaux profonds, régularisation, **test de la V2**

Les deux versions (`V1_one_hidden_layer`, `V2_deep_network`) sont **autonomes** : tout leur code est dans le
notebook, et elles ne demandent que `data/HAR/`. Les expériences longues de la V2 (P3, P4, P7, ≈ 1 h 30) enregistrent
leurs résultats dans `results/` ; à l'exécution suivante, le notebook les relit (`RECALCULER = True` pour tout refaire).
`src/` et `scripts/` contiennent la première version de ce code, avant son intégration dans le notebook V2.

## Méthodologie

- **Validation** : 5 volontaires entiers (3, 15, 16, 21, 22), les mêmes dans les deux versions.
- **Prétraitement** : standardisation avec les statistiques de l'apprentissage uniquement ; `subject` n'est pas une entrée.
- **Comparaisons** : 3 graines (0, 1, 2) ; un écart n'est jugé significatif que s'il dépasse la somme des écarts-types.
- **Test** : évalué une seule fois, en dernière section de la V1 et de la V2, sur des configurations figées.

## Contenu

```
data/HAR/      train.csv, test.csv (non fournis)
notebooks/     les 4 notebooks exécutés
src/har/       première version du code commun de la V2 (intégré depuis dans le notebook)
scripts/       premières versions des expériences V2, et construction de l'archive de rendu
results/       figures et résultats chiffrés produits par les notebooks et les scripts
report/        rapport (main.tex, main.pdf) et déclaration d'usage de l'IA (AI_USAGE.md)
```
