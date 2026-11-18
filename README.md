# M2 MALIA — Deep Learning : Étude progressive des MLP sur le jeu de données HAR

Projet de Deep Learning (enseignant : Walid Bechkit, 2026–2027).
Binôme : **DIOP Alioune** · **FALL Madjiguène**

Objectif : construire et analyser des réseaux de neurones multicouches (MLP) pour la
reconnaissance d'activités humaines (6 classes) à partir des données de capteurs de smartphone,
avec une méthodologie rigoureuse : référence simple, validation par volontaires, résultats répétés
sur plusieurs graines, test utilisé **une seule fois**.

---

## Structure du dépôt

```
M2_DL_HAR/
├── data/HAR/                 # train.csv, test.csv (Kaggle) — NON versionnés
├── src/har/                  # code commun importé par les notebooks
│   ├── config.py             # chemins, graines, constantes
│   ├── data.py               # chargement, découpage par volontaire, DataLoaders
│   ├── models.py             # MLP configurable (largeur, profondeur, activations, dropout, BN)
│   ├── training.py           # train_epoch, evaluate, fit (early stopping), multi-graines
│   ├── metrics.py            # accuracy, F1 macro / par classe, matrice de confusion
│   └── plots.py              # courbes d'apprentissage, matrices de confusion, projections 2D
├── notebooks/
│   ├── 00_exploration.ipynb  # Partie 1.2 (EDA)
│   ├── V1_one_hidden_layer.ipynb   # Version 1 du rendu (Parties 1–2)
│   └── V2_deep_network.ipynb       # Version 2 du rendu (Parties 3–7)
├── results/                  # figures + logs JSON des expériences (résultats val, mean ± std)
├── report/                   # rapport (6 pages max) : main.tex + figures
├── scripts/
│   ├── create_github_issues.sh     # crée labels, jalons et issues de la répartition
│   └── make_submission.sh          # construit M2_DL_2026_DIOP_FALL.zip
└── TASKS.md                  # répartition détaillée des tâches
```

## Installation

```bash
git clone https://github.com/<compte>/M2_DL_HAR.git
cd M2_DL_HAR
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Télécharger le jeu de données sur
[Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
et placer `train.csv` et `test.csv` dans `data/HAR/`.

Dans les notebooks, le code commun s'importe ainsi :

```python
import sys; sys.path.append("../src")
from har import data, models, training, metrics, plots, config
```

## Répartition (résumé)

| Partie | Responsable | Relecteur |
|---|---|---|
| 0. Socle commun (`src/har`) | les deux (voir TASKS.md) | croisé |
| 1.2 Exploration des données | Madjiguène | Alioune |
| 1.3 Découpage par volontaire + comparaison | Alioune | Madjiguène |
| 1.4 Référence (régression logistique) | Alioune | Madjiguène |
| 2. MLP 1 couche cachée (activations, optimiseurs, LR, métriques, graines) | Alioune | Madjiguène |
| 3. Largeur et profondeur | Madjiguène | Alioune |
| 4. Régularisation | Alioune | Madjiguène |
| 5. Courbes + évaluation finale test + matrices de confusion | Madjiguène | Alioune |
| 6. t-SNE / UMAP | Madjiguène | Alioune |
| 7. Bonus : recherche d'hyperparamètres auto | Alioune | Madjiguène |
| 7. Bonus : CNN 1D sur signaux bruts | optionnel, à deux | — |
| Notebook **V1** (assemblage) | Alioune | Madjiguène |
| Notebook **V2** (assemblage) | Madjiguène | Alioune |
| Rapport | chacun ses parties, voir TASKS.md | croisé |

Détails, critères de fin et calendrier : **[TASKS.md](TASKS.md)**.

## Règles de travail (Git)

1. `main` est protégée : on n'y pousse jamais directement.
2. Une branche par tâche : `feat/p2-optimiseurs`, `fix/data-split`, `report/intro`…
3. Une Pull Request par tâche, relue et approuvée par l'autre membre avant fusion
   (cela garantit que chacun comprend tout le projet).
4. Messages de commit clairs : `P3: ajout des réseaux 2-4 couches`.
5. Avant d'ouvrir une PR : *Kernel → Restart & Run All* doit passer sans erreur.
6. **Le jeu de test n'est utilisé qu'une seule fois**, en Partie 5.2, ensemble, à la fin.
   Aucun autre code ne doit appeler `load_test()`.
7. Toute utilisation d'outils d'IA générative est notée dans `report/AI_USAGE.md`
   (où, et comment le résultat a été vérifié), comme demandé par le sujet.

### Éviter les conflits dans les notebooks

Les notebooks fusionnent mal. Conseils :
- chacun travaille dans **son** notebook de brouillon (`notebooks/scratch_<prenom>_*.ipynb`, ignoré par Git) ;
- le code réutilisable va dans `src/har/` (fichiers `.py`, fusion facile) ;
- seul le responsable d'un notebook de rendu (V1 ou V2) le modifie ; l'autre propose ses cellules par PR ;
- vider les sorties avant de commiter un notebook de travail : `jupyter nbconvert --clear-output --inplace <nb>`
  (les notebooks de rendu finaux, eux, sont commités **avec** leurs sorties).

## Reproductibilité

`training.set_seed(seed)` fixe les graines Python / NumPy / PyTorch. Graines par défaut : `config.SEEDS = [0, 1, 2]`.
Toute comparaison importante se fait en moyenne ± écart-type sur ces graines.
