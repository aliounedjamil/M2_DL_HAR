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
│   ├── plots.py              # courbes d'apprentissage, matrices de confusion, projections 2D
│   └── interpret.py          # t-SNE / UMAP sur entrées et couche cachée (Partie 6)
├── notebooks/
│   ├── 00_exploration.ipynb  # Partie 1.2 (EDA)
│   ├── V1_one_hidden_layer.ipynb   # Version 1 du rendu (Parties 1–2) — Alioune
│   ├── 06_interpretability.ipynb   # Partie 6 (mise au point sur le modèle V1) — Alioune
│   └── V2_deep_network.ipynb       # Version 2 du rendu (Parties 3–5, 7) — Madjiguène
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

Les notebooks du parcours A (`00_exploration`, `V1_one_hidden_layer`, `06_interpretability`) sont **autonomes**,
dans le style des TP du cours : les fonctions (`train_epoch`, `evaluate`, `fit`, `lr_range_test`…) y sont écrites
directement. Le module `src/har` reste disponible pour la V2 :

```python
import sys; sys.path.append("../src")
from har import data, models, training, metrics, plots, config
```

## Exécuter les notebooks

**En local** : après l'installation ci-dessus, `jupyter notebook`, ouvrir le dossier `notebooks/`.

**Sur Google Colab** (GPU inutile, le CPU suffit) :

```python
!git clone https://github.com/aliounedjamil/M2_DL_HAR.git   # dépôt privé : utiliser un jeton GitHub
%cd M2_DL_HAR/notebooks
!pip -q install umap-learn
# placer train.csv et test.csv dans M2_DL_HAR/data/HAR/ (upload, ou Google Drive + variable HAR_DATA_DIR)
```

Ordre conseillé (parcours A) : `00_exploration` → `V1_one_hidden_layer` (15–25 min sur CPU) → `06_interpretability`
(lit la configuration retenue écrite par V1 dans `results/logs/v1_configuration_retenue.json`).
Essai rapide de bout en bout : variable d'environnement `HAR_FAST=1` (peu d'époques ; résultats non valables).

## Répartition (résumé) — deux parcours indépendants

| Parcours A — **Alioune** (commence en premier) | Parcours B — **Madjiguène** |
|---|---|
| P1.2 Exploration · P1.3 Découpage · P1.4 Référence | P3 Largeur et profondeur |
| P2 MLP 1 couche cachée (activations, optimiseurs, LR, graines) | P4 Régularisation |
| P6 t-SNE / UMAP (`interpret.py`) | P5 Courbes + test unique V2 + matrices de confusion |
| Notebook **V1** + son test unique | P7 Bonus · Notebook **V2** |

Aucune tâche d'un parcours n'attend un résultat de l'autre : le découpage de validation est figé
dans le code et chacun évalue ses propres modèles sur le test, une seule fois.
Détails et calendrier : **[TASKS.md](TASKS.md)**.

## Règles de travail (Git)

1. `main` est protégée : on n'y pousse jamais directement.
2. Une branche par tâche : `feat/p2-optimiseurs`, `fix/data-split`, `report/intro`…
3. Une Pull Request par tâche, relue et approuvée par l'autre membre avant fusion
   (cela garantit que chacun comprend tout le projet).
4. Messages de commit clairs : `P3: ajout des réseaux 2-4 couches`.
5. Avant d'ouvrir une PR : *Kernel → Restart & Run All* doit passer sans erreur.
6. **Le jeu de test n'est utilisé qu'une seule fois** : uniquement dans la section 7 de V1 et de V2,
   une fois les configurations figées. Aucun autre code ne doit appeler `load_test()`.
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
