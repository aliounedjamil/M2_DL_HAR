# Usage d'outils d'IA générative

Le sujet tolère ChatGPT (ou similaire) à condition d'indiquer **où** il a été utilisé et **comment**
les résultats ont été vérifiés. Ajouter une ligne à chaque usage ; ce tableau alimente la section
correspondante du rapport.

| Date | Qui | Outil | Où (fichier / section) | Usage | Vérification / ajustements |
|---|---|---|---|---|---|
| 2026-11-18 | Alioune | Claude | structure du dépôt, `src/har/*.py`, squelettes de notebooks, `TASKS.md` | écriture de l'organisation et du code de base à partir de nos consignes | code testé de bout en bout sur les vraies données ; contrôles de découpage, de perte initiale et de nombre de paramètres |
| 2026-11-20 | Madjiguène | Claude | installation (venv, clonage, PowerShell), vérification de `src/har/*.py` (P0) | guidage pas à pas ; explication du code existant (`models.py`, `metrics.py`, `plots.py`, `data.py`, `training.py`). Une version de `models.py` proposée par l'outil n'a **pas** été retenue : celle du dépôt est conservée | commandes exécutées et erreurs corrigées ; test de bout en bout lancé sur les vraies données (découpage par volontaires, perte initiale ≈ ln 6, courbes) ; nombre de paramètres recalculé à la main (36 358) |
| 2026-11-20 | Madjiguène | Claude | `scripts/p31_width.py` (P3.1), `scripts/p32_depth_activation.py` (P3.2) | écriture des scripts d'expériences (largeur, profondeur, activations, 3 graines) ; aide à la lecture des tableaux | scripts exécutés ; paramètres recalculés ; conclusions formulées en comparant les écarts à l'écart-type entre graines ; cas « 4 couches + sigmoid » signalé comme hypothèse (gradients) non vérifiée |
| 2026-11-20 | Madjiguène | Claude | `scripts/p4_regularisation.py` (P4), choix de l'architecture de base | écriture du script ; discussion du choix de l'architecture régularisée (3 couches, ReLU + tanh) | exécuté deux fois (résultats identiques) ; choix présenté comme défendable et non démontré ; réserve notée sur le biais de l'early stopping (sélection sur la validation) |
| 2026-11-20 | Madjiguène | Claude | `scripts/p51_curves.py`, `scripts/p51_stability.py` (P5.1) | code des figures et de la mesure de stabilité ; aide à la lecture des courbes | figures regardées une à une ; chiffres de stabilité comparés aux courbes ; première lecture corrigée ensuite sur un point (époque retenue par le F1 contre minimum de loss) |
| 2026-11-21 | Madjiguène | Claude | `scripts/p7_random_search.py` (P7) | écriture de la recherche aléatoire et de la ré-évaluation du top 3 | exécuté ; ré-évaluation sur des graines différentes de celle de la recherche pour limiter le biais de sélection ; résultat (pas de gain net) formulé prudemment |
| 2026-11-21 | Madjiguène | Claude | `notebooks/V2_deep_network.ipynb` (sections 5 à 7) | écriture des cellules (cache des résultats, modèles retenus, courbes, t-SNE / UMAP, évaluation finale) ; dépannage du noyau Jupyter et de `seaborn` | *Restart & Run All* ; la cellule 5.5 redonne les chiffres de P4 (contrôle de cohérence) ; `load_test()` appelé une seule fois (section 7) ; configurations figées avant lecture du test |
| 2026-11-22 | Madjiguène | Claude | cellules Markdown du notebook V2 (6.2, 7), description de la Pull Request | aide à la rédaction des interprétations et de la description | texte relu et comparé aux tableaux, matrices et figures ; explications non vérifiées (capteurs, volontaires atypiques) marquées comme hypothèses |
| 2026-11-22 | Madjiguène | Claude | Git (branches, comparaison avec un second dépôt) | explication des commandes | sorties des commandes vérifiées avant toute action ; aucune modification poussée sur l'autre dépôt |
| 2026-11-18 → 22 | Alioune | Claude | `notebooks/00_exploration.ipynb`, `V1_one_hidden_layer.ipynb`, `06_interpretability.ipynb` (parcours A) | écriture des notebooks dans le style des TP (découpage par volontaire, référence, contrôles, réglage du pas, 3 graines, test unique, t-SNE/UMAP) ; alignement de la validation sur celle de la V2 | Restart & Run All sur les vraies données (2026-11-22) ; perte initiale 1,77–1,87 (≈ ln 6), mini-lot appris à 100 %, aucun volontaire commun entre apprentissage et validation, test évalué une seule fois ; contrôle 5.5 de la V2 (F1 réentraînés = P4 à 0,001 près) |
| 2026-11-22 | Alioune | Claude | `report/main.tex` : résumé, introduction, données, référence, MLP 1 couche, espace latent, conclusion ; lignes référence/V1 du tableau de test ; mise à jour des chiffres de test de la V2 après réexécution | rédaction à partir des sorties des notebooks | chaque chiffre relu face aux sorties ; formulations trop fortes corrigées (significativité, écarts validation–test) |
| 2026-11-22 | Alioune | Claude | cellules « Réponse » de `00_exploration.ipynb`, `V1_one_hidden_layer.ipynb` et `06_interpretability.ipynb` (et tableau de synthèse 7.1) | rédaction des réponses à partir des sorties et des figures des notebooks exécutés | chaque chiffre relu face aux sorties (une valeur corrigée : part des classes en validation) |

## Parties où l'apport de l'assistant a été le plus marqué

Le tableau ci-dessus couvre tout l'usage ; ces points sont ceux où l'assistant a apporté des techniques ou du
code que nous n'aurions pas écrits seuls aussi vite :

- **Test d'amplitude du pas** (`lr_range_test`, V1, d'après Smith) : principe proposé et code écrit par l'assistant ;
  ses indications ont été confrontées au balayage du pas sur grille.
- **Mesures de séparation de l'espace latent** (`src/har/interpret.py`, notebook 06) : silhouette, pureté des
  k plus proches voisins et comparaison au chevauchement de la matrice de confusion.
- **Cellule de cohérence de la V2** (section 5.5) : réentraînement des configurations retenues et comparaison
  aux résultats de P4 avant l'évaluation sur le test.
- **Outillage** : script de construction de l'archive (`scripts/make_submission.sh`) et mise en page LaTeX du rapport.
