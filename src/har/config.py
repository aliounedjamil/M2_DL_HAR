"""Constantes partagées : chemins, graines, colonnes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "HAR"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"

TARGET_COL = "Activity"
SUBJECT_COL = "subject"

CLASSES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
N_CLASSES = len(CLASSES)

SEEDS = [0, 1, 2]          # graines utilisées pour toutes les comparaisons répétées
SPLIT_SEED = 42            # graine du choix des volontaires de validation
N_VAL_SUBJECTS = 5         # volontaires entiers réservés à la validation
BATCH_SIZE = 64
