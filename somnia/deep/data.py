"""
Données SHHS pour les réseaux : lecture des .npz par personne, Dataset PyTorch (tâche 4.1).

Deux tâches, même source :
  - "eeg" : une époque de 30 s (3000 points, µV) -> stade 0-4 ;
  - "ecg" : une fenêtre de 60 s (6000 points, mV) = deux époques en sommeil -> apnée 0/1,
    exactement les mêmes fenêtres et étiquettes que la référence Random Forest
    (`somnia.shhs_prepare.fenetres_60s`).

Normalisation : chaque exemple est centré-réduit SUR LUI-MÊME, au moment du lot. Rien n'est
appris sur l'ensemble des données, donc rien ne fuit de la validation ou du test.

Augmentation (entraînement seulement) :
  - ECG : inversion de signe aléatoire (50 %). 80 % des nuits SHHS ont un ECG inversé par
    câblage : sans ça, le réseau apprendrait que « négatif » veut dire « SHHS ».
  - décalage circulaire de quelques secondes (les deux tâches) : une époque commence là où le
    scoreur l'a décidée, pas là où le signal change.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from somnia.shhs_prepare import charger_nuit, fenetres_60s

PROCESSED = Path(os.environ.get("SHHS_PROCESSED", Path.home() / "data" / "shhs" / "processed"))
SPLIT = Path(__file__).resolve().parents[2] / "data" / "splits" / "shhs_v1.json"
N_CLASSES = {"eeg": 5, "ecg": 2}
LONGUEUR = {"eeg": 3000, "ecg": 6000}


@dataclass
class Tableau:
    X: np.ndarray          # (n, longueur) float16
    y: np.ndarray          # (n,) int64
    personne: np.ndarray   # (n,) str

    def __len__(self):
        return len(self.y)


def personnes_du_split(ensemble: str, split_path: Path = SPLIT) -> list[str]:
    return json.loads(Path(split_path).read_text(encoding="utf-8"))["taches"]["shhs"][ensemble]


def sous_ensemble_de_personnes(personnes: list[str], fraction: float, graine: int = 42) -> list[str]:
    """Pour les lignes « 10 % » et « 1 % des étiquettes » : une fraction des PERSONNES, tirée avec graine."""
    if fraction >= 1.0:
        return sorted(personnes)
    rng = np.random.default_rng(graine)
    n = max(2, int(round(fraction * len(personnes))))
    return sorted(rng.choice(sorted(personnes), size=n, replace=False).tolist())


def charger_tableau(tache: str, personnes: list[str], dossier: Path = PROCESSED) -> Tableau:
    """Concatène les nuits des personnes données. Signaux gardés en float16 (mémoire)."""
    Xs, ys, ps = [], [], []
    voulues = set(personnes)
    for f in sorted(dossier.glob("shhs1-*.npz")):
        n = charger_nuit(f)
        pers = n["meta"]["personne"]
        if pers not in voulues:
            continue
        if tache == "eeg":
            ok = n["stades"] >= 0
            Xs.append(n["eeg"][ok].astype(np.float16)); ys.append(n["stades"][ok])
        else:
            garde, y2, _ = fenetres_60s(n["stades"], n["apnee"])
            m = len(garde)
            Xs.append(n["ecg"][: 2 * m].reshape(m, 6000)[garde].astype(np.float16)); ys.append(y2[garde])
        ps.append(np.full(len(ys[-1]), pers))
    if not Xs:
        raise ValueError(f"aucune nuit trouvée pour {len(voulues)} personnes sous {dossier}")
    return Tableau(np.concatenate(Xs), np.concatenate(ys).astype(np.int64), np.concatenate(ps))


class SignalDataset(Dataset):
    """Un exemple = (signal (1, longueur) float32 brut, étiquette). La normalisation se fait sur le lot."""

    def __init__(self, tableau: Tableau, tache: str, entrainement: bool = False, decalage_max_s: float = 2.0, fs: int = 100):
        self.t, self.tache, self.entrainement = tableau, tache, entrainement
        self.decalage_max = int(decalage_max_s * fs)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        x = self.t.X[i].astype(np.float32)
        if self.entrainement:
            if self.decalage_max > 0:
                x = np.roll(x, np.random.randint(-self.decalage_max, self.decalage_max + 1))
            if self.tache == "ecg" and np.random.rand() < 0.5:
                x = -x
        return torch.from_numpy(x).unsqueeze(0), int(self.t.y[i])


def normaliser_lot(x: torch.Tensor) -> torch.Tensor:
    """Centré-réduit par exemple : (B, 1, L) -> même forme. Un signal plat donne 0, pas NaN."""
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    return (x - m) / (s + 1e-6)


def poids_de_classes(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Poids inversement proportionnels à la fréquence (équivalent de class_weight='balanced')."""
    comptes = np.bincount(y, minlength=n_classes).astype(np.float64)
    comptes[comptes == 0] = 1.0
    w = len(y) / (n_classes * comptes)
    return torch.tensor(w, dtype=torch.float32)
