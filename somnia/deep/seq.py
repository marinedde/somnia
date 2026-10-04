"""
Modèle de séquence pour les stades : un encodeur par époque, puis une lecture de la nuit.

Pourquoi : un technicien ne score pas une époque seule. Il regarde ce qui précède et ce qui
suit (« on ne passe pas d'éveil à REM d'un coup », « ce N1 est coincé entre deux N2 »). Le
réseau par époque ne peut pas : c'est la première limite mesurée (F1 du N1 à 0,28, accord
éveil / sommeil de 91 %), et le plafond du réseau d'événements respiratoires.

Architecture :
    époque (1, 3000) ──encodeur CNN──> vecteur (64)        # le même encodeur qu'avant
    suite des vecteurs de la nuit ──GRU dans les deux sens──> un stade par époque

L'entraînement se fait sur des tranches de 32 époques consécutives (16 minutes) tirées au
hasard dans les nuits ; l'inférence lit la nuit entière d'un coup.
Les époques non scorées (stade −1) restent dans la séquence, pour ne pas casser la continuité,
mais ne comptent pas dans la perte.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from somnia.deep.data import PROCESSED, normaliser_lot
from somnia.deep.model import Encodeur
from somnia.shhs_prepare import charger_nuit

IGNORER = -100          # valeur ignorée par la perte (époque non scorée)
N_STADES = 5


@dataclass
class NuitEEG:
    ident: str
    personne: str
    X: np.ndarray           # (n_epoques, 3000) float16, µV
    y: np.ndarray           # (n_epoques,) int64, −1 si non scoré

    def __len__(self):
        return len(self.y)


def charger_nuits_eeg(personnes: list[str], dossier: Path = PROCESSED) -> list[NuitEEG]:
    voulues, nuits = set(personnes), []
    for f in sorted(dossier.glob("shhs1-*.npz")):
        n = charger_nuit(f)
        if n["meta"]["personne"] in voulues:
            nuits.append(NuitEEG(n["meta"]["enregistrement"], n["meta"]["personne"], n["eeg"].astype(np.float16), n["stades"]))
    return nuits


class Tranches(Dataset):
    """Tranches de `longueur` époques consécutives. À l'entraînement : début tiré au hasard dans la nuit."""

    def __init__(self, nuits: list[NuitEEG], longueur: int = 32, entrainement: bool = False, graine: int = 42):
        self.nuits, self.longueur, self.entrainement = nuits, longueur, entrainement
        self.index = [(i, d) for i, n in enumerate(nuits) for d in range(0, max(1, len(n) - longueur + 1), longueur)]
        self.rng = np.random.default_rng(graine)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, d = self.index[k]
        n = self.nuits[i]
        if self.entrainement:
            d = int(self.rng.integers(0, max(1, len(n) - self.longueur + 1)))
        x = n.X[d:d + self.longueur].astype(np.float32)
        y = n.y[d:d + self.longueur].astype(np.int64)
        if len(y) < self.longueur:                      # nuit plus courte que la tranche
            pad = self.longueur - len(y)
            x = np.concatenate([x, np.zeros((pad, *x.shape[1:]), dtype=np.float32)])
            y = np.concatenate([y, np.full(pad, -1, dtype=np.int64)])
        y = np.where(y < 0, IGNORER, y)
        return torch.from_numpy(x), torch.from_numpy(y)


class ReseauSequence(nn.Module):
    """Encodeur par époque + GRU bidirectionnel sur la suite des époques."""

    def __init__(self, dim: int = 64, cache: int = 64, couches: int = 1, dropout: float = 0.3):
        super().__init__()
        self.encodeur = Encodeur(dim=dim)
        self.sequence = nn.GRU(dim, cache, num_layers=couches, batch_first=True, bidirectional=True)
        self.tete = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * cache, N_STADES))

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, 3000) -> (B, L, dim) : chaque époque est encodée séparément, puis remise en séquence."""
        B, L, T = x.shape
        z = self.encodeur(normaliser_lot(x.reshape(B * L, 1, T)))
        return z.reshape(B, L, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, L, 3000) -> (B, L, 5)
        h, _ = self.sequence(torch.relu(self.encoder(x)))
        return self.tete(h)


@torch.no_grad()
def predire_nuit(modele: ReseauSequence, nuit: NuitEEG, dev: torch.device, lot: int = 512) -> np.ndarray:
    """Logits (n_epoques, 5) : les époques sont encodées par lots, puis la nuit entière est lue d'un coup."""
    modele.eval()
    z = []
    for i in range(0, len(nuit), lot):
        x = torch.from_numpy(nuit.X[i:i + lot].astype(np.float32)).to(dev)
        z.append(modele.encodeur(normaliser_lot(x.unsqueeze(1))))
    h, _ = modele.sequence(torch.relu(torch.cat(z)).unsqueeze(0))
    return modele.tete(h).squeeze(0).float().cpu().numpy()


def kappa_par_nuit(nuits: list[NuitEEG], preds: list[np.ndarray]) -> np.ndarray:
    from sklearn.metrics import cohen_kappa_score
    out = []
    for n, p in zip(nuits, preds):
        ok = n.y >= 0
        out.append(cohen_kappa_score(n.y[ok], p[ok]) if ok.sum() > 10 and len(set(n.y[ok])) > 1 else np.nan)
    return np.array(out)


def accord_eveil_sommeil(y: np.ndarray, pred: np.ndarray) -> dict:
    """Ce qui compte pour l'index et pour le réseau d'événements : éveil contre sommeil."""
    ve, pe = y == 0, pred == 0
    return {
        "accord": float((ve == pe).mean()),
        "sensibilite_eveil": float((pe & ve).sum() / max(ve.sum(), 1)),     # éveils reconnus
        "specificite_eveil": float((~pe & ~ve).sum() / max((~ve).sum(), 1)),  # sommeils reconnus
    }
