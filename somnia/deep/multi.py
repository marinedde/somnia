"""
Stades à partir de plusieurs capteurs : EEG, second EEG, yeux (EOG), menton (EMG) — horizon 1.3.

Pourquoi : un technicien ne score pas avec l'EEG seul. Les mouvements oculaires rapides et
l'effondrement du tonus du menton signent le REM ; les mouvements oculaires lents, l'endormissement ;
un menton tonique et des clignements, l'éveil. Avec l'EEG seul, éveil calme, N1 et REM se ressemblent.

Même architecture que `somnia.deep.seq` (encodeur par époque + GRU sur la nuit). Seule la première
convolution change : elle lit C capteurs au lieu d'un.

Deux normalisations, et c'est voulu :
  - EEG (canaux 0 et 1) : centré-réduit PAR ÉPOQUE, comme avant (c'est la forme des ondes qui compte) ;
  - EOG et EMG (canaux 2, 3, 4) : centrés par époque mais réduits PAR NUIT. Ce qui compte pour le
    menton est son niveau par rapport au reste de la nuit ; le réduire par époque effacerait
    justement l'effondrement du tonus en REM.

Départ : l'encodeur reprend les poids du réseau EEG ; les poids des nouveaux capteurs partent de
zéro. Le réseau commence donc exactement comme le modèle EEG et apprend ce que les autres ajoutent.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from somnia.deep.data import PROCESSED, normaliser_lot
from somnia.deep.model import Encodeur
from somnia.deep.seq import N_STADES

MULTI_DIR = Path(os.environ.get("SHHS_MULTI", PROCESSED.parent / "processed_multi"))
NOMS_CANAUX = ("EEG", "EEG2", "EOG-G", "EOG-D", "EMG")
PAR_NUIT = (2, 3, 4)            # canaux réduits par nuit (EOG, EMG) ; les EEG le sont par époque
ECRETAGE = 20.0                 # en écarts-types de la nuit : un artefact ne doit pas dominer l'époque
VARIANTES = {
    "eeg": (0,),
    "eeg_eeg2": (0, 1),
    "eeg_eog": (0, 2, 3),
    "eeg_emg": (0, 4),
    "eeg_eog_emg": (0, 2, 3, 4),
    "tout": (0, 1, 2, 3, 4),
}


@dataclass
class NuitMulti:
    ident: str
    personne: str
    X: np.ndarray           # (n_epoques, C, 3000) float16 ; EOG / EMG déjà réduits par nuit
    y: np.ndarray           # (n_epoques,) int64, −1 si non scoré

    def __len__(self):
        return len(self.y)


def reduire_par_nuit(x: np.ndarray) -> np.ndarray:
    """(n_epoques, 3000) -> centré par époque, divisé par l'écart-type MÉDIAN des époques de la nuit.

    Rien n'est emprunté au technicien : l'échelle se calcule sur le signal seul, comme le ferait un outil réel."""
    x = x.astype(np.float32)
    x = x - x.mean(axis=1, keepdims=True)
    et = x.std(axis=1)
    ok = et > 0
    echelle = float(np.median(et[ok])) if ok.any() else 0.0
    if echelle <= 0:
        return np.zeros_like(x)                 # capteur absent ou plat : des zéros, pas des NaN
    return np.clip(x / echelle, -ECRETAGE, ECRETAGE)


def preparer_signaux(signaux: np.ndarray, canaux: tuple[int, ...]) -> np.ndarray:
    """Sélectionne les capteurs voulus et applique la réduction par nuit aux EOG / EMG."""
    X = np.empty((len(signaux), len(canaux), signaux.shape[2]), dtype=np.float16)
    for k, c in enumerate(canaux):
        X[:, k] = reduire_par_nuit(signaux[:, c]) if c in PAR_NUIT else signaux[:, c]
    return X


def personnes_des_nuits(dossier: Path = PROCESSED) -> dict[str, str]:
    """{enregistrement: personne} d'après la cohorte de l'étape 2."""
    with open(dossier / "cohorte.csv", encoding="utf-8") as f:
        return {r["enregistrement"]: r["personne"] for r in csv.DictReader(f) if r.get("exclue") != "True"}


def charger_nuits_multi(personnes: list[str], canaux: tuple[int, ...], dossier: Path = MULTI_DIR) -> list[NuitMulti]:
    voulues, table, nuits = set(personnes), personnes_des_nuits(), []
    for f in sorted(dossier.glob("shhs1-*.npz")):
        if table.get(f.stem) not in voulues:
            continue
        with np.load(f, allow_pickle=False) as d:
            y = d["stades"].astype(np.int64)
            nuits.append(NuitMulti(f.stem, table[f.stem], preparer_signaux(d["signaux"], canaux), y))
    return nuits


class ReseauMulti(nn.Module):
    """Encodeur à C capteurs par époque + GRU bidirectionnel sur la nuit."""

    def __init__(self, canaux: tuple[int, ...] = (0,), dim: int = 64, cache: int = 64, dropout: float = 0.3):
        super().__init__()
        self.canaux = tuple(canaux)
        self.encodeur = Encodeur(dim=dim, c_entree=len(canaux))
        self.sequence = nn.GRU(dim, cache, batch_first=True, bidirectional=True)
        self.tete = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * cache, N_STADES))
        # vrai pour les capteurs à réduire par époque (les EEG) ; les autres arrivent déjà réduits par nuit
        self.register_buffer("par_epoque", torch.tensor([c not in PAR_NUIT for c in canaux]), persistent=False)

    def normaliser(self, x: torch.Tensor) -> torch.Tensor:        # (N, C, T)
        return torch.where(self.par_epoque[None, :, None], normaliser_lot(x), x)

    def encoder_epoques(self, x: torch.Tensor) -> torch.Tensor:   # (N, C, T) -> (N, dim)
        return self.encodeur(self.normaliser(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B, L, C, T) -> (B, L, 5)
        B, L, C, T = x.shape
        z = self.encoder_epoques(x.reshape(B * L, C, T)).reshape(B, L, -1)
        h, _ = self.sequence(torch.relu(z))
        return self.tete(h)

    def partir_de_l_encodeur_eeg(self, etat_encodeur: dict) -> None:
        """Reprend les poids d'un encodeur EEG à un capteur. Les capteurs ajoutés partent de zéro :
        au départ, l'encodeur rend exactement le même vecteur que l'encodeur EEG."""
        if self.canaux[0] != 0:
            raise ValueError("le premier capteur doit être l'EEG principal pour reprendre l'encodeur EEG")
        etat = {k: v.clone() for k, v in etat_encodeur.items()}
        w = etat["blocs.0.net.0.weight"]                          # (16, 1, k)
        neuf = torch.zeros(w.shape[0], len(self.canaux), w.shape[2], dtype=w.dtype)
        neuf[:, 0] = w[:, 0]
        etat["blocs.0.net.0.weight"] = neuf
        self.encodeur.load_state_dict(etat)


@torch.no_grad()
def predire_nuit_multi(modele: ReseauMulti, nuit: NuitMulti, dev: torch.device, lot: int = 256) -> np.ndarray:
    """Logits (n_epoques, 5) : époques encodées par lots, puis la nuit entière lue d'un coup."""
    modele.eval()
    z = [modele.encoder_epoques(torch.from_numpy(nuit.X[i:i + lot].astype(np.float32)).to(dev)) for i in range(0, len(nuit), lot)]
    h, _ = modele.sequence(torch.relu(torch.cat(z)).unsqueeze(0))
    return modele.tete(h).squeeze(0).float().cpu().numpy()
