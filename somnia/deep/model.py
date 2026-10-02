"""
Réseau convolutif 1D pour un signal d'une voie (tâche 4.2), séparé en encodeur + tête (tâche 5.2).

Entrée  : (B, 1, L) avec L = 3000 (EEG, 30 s) ou 6000 (ECG, 60 s), à 100 Hz.
Sortie  : (B, n_classes) logits.

L'encodeur est une pile de blocs [convolution -> normalisation -> ReLU -> sous-échantillonnage],
suivie d'un pooling global : il produit un vecteur de taille fixe quelle que soit la longueur
du signal. La tête est une petite couche dense. Pour le pré-entraînement contrastif (étape 5),
on réutilise l'encodeur seul.

Taille : ~70 000 paramètres avec les réglages par défaut. Assez petit pour s'entraîner sur un
Mac en quelques minutes par époque, assez grand pour apprendre des motifs de quelques
centaines de millisecondes (QRS, fuseaux, ondes lentes).
"""

from __future__ import annotations

import torch
from torch import nn


class Bloc(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Encodeur(nn.Module):
    """Signal (B, 1, L) -> vecteur (B, dim)."""

    def __init__(self, canaux=(16, 32, 64, 64), noyaux=(15, 9, 7, 5), pas=(4, 4, 2, 2), dim: int = 64):
        super().__init__()
        couches, c_in = [], 1
        for c, k, s in zip(canaux, noyaux, pas):
            couches.append(Bloc(c_in, c, k, s))
            c_in = c
        self.blocs = nn.Sequential(*couches)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(c_in, dim)
        self.dim = dim

    def forward(self, x):
        h = self.blocs(x)                 # (B, C, L')
        h = self.pool(h).squeeze(-1)      # (B, C)
        return self.proj(h)               # (B, dim)


class CNN1D(nn.Module):
    """Encodeur + tête de classification."""

    def __init__(self, n_classes: int, dim: int = 64, dropout: float = 0.2, **kw_encodeur):
        super().__init__()
        self.encodeur = Encodeur(dim=dim, **kw_encodeur)
        self.tete = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, n_classes))

    def forward(self, x):
        return self.tete(self.encodeur(x))


def n_parametres(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
