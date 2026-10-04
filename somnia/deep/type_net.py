"""
Type d'apnée : obstructive ou centrale, par un petit réseau qui lit les ceintures (horizon 2).

La règle à quatre traits (somnia.resp.effort_respiratoire) ne suffit pas : mesurer une amplitude
« pendant » contre « avant » est fragile quand la minute qui précède contient déjà une autre apnée.
Un lecteur, lui, regarde la forme : des ceintures qui continuent d'osciller (obstructive) ou qui
s'aplatissent avec le flux (centrale). On donne donc la forme au réseau.

Entrée : flux, thorax, abdomen à 10 Hz, de 30 s avant le début de l'apnée à 60 s après (90 s,
900 points), chaque canal centré-réduit sur la fenêtre. Sortie : un logit, P(centrale).
Même encodeur convolutif que pour les stades, à trois canaux d'entrée.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from somnia.deep.model import Encodeur

AVANT_S, APRES_S, FS = 30, 60, 10
N_POINTS = (AVANT_S + APRES_S) * FS


def fenetre_apnee(signaux_10hz: np.ndarray, debut_s: int, decalage_s: int = 0) -> np.ndarray | None:
    """(3, 900) autour du début de l'apnée, ou None si la fenêtre sort de la nuit ou si un canal est plat."""
    a = (debut_s + decalage_s - AVANT_S) * FS
    if a < 0 or a + N_POINTS > signaux_10hz.shape[1]:
        return None
    x = signaux_10hz[:3, a:a + N_POINTS].astype(np.float32)
    et = x.std(axis=1, keepdims=True)
    if not np.isfinite(x).all() or (et < 1e-9).any():
        return None
    return (x - x.mean(axis=1, keepdims=True)) / et


class ReseauType(nn.Module):
    def __init__(self, dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.encodeur = Encodeur(dim=dim, c_entree=3)
        self.tete = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, 3, 900) -> (B,) logits
        return self.tete(self.encodeur(x)).squeeze(-1)


@torch.no_grad()
def proba_centrale_reseau(modele: ReseauType, fenetres: np.ndarray, dev: torch.device | None = None) -> np.ndarray:
    """(N, 3, 900) -> P(centrale) pour chaque apnée."""
    if len(fenetres) == 0:
        return np.zeros(0)
    modele.eval()
    dev = dev or next(modele.parameters()).device
    return torch.sigmoid(modele(torch.from_numpy(np.asarray(fenetres, dtype=np.float32)).to(dev))).cpu().numpy()
