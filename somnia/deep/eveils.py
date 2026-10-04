"""
Détection des micro-éveils (horizon 2.1).

Un micro-éveil est une accélération brusque de l'EEG pendant le sommeil, de 3 à 15 secondes,
souvent avec une bouffée de tonus au menton. Il fragmente le sommeil sans réveiller. Un lecteur
en marque plus d'une centaine par nuit : c'est, après les événements respiratoires, l'autre
grande source de clics.

Même idée que le réseau d'événements respiratoires : une décision par seconde, puis des événements.
    entrée  : EEG, second EEG, menton à 100 Hz, fenêtres de 2 minutes (3 × 12 000 points)
    tronc   : trois convolutions qui descendent de 100 Hz à 1 Hz
    contexte: convolutions dilatées (le réseau voit ± 30 s autour de chaque seconde)
    sortie  : pour chaque seconde, rien / micro-éveil

Normalisation : chaque capteur est centré par époque et réduit par nuit (somnia.deep.multi) :
un micro-éveil est un CHANGEMENT d'amplitude et de fréquence par rapport au reste de la nuit ;
réduire par fenêtre l'effacerait en partie.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from somnia.deep.multi import MULTI_DIR, personnes_des_nuits, reduire_par_nuit
from somnia.deep.resp_net import BlocDilate
from somnia.resp import Ev, apparier, masque_vers_evenements, sommeil_par_seconde
from somnia.shhs import Annotations

RAW = Path(os.environ.get("SHHS_DIR", Path.home() / "data" / "shhs" / "raw"))
CANAUX_EVEILS = (0, 1, 4)        # EEG, EEG2, EMG dans les fichiers multi-capteurs
FS = 100
FENETRE_S, PAS_S = 120, 60
DUREE_MIN_S, TROU_MAX_S = 3, 1   # un micro-éveil dure au moins 3 s (règle de scorage) ; un trou d'1 s est comblé
SEUIL_EVEIL = 0.5


def etiquettes_eveils(ann: Annotations, n_sec: int) -> np.ndarray:
    """Booléen par seconde : un micro-éveil marqué par le technicien couvre cette seconde."""
    y = np.zeros(n_sec, dtype=np.int8)
    for e in ann.evenements:
        if e.type.lower().startswith("arousal") or "arousal" in e.nom.lower():
            a, b = max(0, int(np.floor(e.debut))), min(n_sec, int(np.ceil(e.fin)))
            y[a:b] = 1
    return y


@dataclass
class NuitEveils:
    ident: str
    personne: str
    X: np.ndarray            # (3, n_sec * 100) float16, réduit par nuit
    y: np.ndarray            # (n_sec,) int8
    stades: np.ndarray       # (n_epoques,)
    p_sommeil: np.ndarray | None = None

    @property
    def n_sec(self) -> int:
        return len(self.y)


def signaux_continus(signaux: np.ndarray, canaux: tuple[int, ...] = CANAUX_EVEILS) -> np.ndarray:
    """(n_epoques, C, 3000) -> (len(canaux), n_epoques * 3000) float16, chaque capteur réduit par nuit."""
    return np.stack([reduire_par_nuit(signaux[:, c]).reshape(-1) for c in canaux]).astype(np.float16)


def charger_nuits_eveils(personnes: list[str], dossier: Path = MULTI_DIR, sommeil_dir: Path | None = None) -> list[NuitEveils]:
    from somnia.shhs import lire_annotations
    xmls = {p.name.replace("-nsrr.xml", ""): p for p in RAW.rglob("*-nsrr.xml")}
    voulues, table, nuits = set(personnes), personnes_des_nuits(), []
    for f in sorted(dossier.glob("shhs1-*.npz")):
        if table.get(f.stem) not in voulues:
            continue
        with np.load(f, allow_pickle=False) as d:
            X, stades = signaux_continus(d["signaux"]), d["stades"].astype(np.int8)
        n_sec = X.shape[1] // FS
        nuit = NuitEveils(f.stem, table[f.stem], X, etiquettes_eveils(lire_annotations(xmls[f.stem]), n_sec), stades)
        if sommeil_dir is not None and (sommeil_dir / f"{f.stem}_psommeil_multi.npy").exists():
            ps = np.load(sommeil_dir / f"{f.stem}_psommeil_multi.npy").astype(np.float32)[:n_sec]
            nuit.p_sommeil = np.concatenate([ps, np.zeros(n_sec - len(ps), dtype=np.float32)])
        nuits.append(nuit)
    return nuits


def extraire(nuit: NuitEveils, debut_s: int):
    x = nuit.X[:, debut_s * FS:(debut_s + FENETRE_S) * FS].astype(np.float32)
    y = nuit.y[debut_s:debut_s + FENETRE_S].astype(np.int64)
    if x.shape[1] < FENETRE_S * FS:
        x = np.pad(x, ((0, 0), (0, FENETRE_S * FS - x.shape[1])))
        y = np.pad(y, (0, FENETRE_S - len(y)), constant_values=-100)      # hors de la nuit : ignoré par la perte
    return x, y


class FenetresEveils(Dataset):
    """Fenêtres de 2 minutes tirées au hasard dans les nuits : `par_nuit` fenêtres par nuit et par époque d'entraînement."""

    def __init__(self, nuits: list[NuitEveils], par_nuit: int = 120, graine: int = 42):
        self.nuits, self.par_nuit, self.rng = nuits, par_nuit, np.random.default_rng(graine)

    def __len__(self):
        return len(self.nuits) * self.par_nuit

    def __getitem__(self, k):
        nuit = self.nuits[k % len(self.nuits)]
        d = int(self.rng.integers(0, max(1, nuit.n_sec - FENETRE_S)))
        x, y = extraire(nuit, d)
        x = x * self.rng.uniform(0.8, 1.25, size=(x.shape[0], 1)).astype(np.float32)       # gain des capteurs
        return torch.from_numpy(x), torch.from_numpy(y)


class ReseauEveils(nn.Module):
    def __init__(self, canaux: int = len(CANAUX_EVEILS), largeur: int = 64, dilatations=(1, 2, 4, 8, 16), dropout: float = 0.1):
        super().__init__()
        def bloc(c_in, c_out, k, pas):
            return [nn.Conv1d(c_in, c_out, k, stride=pas, padding=k // 2, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True)]
        self.entree = nn.Sequential(*bloc(canaux, 32, 25, 5), *bloc(32, 48, 9, 4), *bloc(48, largeur, 9, 5))     # 100 -> 20 -> 5 -> 1 Hz
        self.contexte = nn.Sequential(*[BlocDilate(largeur, d) for d in dilatations])
        self.tete = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(largeur, 2, 1))

    def forward(self, x):                                   # (B, 3, 12000) -> (B, 2, 120)
        return self.tete(self.contexte(self.entree(x)))


@torch.no_grad()
def proba_eveils(modele: nn.Module, nuit: NuitEveils, dev: torch.device, batch: int = 32) -> np.ndarray:
    """P(micro-éveil) par seconde ; les fenêtres se recouvrent de moitié, on moyenne."""
    modele.eval()
    debuts = list(range(0, max(1, nuit.n_sec - FENETRE_S + 1), PAS_S))
    if debuts[-1] != max(0, nuit.n_sec - FENETRE_S):
        debuts.append(max(0, nuit.n_sec - FENETRE_S))
    somme, compte = np.zeros(nuit.n_sec), np.zeros(nuit.n_sec)
    for i in range(0, len(debuts), batch):
        lot = debuts[i:i + batch]
        x = torch.from_numpy(np.stack([extraire(nuit, d)[0] for d in lot])).to(dev)
        p = torch.softmax(modele(x), dim=1)[:, 1].float().cpu().numpy()
        for d, pi in zip(lot, p):
            n = min(FENETRE_S, nuit.n_sec - d)
            somme[d:d + n] += pi[:n]; compte[d:d + n] += 1
    return somme / np.maximum(compte, 1)


def evenements_eveils(proba: np.ndarray, seuil: float = SEUIL_EVEIL) -> list[Ev]:
    return masque_vers_evenements((proba >= seuil).astype(np.int8), duree_min=DUREE_MIN_S, trou_max=TROU_MAX_S)


def mesurer_eveils(nuits: list[NuitEveils], probas: list[np.ndarray], seuil: float = SEUIL_EVEIL, sommeil_predit: bool = False) -> dict:
    """Par événement (appariement un à un par recouvrement) et par personne (micro-éveils par heure de sommeil).

    Référence : micro-éveils qui commencent pendant le sommeil du technicien. Propositions : toutes,
    ou seulement celles qui commencent pendant le sommeil PRÉDIT (`sommeil_predit`, mesure de bout en bout).
    """
    from scipy.stats import spearmanr
    vp = fp = fn = 0
    ref_i, est_i = [], []
    for n, p in zip(nuits, probas):
        som_ref = sommeil_par_seconde(n.stades)[: n.n_sec]
        som_ref = np.concatenate([som_ref, np.zeros(n.n_sec - len(som_ref), dtype=bool)])
        som = (n.p_sommeil >= 0.5) if sommeil_predit else som_ref
        ref = [e for e in masque_vers_evenements(n.y, duree_min=DUREE_MIN_S, trou_max=0) if som_ref[e.debut]]
        pred = [e for e in evenements_eveils(p, seuil) if som[e.debut]]
        m = apparier(pred, ref)
        vp += m["vp"]; fp += m["fp"]; fn += m["fn"]
        ref_i.append(len(ref) / max(som_ref.sum() / 3600, 1e-9)); est_i.append(len(pred) / max(som.sum() / 3600, 1e-9))
    pr, ra = vp / max(vp + fp, 1), vp / max(vp + fn, 1)
    ref_i, est_i = np.array(ref_i), np.array(est_i)
    return {"precision": pr, "rappel": ra, "f1": 2 * pr * ra / max(pr + ra, 1e-9), "n_reference": vp + fn, "n_proposes": vp + fp,
            "index": {"spearman": float(spearmanr(est_i, ref_i).correlation), "erreur_absolue_mediane": float(np.median(np.abs(est_i - ref_i))),
                      "biais": float(np.mean(est_i - ref_i)), "mediane_reference": float(np.median(ref_i))}}
