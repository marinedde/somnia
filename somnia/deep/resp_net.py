"""
Réseau de détection d'événements respiratoires : segmentation à la seconde sur 4 canaux.

Entrée  : fenêtre de 5 minutes, 4 canaux à 10 Hz (flux, thorax, abdomen, SaO2) -> (B, 4, 3000).
Sortie  : pour chaque seconde de la fenêtre, 3 logits (rien, apnée, hypopnée)   -> (B, 3, 300).

Pourquoi cette forme :
  - un lecteur humain regarde plusieurs minutes de flux, de ceintures et de saturation, pas 60 s
    d'ECG ; la désaturation arrive 20 à 40 s APRÈS l'événement, il faut donc du contexte ;
  - on veut des événements avec début et fin : la sortie est une étiquette par seconde, qu'on
    regroupe ensuite en événements (somnia.resp.masque_vers_evenements).

Architecture : deux convolutions qui ramènent 10 Hz à 1 Hz, puis six blocs résiduels à
convolutions DILATÉES (1, 2, 4, 8, 16, 32 s). La dilatation fait voir ~4 minutes de contexte
à chaque seconde sans réduire la résolution. Environ 145 000 paramètres.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from somnia.resp import (  # noqa: F401  (proba_vers_masque et SEUIL_EVENEMENT réexportés)
    FS_RESP, SEUIL_EVENEMENT, apparier, index_par_heure, masque_vers_evenements, normaliser_fenetre,
    proba_vers_masque, sommeil_par_seconde,
)

RESP_DIR = Path(os.environ.get("SHHS_PROCESSED", Path.home() / "data" / "shhs" / "processed")).parent / "processed_resp"
FENETRE_S, PAS_S = 300, 150
N_CANAUX, N_CLASSES = 4, 3


# ── Données ───────────────────────────────────────────────────────────────
@dataclass
class NuitResp:
    ident: str
    personne: str
    signaux: np.ndarray     # (4, n_sec * 10) float32
    y: np.ndarray           # (n_sec,) int8
    stades: np.ndarray      # (n_epoques,)

    @property
    def n_sec(self) -> int:
        return len(self.y)


def nuits_retenues(dossier: Path = RESP_DIR) -> dict[str, str]:
    """{personne: enregistrement} des nuits non exclues de la préparation respiratoire."""
    with open(dossier / "cohorte_resp.csv", encoding="utf-8") as f:
        return {r["personne"]: r["enregistrement"] for r in csv.DictReader(f) if r["exclue"] != "True"}


def charger_nuits(personnes: list[str], dossier: Path = RESP_DIR) -> list[NuitResp]:
    dispo = nuits_retenues(dossier)
    nuits = []
    for p in sorted(personnes):
        if p not in dispo:
            continue
        with np.load(dossier / f"{dispo[p]}.npz", allow_pickle=False) as d:
            nuits.append(NuitResp(dispo[p], p, d["signaux"].astype(np.float32), d["y"].astype(np.int8), d["stades"].astype(np.int8)))
    return nuits


def debuts_de_fenetres(n_sec: int, fenetre: int = FENETRE_S, pas: int = PAS_S) -> list[int]:
    """Débuts (en secondes) des fenêtres qui couvrent toute la nuit, la dernière calée sur la fin."""
    if n_sec <= fenetre:
        return [0]
    d = list(range(0, n_sec - fenetre + 1, pas))
    if d[-1] != n_sec - fenetre:
        d.append(n_sec - fenetre)
    return d


def extraire_fenetre(nuit: NuitResp, debut_s: int, fenetre: int = FENETRE_S):
    """(x (4, fenetre*10) normalisé, y (fenetre,)) ; complété par répétition du bord si la nuit est plus courte."""
    a, b = debut_s * FS_RESP, (debut_s + fenetre) * FS_RESP
    x = nuit.signaux[:, a:b]
    y = nuit.y[debut_s:debut_s + fenetre]
    if x.shape[1] < fenetre * FS_RESP:
        x = np.pad(x, ((0, 0), (0, fenetre * FS_RESP - x.shape[1])), mode="edge")
        y = np.pad(y, (0, fenetre - len(y)))
    return normaliser_fenetre(x), y.astype(np.int64)


class FenetresResp(Dataset):
    """Fenêtres de 5 minutes. À l'entraînement, le début de chaque fenêtre est décalé au hasard (±75 s)."""

    def __init__(self, nuits: list[NuitResp], entrainement: bool = False, graine: int = 42):
        self.nuits, self.entrainement = nuits, entrainement
        self.index = [(i, d) for i, n in enumerate(nuits) for d in debuts_de_fenetres(n.n_sec)]
        self.rng = np.random.default_rng(graine)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, d = self.index[k]
        nuit = self.nuits[i]
        if self.entrainement:
            d = int(np.clip(d + self.rng.integers(-PAS_S // 2, PAS_S // 2 + 1), 0, max(0, nuit.n_sec - FENETRE_S)))
        x, y = extraire_fenetre(nuit, d)
        return torch.from_numpy(x), torch.from_numpy(y)


# ── Modèle ────────────────────────────────────────────────────────────────
class BlocDilate(nn.Module):
    def __init__(self, c: int, dilatation: int, k: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, k, padding=dilatation * (k // 2), dilation=dilatation, bias=False),
            nn.BatchNorm1d(c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.net(x)          # connexion résiduelle : le bloc apprend une correction


class ReseauEvenements(nn.Module):
    def __init__(self, canaux: int = N_CANAUX, largeur: int = 64, dilatations=(1, 2, 4, 8, 16, 32), dropout: float = 0.1):
        super().__init__()
        self.entree = nn.Sequential(                                   # 10 Hz -> 1 Hz
            nn.Conv1d(canaux, 32, 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, largeur, 9, stride=5, padding=4, bias=False), nn.BatchNorm1d(largeur), nn.ReLU(inplace=True),
        )
        self.contexte = nn.Sequential(*[BlocDilate(largeur, d) for d in dilatations])
        self.tete = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(largeur, N_CLASSES, 1))

    def forward(self, x):                                              # (B, 4, 3000)
        return self.tete(self.contexte(self.entree(x)))                # (B, 3, 300)


# ── Inférence sur une nuit et mesures ─────────────────────────────────────
@torch.no_grad()
def proba_nuit(modele: nn.Module, nuit: NuitResp, dev: torch.device, batch: int = 64) -> np.ndarray:
    """Probabilités (n_sec, 3) : les fenêtres se recouvrent de moitié, on moyenne."""
    modele.eval()
    debuts = debuts_de_fenetres(nuit.n_sec)
    somme = np.zeros((nuit.n_sec, N_CLASSES), dtype=np.float64)
    compte = np.zeros(nuit.n_sec, dtype=np.float64)
    for i in range(0, len(debuts), batch):
        lot = debuts[i:i + batch]
        x = torch.from_numpy(np.stack([extraire_fenetre(nuit, d)[0] for d in lot])).to(dev)
        p = torch.softmax(modele(x), dim=1).permute(0, 2, 1).float().cpu().numpy()      # (B, 300, 3)
        for d, pi in zip(lot, p):
            n = min(FENETRE_S, nuit.n_sec - d)
            somme[d:d + n] += pi[:n]; compte[d:d + n] += 1
    return somme / np.maximum(compte, 1)[:, None]


def mesurer_nuits(nuits: list[NuitResp], masques_pred: list[np.ndarray]) -> dict:
    """Toutes les mesures : par seconde, par événement (deux exigences), par classe, par personne."""
    from sklearn.metrics import f1_score

    y_tout = np.concatenate([n.y for n in nuits]); p_tout = np.concatenate(masques_pred)
    tot = {"large": dict(vp=0, fp=0, fn=0), "strict": dict(vp=0, fp=0, fn=0)}
    par_personne, rappel_classe = [], {1: [0, 0], 2: [0, 0]}
    for n, mp in zip(nuits, masques_pred):
        ref, pred = masque_vers_evenements(n.y), masque_vers_evenements(mp)
        for nom, iou in (("large", 0.0), ("strict", 0.3)):
            m = apparier(pred, ref, iou)
            for k in ("vp", "fp", "fn"):
                tot[nom][k] += m[k]
        for c in (1, 2):                                               # rappel par type d'événement de référence
            ref_c = [e for e in ref if e.classe == c]
            rappel_classe[c][0] += apparier(pred, ref_c)["vp"]; rappel_classe[c][1] += len(ref_c)
        sommeil = sommeil_par_seconde(n.stades)[: n.n_sec]
        par_personne.append({"personne": n.personne, "index_reference": index_par_heure(ref, sommeil),
                             "index_estime": index_par_heure(pred, sommeil), "n_ref": len(ref), "n_pred": len(pred)})

    def prf(d):
        p = d["vp"] / (d["vp"] + d["fp"]) if d["vp"] + d["fp"] else 0.0
        r = d["vp"] / (d["vp"] + d["fn"]) if d["vp"] + d["fn"] else 0.0
        return {"precision": p, "rappel": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, **d}
    ref_i = np.array([x["index_reference"] for x in par_personne]); est_i = np.array([x["index_estime"] for x in par_personne])
    from scipy.stats import spearmanr
    return {
        "par_seconde": {"f1_evenement": float(f1_score(y_tout > 0, p_tout > 0)),
                        "f1_macro_3_classes": float(f1_score(y_tout, p_tout, average="macro", labels=[0, 1, 2], zero_division=0)),
                        "part_reference": float((y_tout > 0).mean()), "part_predite": float((p_tout > 0).mean())},
        "par_evenement": {"recouvrement": prf(tot["large"]), "iou_0.3": prf(tot["strict"]),
                          "rappel_apnees": rappel_classe[1][0] / max(rappel_classe[1][1], 1),
                          "rappel_hypopnees": rappel_classe[2][0] / max(rappel_classe[2][1], 1),
                          "n_apnees_ref": rappel_classe[1][1], "n_hypopnees_ref": rappel_classe[2][1]},
        "par_personne": {"n": len(par_personne), "spearman": float(spearmanr(est_i, ref_i).correlation),
                         "erreur_absolue_mediane": float(np.median(np.abs(est_i - ref_i))),
                         "biais": float(np.mean(est_i - ref_i)),
                         "limites_accord_95": [float(np.mean(est_i - ref_i) - 1.96 * np.std(est_i - ref_i)),
                                               float(np.mean(est_i - ref_i) + 1.96 * np.std(est_i - ref_i))],
                         "detail": par_personne},
    }


def poids_des_classes(nuits: list[NuitResp], plafond: float = 20.0) -> torch.Tensor:
    y = np.concatenate([n.y for n in nuits])
    c = np.bincount(y, minlength=N_CLASSES).astype(np.float64); c[c == 0] = 1
    w = np.minimum(len(y) / (N_CLASSES * c), plafond)
    return torch.tensor(w / w.min(), dtype=torch.float32)
