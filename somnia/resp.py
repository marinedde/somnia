"""
Détection d'événements respiratoires : préparation des signaux, étiquettes à la seconde,
passage masque <-> événements, appariement, index par personne, et la référence par désaturation.

Pourquoi ce module : ce qui prend du temps à un lecteur de polysomnographie, c'est de marquer
chaque apnée et chaque hypopnée avec son début et sa fin. Un modèle utile doit donc sortir des
ÉVÉNEMENTS (début, durée, type), à partir de ce que regarde un humain : le flux, les ceintures
thoracique et abdominale, et la saturation. L'ECG seul ne suffit pas (voir RESULTATS_CNN.md).

Conventions :
  - signaux respiratoires à 10 Hz (fréquence native de SHHS pour le flux et les ceintures) ;
    la saturation, native à 1 Hz, est répétée (maintien) à 10 Hz, jamais interpolée par FFT ;
  - étiquettes à 1 Hz : 0 rien, 1 apnée (obstructive, centrale, mixte), 2 hypopnée ;
  - un événement prédit est une suite de secondes positives d'au moins 10 s (définition clinique),
    les trous de moins de 3 s étant comblés.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from somnia.shhs import Annotations, Evenement

FS_RESP = 10                       # Hz
CLASSES = {0: "rien", 1: "apnée", 2: "hypopnée"}
APNEES = ("obstructive apnea", "central apnea", "mixed apnea")
HYPOPNEES = ("hypopnea",)
DUREE_MIN_S = 10
TROU_MAX_S = 3

# Noms possibles des canaux (comparés sans casse ni espaces). Le flux s'appelle NEW AIR ou AIRFLOW.
CANAUX_RESP = {
    "flux": ["NEW AIR", "AIRFLOW", "AIR", "NEWAIR"],
    "thorax": ["THOR RES", "THOR"],
    "abdomen": ["ABDO RES", "ABDO"],
    "sao2": ["SaO2", "SpO2"],
}
ORDRE_CANAUX = ("flux", "thorax", "abdomen", "sao2")


# ── Étiquettes ────────────────────────────────────────────────────────────
def classe_evenement(nom: str) -> int:
    n = nom.lower()
    if n in APNEES:
        return 1
    if n in HYPOPNEES:
        return 2
    return 0


def etiquettes_par_seconde(ann: Annotations, n_secondes: int | None = None) -> np.ndarray:
    """Un entier par seconde : 0 rien, 1 apnée, 2 hypopnée. L'apnée l'emporte si deux événements se recouvrent."""
    n = int(n_secondes if n_secondes is not None else ann.n_epoques * ann.duree_epoque)
    y = np.zeros(n, dtype=np.int8)
    for ev in sorted(ann.evenements, key=lambda e: classe_evenement(e.nom), reverse=True):   # hypopnées d'abord, apnées par-dessus
        c = classe_evenement(ev.nom)
        if c == 0:
            continue
        a, b = max(0, int(np.floor(ev.debut))), min(n, int(np.ceil(ev.fin)))
        if b > a:
            y[a:b] = np.where(y[a:b] == 1, 1, c) if c == 2 else 1
    return y


def sommeil_par_seconde(stades: np.ndarray, duree_epoque: float = 30) -> np.ndarray:
    """Booléen par seconde : vrai si l'époque est un stade de sommeil scoré."""
    return np.repeat(np.isin(stades, [1, 2, 3, 4]), int(duree_epoque))


# ── Masque <-> événements ─────────────────────────────────────────────────
@dataclass(frozen=True)
class Ev:
    debut: int      # seconde
    fin: int        # seconde exclue
    classe: int     # 1 apnée, 2 hypopnée

    @property
    def duree(self) -> int:
        return self.fin - self.debut


def masque_vers_evenements(y: np.ndarray, duree_min: int = DUREE_MIN_S, trou_max: int = TROU_MAX_S) -> list[Ev]:
    """Suites de secondes positives -> événements. Comble les trous courts, retire les événements trop brefs.

    La classe d'un événement est la classe majoritaire de ses secondes positives.
    """
    y = np.asarray(y)
    pos = (y > 0).astype(np.int8)
    if trou_max > 0 and pos.any():
        # combler les trous : un zéro entouré de positifs à moins de trou_max secondes
        idx = np.flatnonzero(pos)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= trou_max + 1:
                pos[a:b] = 1
    bords = np.diff(np.concatenate([[0], pos, [0]]))
    debuts, fins = np.flatnonzero(bords == 1), np.flatnonzero(bords == -1)
    evs = []
    for a, b in zip(debuts, fins):
        if b - a < duree_min:
            continue
        seg = y[a:b][y[a:b] > 0]
        classe = 1 if (seg == 1).sum() >= (seg == 2).sum() else 2
        evs.append(Ev(int(a), int(b), int(classe)))
    return evs


def evenements_de_reference(ann: Annotations) -> list[Ev]:
    """Les apnées et hypopnées annotées, arrondies à la seconde, telles que le technicien les a marquées."""
    evs = []
    for e in ann.evenements:
        c = classe_evenement(e.nom)
        if c:
            evs.append(Ev(int(np.floor(e.debut)), int(np.ceil(e.fin)), c))
    return sorted(evs, key=lambda e: e.debut)


def apparier(pred: list[Ev], ref: list[Ev], iou_min: float = 0.0) -> dict:
    """Appariement un-à-un par recouvrement temporel (glouton, par début croissant).

    iou_min = 0 : tout recouvrement compte (convention courante en détection d'apnées).
    iou_min = 0,3 : exigeant sur les bornes. Renvoie vrais positifs, faux positifs, faux négatifs, F1.
    """
    pred, ref = sorted(pred, key=lambda e: e.debut), sorted(ref, key=lambda e: e.debut)
    pris, vp, j0 = set(), 0, 0
    for p in pred:
        while j0 < len(ref) and ref[j0].fin <= p.debut:
            j0 += 1
        meilleur, meilleur_iou = None, 0.0
        for j in range(j0, len(ref)):
            r = ref[j]
            if r.debut >= p.fin:
                break
            if j in pris:
                continue
            inter = min(p.fin, r.fin) - max(p.debut, r.debut)
            if inter <= 0:
                continue
            iou = inter / (max(p.fin, r.fin) - min(p.debut, r.debut))
            if iou > meilleur_iou:
                meilleur, meilleur_iou = j, iou
        if meilleur is not None and meilleur_iou > iou_min:
            pris.add(meilleur); vp += 1
    fp, fn = len(pred) - vp, len(ref) - vp
    precision = vp / (vp + fp) if vp + fp else 0.0
    rappel = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
    return {"vp": vp, "fp": fp, "fn": fn, "precision": precision, "rappel": rappel, "f1": f1}


def index_par_heure(evenements: list[Ev], sommeil: np.ndarray) -> float:
    """Événements dont le début tombe pendant le sommeil, par heure de sommeil (définition de l'IAH)."""
    heures = float(sommeil.sum()) / 3600
    if heures <= 0:
        return float("nan")
    n = sum(1 for e in evenements if e.debut < len(sommeil) and sommeil[e.debut])
    return n / heures


# ── Saturation : nettoyage et référence par désaturation ──────────────────
def nettoyer_sao2(sao2: np.ndarray) -> np.ndarray:
    """Capteur décollé = valeurs absurdes (< 50 %). On les remplace par la dernière valeur valide."""
    x = np.asarray(sao2, dtype=np.float64).copy()
    invalide = (x < 50) | (x > 100) | ~np.isfinite(x)
    if invalide.all():
        return np.full_like(x, 95.0)
    idx = np.where(~invalide, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    premier = int(np.flatnonzero(~invalide)[0])
    x[:premier] = x[premier]
    return x


def index_de_desaturation(sao2_1hz: np.ndarray, sommeil: np.ndarray, chute: float = 3.0,
                          fenetre_s: int = 120) -> float:
    """ODI : nombre de chutes de saturation d'au moins `chute` points par heure de sommeil.

    Référence classique et forte pour estimer l'index d'apnées : chaque événement qui compte
    cliniquement est suivi d'une désaturation. Une chute est comptée quand la saturation passe
    `chute` points sous le maximum des `fenetre_s` secondes précédentes ; elle se termine quand
    la saturation remonte à moins d'un point de ce maximum.
    """
    x = nettoyer_sao2(sao2_1hz)
    n = min(len(x), len(sommeil))
    x, sommeil = x[:n], sommeil[:n]
    # maximum glissant sur la fenêtre précédente
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([np.full(fenetre_s - 1, x[0]), x])
    base = sliding_window_view(pad, fenetre_s).max(axis=1)
    en_chute, compte = False, 0
    for t in range(n):
        d = base[t] - x[t]
        if not en_chute and d >= chute:
            en_chute = True
            if sommeil[t]:
                compte += 1
        elif en_chute and d < 1.0:
            en_chute = False
    heures = float(sommeil.sum()) / 3600
    return compte / heures if heures > 0 else float("nan")


# ── Lecture des canaux ────────────────────────────────────────────────────
def _normaliser_nom(nom: str) -> str:
    return nom.replace(" ", "").lower()


def trouver(noms: list[str], candidats: list[str]) -> str | None:
    table = {_normaliser_nom(n): n for n in noms}
    for c in candidats:
        if _normaliser_nom(c) in table:
            return table[_normaliser_nom(c)]
    return None


def vers_10hz(x: np.ndarray, fs: float) -> np.ndarray:
    """Ramène un canal à 10 Hz : maintien pour un signal plus lent (SaO2 à 1 Hz), rééchantillonnage sinon."""
    fs = int(round(fs))
    if fs == FS_RESP:
        return x
    if fs < FS_RESP and FS_RESP % fs == 0:
        return np.repeat(x, FS_RESP // fs)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(fs, FS_RESP)
    return resample_poly(x, FS_RESP // g, fs // g)


def normaliser_fenetre(x: np.ndarray) -> np.ndarray:
    """(4, L) -> (4, L) float32. Flux et ceintures : centrés-réduits sur la fenêtre (unités arbitraires).
    Saturation : (SaO2 − 95) / 5, bornée : 100 % -> 1, 90 % -> −1, 80 % -> −3."""
    out = np.empty_like(x, dtype=np.float32)
    for i in range(3):
        m, s = x[i].mean(), x[i].std()
        out[i] = (x[i] - m) / (s + 1e-6)
    out[3] = np.clip((x[3] - 95.0) / 5.0, -6, 1)
    return out
