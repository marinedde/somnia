"""
Analyse d'une nuit entière : hypnogramme, indices, et file de relecture triée par incertitude.

C'est l'idée de départ de Somnia : un médecin met quatre heures à lire une nuit. Le modèle ne
remplace pas cette lecture ; il la trie. Pour chaque époque il donne un stade, une probabilité,
et une confiance calibrée ; les époques les moins sûres forment la file de relecture.

Entrée  : les époques EEG d'une nuit (n, 3000) à 100 Hz, en volts (comme un EDF lu par MNE).
Sortie  : un dictionnaire sérialisable (voir `analyser_nuit`).

Les étiquettes « non validé » sont systématiques : rien ici n'est un diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STADES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
DUREE_EPOQUE_S = 30


@dataclass
class ResultatNuit:
    stades: np.ndarray          # (n,) codes 0-4 prédits
    proba: np.ndarray           # (n, 5) probabilités calibrées
    confiance: np.ndarray       # (n,) max des probabilités

    @property
    def n(self) -> int:
        return len(self.stades)


def indices_de_nuit(stades: np.ndarray, duree_epoque_s: int = DUREE_EPOQUE_S) -> dict:
    """Les chiffres qu'un compte rendu de sommeil donne toujours (sur les stades prédits)."""
    n = len(stades)
    sommeil = np.isin(stades, [1, 2, 3, 4])
    idx = np.flatnonzero(sommeil)
    tst_min = sommeil.sum() * duree_epoque_s / 60
    periode = (idx[-1] - idx[0] + 1) if len(idx) else 0
    return {
        "temps_enregistrement_min": round(n * duree_epoque_s / 60, 1),
        "temps_sommeil_total_min": round(float(tst_min), 1),
        "efficacite_sommeil_pct": round(float(sommeil.mean() * 100), 1) if n else 0.0,
        "latence_endormissement_min": round(float(idx[0] * duree_epoque_s / 60), 1) if len(idx) else None,
        "eveil_intra_sommeil_min": round(float((periode - len(idx)) * duree_epoque_s / 60), 1) if len(idx) else None,
        "parts_pct": {STADES[k]: round(float((stades == k).mean() * 100), 1) for k in STADES},
        "n_epoques": int(n),
    }


def file_de_relecture(confiance: np.ndarray, stades: np.ndarray, seuil: float = 0.6,
                      duree_epoque_s: int = DUREE_EPOQUE_S) -> dict:
    """Les époques sous le seuil de confiance, groupées en segments contigus, triées de la moins sûre à la plus sûre."""
    douteuses = confiance < seuil
    segments, debut = [], None
    for i, d in enumerate(np.append(douteuses, False)):
        if d and debut is None:
            debut = i
        elif not d and debut is not None:
            seg = slice(debut, i)
            segments.append({
                "debut_epoque": int(debut), "fin_epoque": int(i - 1),
                "debut_hhmm": _hhmm(debut * duree_epoque_s), "duree_min": round((i - debut) * duree_epoque_s / 60, 1),
                "confiance_min": round(float(confiance[seg].min()), 3),
                "stades_proposes": "".join(STADES[int(s)] for s in stades[seg][:12]) + ("…" if i - debut > 12 else ""),
            })
            debut = None
    segments.sort(key=lambda s: s["confiance_min"])
    n_douteuses = int(douteuses.sum())
    return {
        "seuil_confiance": seuil,
        "n_epoques_a_relire": n_douteuses,
        "part_a_relire_pct": round(float(douteuses.mean() * 100), 1) if len(confiance) else 0.0,
        "temps_relecture_estime_min": round(n_douteuses * duree_epoque_s / 60, 1),
        "segments": segments,
    }


def _hhmm(secondes: float) -> str:
    h, m = divmod(int(secondes // 60), 60)
    return f"{h:02d}:{m:02d}"


def analyser_nuit(epoques_volts: np.ndarray, predire_proba, temperature: float = 1.0, seuil: float = 0.6) -> dict:
    """Tout le compte rendu d'une nuit à partir d'une fonction `predire_proba(epoques) -> logits (n, 5)`.

    `temperature` : celle ajustée sur la validation (calibration). `seuil` : sous cette confiance,
    l'époque part en relecture.
    """
    logits = np.asarray(predire_proba(epoques_volts), dtype=np.float64) / temperature
    z = logits - logits.max(axis=1, keepdims=True)
    proba = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    stades = proba.argmax(axis=1)
    confiance = proba.max(axis=1)
    return {
        "statut": "NON VALIDÉ — proposition automatique, à relire par un professionnel formé",
        "hypnogramme": [STADES[int(s)] for s in stades],
        "confiance": [round(float(c), 3) for c in confiance],
        "indices": indices_de_nuit(stades),
        "relecture": file_de_relecture(confiance, stades, seuil),
    }
