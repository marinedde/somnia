"""
Préparation d'une nuit SHHS : EDF + XML -> un fichier .npz par nuit (étape 2).

Pour chaque nuit :
  - EEG C4-A1 (canal `EEG`) et ECG, lus à leur fréquence native (125 Hz) ;
  - rééchantillonnés à 100 Hz (rapport 4/5, `resample_poly` : filtre anti-repliement inclus),
    pour garder la compatibilité avec PhysioNet et permettre la validation externe ;
  - découpés en époques de 30 s alignées sur les stades du XML ;
  - étiquette apnée par époque : voir `somnia.shhs.etiquettes_apnee` (≥ 10 s couvertes).

Décision de prétraitement (2 octobre 2026) : pas d'autre filtre que l'anti-repliement, comme
pour PhysioNet. Un filtrage passe-bande est un réglage à comparer plus tard, pas un prérequis.

Règles d'exclusion, écrites AVANT de regarder les résultats (diagramme de cohorte) :
  - fichier EDF ou XML illisible ;
  - canal EEG ou ECG absent ;
  - moins de 4 h de sommeil scoré ;
  - ECG plat (écart-type nul) sur plus de 50 % des époques ;
  - fréquence native de l'EEG ou de l'ECG sous 100 Hz (ajouté le 3 octobre après audit ;
    shhs_v1, déjà figé, contient une nuit à ECG 50 Hz dans l'entraînement).

Audit du 3 octobre : une époque dont le signal manque (annotations plus longues que l'EDF)
est marquée stade -1 et apnée 0, donc jamais apprise. Les signaux sont stockés en float16
(µV, mV) ; le calcul des caractéristiques repasse en float64 dans les extracteurs.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from somnia.shhs import Annotations, etiquettes_apnee, lire_annotations

log = logging.getLogger(__name__)

FS_CIBLE = 100
DUREE_EPOQUE = 30
N_POINTS = FS_CIBLE * DUREE_EPOQUE   # 3000
SOMMEIL_MIN_H = 4.0
ECG_PLAT_MAX = 0.5
FS_MIN_NATIF = 100          # Hz : un ECG natif à 50 Hz dégrade la détection des battements
STADES_SOMMEIL = (1, 2, 3, 4)

# Noms possibles d'un même canal, comparés sans casse ni espaces
CANAUX = {
    "eeg": ["EEG", "EEG C4-A1", "C4-A1"],          # C4-A1 ; `EEG(sec)` est C3-A2
    "ecg": ["ECG", "EKG"],
}


def _normaliser(nom: str) -> str:
    return nom.replace(" ", "").lower()


def trouver_canal(noms_disponibles: list[str], candidats: list[str]) -> str | None:
    table = {_normaliser(n): n for n in noms_disponibles}
    for c in candidats:
        if _normaliser(c) in table:
            return table[_normaliser(c)]
    return None


def lire_canal_natif(chemin_edf: Path, canal: str) -> tuple[np.ndarray, float]:
    """Signal d'un canal à sa fréquence NATIVE, en volts (MNE convertit uV et mV en V)."""
    import mne

    raw = mne.io.read_raw_edf(str(chemin_edf), include=[canal], preload=True, verbose="error")
    fs = float(raw.info["sfreq"])
    return raw.get_data(picks=[canal])[0].astype(np.float64), fs


def reechantillonner(x: np.ndarray, fs_source: float, fs_cible: int = FS_CIBLE) -> np.ndarray:
    """125 Hz -> 100 Hz par le rapport 4/5. Les autres rapports entiers sont acceptés aussi."""
    from math import gcd

    src, dst = int(round(fs_source)), int(fs_cible)
    if src == dst:
        return x
    g = gcd(src, dst)
    return resample_poly(x, up=dst // g, down=src // g)


def decouper_en_epoques(x: np.ndarray, n_epoques: int, n_points: int = N_POINTS):
    """Renvoie (epochs (n_epoques, n_points), incomplet (n_epoques,) booléen).

    Les époques pour lesquelles le signal manque (en tout ou partie) sont complétées par des
    zéros ET marquées `incomplet=True` : l'appelant doit les retirer des étiquettes.
    """
    total = n_epoques * n_points
    n_complets = min(len(x), total) // n_points
    incomplet = np.arange(n_epoques) >= n_complets
    if len(x) < total:
        log.warning("signal plus court que les annotations : %d < %d points, %d époque(s) incomplète(s)",
                    len(x), total, int(incomplet.sum()))
        x = np.concatenate([x, np.zeros(total - len(x))])
    return x[:total].reshape(n_epoques, n_points), incomplet


def construire_nuit(eeg_v: np.ndarray, fs_eeg: float, ecg_v: np.ndarray, fs_ecg: float,
                    ann: Annotations) -> dict[str, np.ndarray]:
    """Cœur de la préparation, sans fichier : signaux en volts -> époques alignées sur les stades.

    Renvoie eeg (µV) et ecg (mV) en float32, de forme (n_epoques, 3000), stades (-1 pour non
    scoré ou incomplet), apnee (0 sur les époques incomplètes), incomplet (booléen).
    """
    eeg, inc1 = decouper_en_epoques(reechantillonner(eeg_v, fs_eeg), ann.n_epoques)
    ecg, inc2 = decouper_en_epoques(reechantillonner(ecg_v, fs_ecg), ann.n_epoques)
    incomplet = inc1 | inc2
    stades = ann.stades.copy()
    stades[incomplet] = -1
    apnee = etiquettes_apnee(ann)
    apnee[incomplet] = 0
    return {
        "eeg": (eeg * 1e6).astype(np.float32),
        "ecg": (ecg * 1e3).astype(np.float32),
        "stades": stades.astype(np.int64),
        "apnee": apnee.astype(np.int64),
        "incomplet": incomplet,
    }


def fenetres_60s(stades: np.ndarray, apnee: np.ndarray, aberrant: np.ndarray | None = None):
    """Fenêtres d'une minute = paires d'époques (2k, 2k+1), pour comparer à Apnea-ECG.

    Renvoie (garde, y, ab) indexés par k : `garde` vrai si les DEUX époques sont du sommeil
    scoré (une apnée se définit pendant le sommeil), `y` = 1 si l'une des deux est « apnée »,
    `ab` vrai si l'une des deux est aberrante. La fenêtre k couvre les points [6000k, 6000(k+1)).
    """
    m = len(stades) // 2
    s2 = np.asarray(stades[: 2 * m]).reshape(m, 2)
    garde = np.all(np.isin(s2, STADES_SOMMEIL), axis=1)
    y = np.asarray(apnee[: 2 * m]).reshape(m, 2).max(axis=1)
    if aberrant is None:
        ab = np.zeros(m, dtype=bool)
    else:
        ab = np.asarray(aberrant[: 2 * m]).reshape(m, 2).any(axis=1)
    return garde, y, ab


@dataclass
class ResumeNuit:
    enregistrement: str
    personne: str
    n_epoques: int
    temps_sommeil_h: float
    n_apnee_epoques: int
    part_ecg_plat: float
    canal_eeg: str | None
    canal_ecg: str | None
    exclue: bool
    motif_exclusion: str | None
    n_incompletes: int = 0


def preparer_nuit(chemin_edf: Path, chemin_xml: Path, sortie: Path) -> ResumeNuit:
    """Écrit `sortie` (npz) si la nuit est retenue ; renvoie le résumé dans tous les cas."""
    from somnia.subjects import identifiant_personne

    ident = chemin_edf.stem
    personne = identifiant_personne(chemin_edf)

    def exclue(motif: str, **kw) -> ResumeNuit:
        log.warning("%s exclue : %s", ident, motif)
        base = dict(enregistrement=ident, personne=personne, n_epoques=0, temps_sommeil_h=0.0,
                    n_apnee_epoques=0, part_ecg_plat=0.0, canal_eeg=None, canal_ecg=None)
        base.update(kw)
        return ResumeNuit(**base, exclue=True, motif_exclusion=motif)

    try:
        ann: Annotations = lire_annotations(chemin_xml)
    except Exception as e:
        return exclue(f"XML illisible ({type(e).__name__}: {e})"[:200])

    try:
        import mne
        noms = mne.io.read_raw_edf(str(chemin_edf), preload=False, verbose="error").ch_names
    except Exception as e:
        return exclue(f"EDF illisible ({type(e).__name__})")

    canal_eeg, canal_ecg = trouver_canal(noms, CANAUX["eeg"]), trouver_canal(noms, CANAUX["ecg"])
    if canal_eeg is None or canal_ecg is None:
        return exclue(f"canal manquant (EEG={canal_eeg}, ECG={canal_ecg})")

    temps_sommeil_h = float(np.isin(ann.stades, STADES_SOMMEIL).sum() * ann.duree_epoque / 3600)
    if temps_sommeil_h < SOMMEIL_MIN_H:
        return exclue(f"sommeil scoré {temps_sommeil_h:.1f} h < {SOMMEIL_MIN_H} h",
                      temps_sommeil_h=temps_sommeil_h, canal_eeg=canal_eeg, canal_ecg=canal_ecg)

    eeg_v, fs_eeg = lire_canal_natif(chemin_edf, canal_eeg)
    ecg_v, fs_ecg = lire_canal_natif(chemin_edf, canal_ecg)
    if min(fs_eeg, fs_ecg) < FS_MIN_NATIF:
        return exclue(f"fréquence native {min(fs_eeg, fs_ecg):.0f} Hz < {FS_MIN_NATIF} Hz",
                      temps_sommeil_h=temps_sommeil_h, canal_eeg=canal_eeg, canal_ecg=canal_ecg)

    nuit = construire_nuit(eeg_v, fs_eeg, ecg_v, fs_ecg, ann)
    part_plat = float((nuit["ecg"].std(axis=1) < 1e-6).mean())
    if part_plat > ECG_PLAT_MAX:
        return exclue(f"ECG plat sur {part_plat:.0%} des époques", temps_sommeil_h=temps_sommeil_h,
                      part_ecg_plat=part_plat, canal_eeg=canal_eeg, canal_ecg=canal_ecg)

    sortie.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sortie,
        eeg=nuit["eeg"].astype(np.float16),   # µV : plage float16 confortable (±125 µV)
        ecg=nuit["ecg"].astype(np.float16),   # mV : ±1,25 mV ; pas de 0,001 mV entre 1 et 2 mV, acceptable
        stades=nuit["stades"].astype(np.int8),
        apnee=nuit["apnee"].astype(np.int8),
        meta=np.array([ident, personne, canal_eeg, canal_ecg, f"{FS_CIBLE}", "eeg:uV", "ecg:mV"]),
    )
    return ResumeNuit(ident, personne, int(ann.n_epoques), round(temps_sommeil_h, 2),
                      int(nuit["apnee"].sum()), round(part_plat, 3), canal_eeg, canal_ecg, False, None,
                      int(nuit["incomplet"].sum()))


def charger_nuit(chemin_npz: Path) -> dict:
    """Relit une nuit : eeg et ecg en float32 (µV, mV), stades et apnée en int64."""
    with np.load(chemin_npz, allow_pickle=False) as f:
        return {
            "eeg": f["eeg"].astype(np.float32),
            "ecg": f["ecg"].astype(np.float32),
            "stades": f["stades"].astype(np.int64),
            "apnee": f["apnee"].astype(np.int64),
            "meta": dict(zip(["enregistrement", "personne", "canal_eeg", "canal_ecg", "fs", "u_eeg", "u_ecg"],
                             f["meta"].tolist())),
        }


def resume_vers_dict(r: ResumeNuit) -> dict:
    return asdict(r)
