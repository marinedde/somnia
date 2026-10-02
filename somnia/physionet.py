"""
Lecture des deux jeux PhysioNet en gardant la PERSONNE à côté de chaque époque (tâche 0.2).

- Sleep-EDF (EEG, canal Fpz-Cz, 100 Hz) : époques de 30 s étiquetées par stade.
- Apnea-ECG (ECG, 100 Hz) : segments de 60 s étiquetés apnée / normal.

Les caractéristiques sont calculées avec les MÊMES extracteurs que l'API
(`app/feature_extractor.py`, `app/ecg_features.py`) : ce qui est appris est
exactement ce qui est servi.

Chaque fonction `extraire_*` renvoie un dictionnaire de tableaux alignés :
    X             (n, 16)  caractéristiques
    y             (n,)     étiquette
    personne      (n,)     identifiant de la personne  -> sert au découpage
    enregistrement(n,)     identifiant de l'enregistrement (nuit) -> sert aux comptages
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.ecg_features import ECGFeatureExtractor
from app.feature_extractor import FeatureExtractor
from somnia.subjects import identifiant_personne

log = logging.getLogger(__name__)

FS = 100
EPOCH_LEN = 30                     # s
EPOCH_SAMPLES = FS * EPOCH_LEN     # 3000
SEG_LEN = 60                       # s
SEG_SAMPLES = FS * SEG_LEN         # 6000

# Annotations Sleep-EDF (R&K) -> 5 classes AASM. Stades 3 et 4 fusionnés en N3.
STAGE_MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Movement time": 0,
}
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
APNEA_LABELS = {"N": 0, "A": 1}


# ── Sleep-EDF ─────────────────────────────────────────────────────────────
def paires_sleep_edf(dossier: Path) -> list[tuple[str, Path, Path]]:
    """[(id_enregistrement, psg, hypnogramme)] pour les PSG qui ont un hypnogramme."""
    dossier = Path(dossier)
    psg = {f.name[:7]: f for f in dossier.glob("*PSG.edf")}
    hyp = {f.name[:7]: f for f in dossier.glob("*Hypnogram.edf")}
    communs = sorted(set(psg) & set(hyp))
    oublies = sorted(set(psg) - set(hyp))
    if oublies:
        log.warning("PSG sans hypnogramme, ignorés : %s", oublies)
    return [(k, psg[k], hyp[k]) for k in communs]


def charger_epoques_eeg(psg: Path, hyp: Path, canal: str = "EEG Fpz-Cz"):
    """Toutes les époques annotées d'une nuit : (epochs (n, 3000) en volts, labels (n,))."""
    import mne  # import tardif : lourd, et inutile pour les tests unitaires

    raw = mne.io.read_raw_edf(str(psg), include=[canal], preload=True, verbose="error")
    data = raw.get_data(picks=[canal])[0]
    fs = int(round(raw.info["sfreq"]))
    if fs != FS:
        raise ValueError(f"{psg.name} : {fs} Hz, attendu {FS} Hz")

    epochs, labels = [], []
    for a in mne.read_annotations(str(hyp)):
        stade = STAGE_MAPPING.get(a["description"])
        if stade is None:
            continue
        debut = int(round(a["onset"] * fs))
        for i in range(int(round(a["duration"] / EPOCH_LEN))):
            s, e = debut + i * EPOCH_SAMPLES, debut + (i + 1) * EPOCH_SAMPLES
            if e <= len(data):
                epochs.append(data[s:e])
                labels.append(stade)
    return np.asarray(epochs, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def elaguer_eveil(epochs: np.ndarray, labels: np.ndarray):
    """Retire l'éveil avant le premier endormissement et après le dernier réveil.

    Sleep-EDF enregistre ~20 h par nuit : sans ça, l'éveil fait 68 % des époques.
    Les micro-éveils à l'intérieur de la nuit sont conservés.
    """
    sommeil = np.flatnonzero(labels != 0)
    if len(sommeil) == 0:
        return epochs, labels
    a, b = sommeil[0], sommeil[-1] + 1
    return epochs[a:b], labels[a:b]


def extraire_eeg(dossier: Path, canal: str = "EEG Fpz-Cz") -> dict[str, np.ndarray]:
    extracteur = FeatureExtractor(fs=FS, expected_len=EPOCH_SAMPLES)
    X, y, personne, enregistrement = [], [], [], []
    for ident, psg, hyp in paires_sleep_edf(dossier):
        epochs, labels = charger_epoques_eeg(psg, hyp, canal)
        n_avant = len(labels)
        epochs, labels = elaguer_eveil(epochs, labels)
        log.info("%s : %d époques -> %d après élagage de l'éveil", ident, n_avant, len(labels))
        X.append(extracteur.transform(epochs))
        y.append(labels)
        personne += [identifiant_personne(psg)] * len(labels)
        enregistrement += [ident] * len(labels)
    return _assembler(X, y, personne, enregistrement, FeatureExtractor.feature_names())


# ── Apnea-ECG ─────────────────────────────────────────────────────────────
def enregistrements_apnea_ecg(dossier: Path) -> list[str]:
    """Identifiants ('a01', …) qui ont à la fois le signal (.dat) et les annotations (.apn)."""
    dossier = Path(dossier)
    dat = {f.stem for f in dossier.glob("*.dat")}
    apn = {f.stem for f in dossier.glob("*.apn")}
    manquants = sorted(apn - dat)
    if manquants:
        log.warning("Annotations sans signal, ignorées : %s", manquants)
    return sorted(dat & apn)


def charger_segments_ecg(chemin_sans_extension: Path):
    """Segments d'une minute d'un enregistrement : (segments (n, 6000), labels (n,))."""
    import wfdb

    rec = wfdb.rdrecord(str(chemin_sans_extension))
    ann = wfdb.rdann(str(chemin_sans_extension), "apn")
    ecg = rec.p_signal[:, 0]
    segments, labels = [], []
    for i, symbole in enumerate(ann.symbol):
        if symbole not in APNEA_LABELS:
            continue
        s, e = i * SEG_SAMPLES, (i + 1) * SEG_SAMPLES
        if e <= len(ecg):
            segments.append(ecg[s:e])
            labels.append(APNEA_LABELS[symbole])
    return np.asarray(segments, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def extraire_ecg(dossier: Path) -> dict[str, np.ndarray]:
    extracteur = ECGFeatureExtractor(fs=FS, expected_len=SEG_SAMPLES)
    X, y, personne, enregistrement = [], [], [], []
    for ident in enregistrements_apnea_ecg(dossier):
        try:
            segments, labels = charger_segments_ecg(Path(dossier) / ident)
        except Exception as e:  # fichier corrompu : on le dit, on continue
            log.warning("%s illisible, ignoré : %s", ident, e)
            continue
        log.info("%s : %d segments, %d apnées", ident, len(labels), int(labels.sum()))
        X.append(extracteur.transform(segments))
        y.append(labels)
        personne += [identifiant_personne(ident)] * len(labels)
        enregistrement += [ident] * len(labels)
    return _assembler(X, y, personne, enregistrement, ECGFeatureExtractor.feature_names())


# ── Commun ────────────────────────────────────────────────────────────────
def _assembler(X, y, personne, enregistrement, noms) -> dict[str, np.ndarray]:
    Xa = np.nan_to_num(np.vstack(X).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "X": Xa,
        "y": np.concatenate(y).astype(np.int64),
        "personne": np.asarray(personne),
        "enregistrement": np.asarray(enregistrement),
        "feature_names": np.asarray(noms),
    }


def charger_tableau(chemin: Path) -> dict[str, np.ndarray]:
    """Relit un fichier .npz écrit par python_scripts/prepare_features.py."""
    with np.load(chemin, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
