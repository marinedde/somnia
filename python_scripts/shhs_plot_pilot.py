#!/usr/bin/env python3
"""
Tâche 1.9 — Regarder les signaux SHHS : 30 s d'EEG et d'ECG pour chaque stade, et une apnée.

Pour un enregistrement :
  - 5 figures « stade » : 30 s d'EEG (C4-A1) et d'ECG pour une époque de chaque stade présent ;
  - 1 figure « apnée » : 90 s d'ECG, de flux et de SaO2 autour d'une apnée obstructive.

Usage :
    python python_scripts/shhs_plot_pilot.py                    # premier enregistrement trouvé
    python python_scripts/shhs_plot_pilot.py --id shhs1-200001
    python python_scripts/shhs_plot_pilot.py --out docs/figures_pilote

Les figures restent locales par défaut (docs/figures_pilote/ est ignoré par git : ce sont
des signaux SHHS, pas des signaux publics).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.shhs import lire_annotations  # noqa: E402

DEFAULT_DIR = Path(os.environ.get("SHHS_DIR", Path.home() / "data" / "shhs" / "raw"))
NOMS = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
CANAUX_PREFERES = {"eeg": ["EEG", "EEG (sec)"], "ecg": ["ECG"], "flux": ["New Air", "Airflow"], "sao2": ["SaO2"]}


def _trouver(raw, candidats):
    for c in candidats:
        if c in raw.ch_names:
            return c
    return None


def _segment(raw, canal, debut_s, duree_s):
    """Renvoie (t, signal) lus dans l'EDF, à la fréquence NATIVE du canal (lecture directe)."""
    import mne  # noqa: F401  (déjà importé par l'appelant)

    idx = raw.ch_names.index(canal)
    fs = raw.info["sfreq"]
    a, b = int(debut_s * fs), int((debut_s + duree_s) * fs)
    x = raw.get_data(picks=[idx], start=a, stop=b)[0]
    return np.arange(len(x)) / fs + debut_s, x


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--id", default=None, help="identifiant, ex. shhs1-200001")
    parser.add_argument("--out", default=str(ROOT / "docs" / "figures_pilote"))
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    racine = Path(args.dir).expanduser()
    edfs = sorted(racine.rglob(f"{args.id or '*'}.edf"))
    if not edfs:
        print(f"Aucun EDF {args.id or ''} sous {racine}")
        return 1
    edf = edfs[0]
    xml = next(racine.rglob(f"{edf.stem}-nsrr.xml"), None)
    if xml is None:
        print(f"Pas d'annotations pour {edf.stem}")
        return 1
    out = Path(args.out) / edf.stem
    out.mkdir(parents=True, exist_ok=True)

    ann = lire_annotations(xml)
    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose="error")
    print(f"{edf.stem} : {len(raw.ch_names)} canaux, MNE à {raw.info['sfreq']} Hz (commune), {ann.n_epoques} époques")
    print("  canaux :", raw.ch_names)
    eeg, ecg = _trouver(raw, CANAUX_PREFERES["eeg"]), _trouver(raw, CANAUX_PREFERES["ecg"])
    flux, sao2 = _trouver(raw, CANAUX_PREFERES["flux"]), _trouver(raw, CANAUX_PREFERES["sao2"])
    if eeg is None or ecg is None:
        print("  EEG ou ECG introuvable : adapte CANAUX_PREFERES avec les noms ci-dessus.")
        return 1

    # 1. Une époque par stade, prise au milieu de la nuit pour éviter les bords
    d = ann.duree_epoque
    for code, nom in NOMS.items():
        idx = np.flatnonzero(ann.stades == code)
        if len(idx) == 0:
            print(f"  {nom}: aucune époque")
            continue
        i = int(idx[len(idx) // 2])
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        for ax, canal, titre in zip(axes, (eeg, ecg), ("EEG " + eeg, "ECG")):
            t, x = _segment(raw, canal, i * d, d)
            ax.plot(t, x * 1e6 if "EEG" in titre else x, lw=0.6, color="#2E4057")
            ax.set_ylabel("µV" if "EEG" in titre else raw._orig_units.get(canal, ""))
            ax.set_title(titre, loc="left", fontsize=9)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("secondes depuis le début de l'enregistrement")
        fig.suptitle(f"{edf.stem} — époque {i} — {nom}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"stade_{nom}.png", dpi=130)
        plt.close(fig)
        print(f"  {nom}: époque {i} ({len(idx)} au total)")

    # 2. Une apnée obstructive avec 30 s avant et après
    apnees = [e for e in ann.respiratoires() if e.nom.lower() == "obstructive apnea"]
    if not apnees:
        print("  aucune apnée obstructive annotée")
    else:
        ev = apnees[len(apnees) // 2]
        debut, duree = max(ev.debut - 30, 0), ev.duree + 60
        canaux = [(ecg, "ECG"), (flux, "Flux (thermistance)"), (sao2, "SaO2 (%)")]
        canaux = [(c, t) for c, t in canaux if c]
        fig, axes = plt.subplots(len(canaux), 1, figsize=(12, 2.6 * len(canaux)), sharex=True)
        for ax, (canal, titre) in zip(np.atleast_1d(axes), canaux):
            t, x = _segment(raw, canal, debut, duree)
            ax.plot(t, x, lw=0.7, color="#2E4057")
            ax.axvspan(ev.debut, ev.fin, color="#E84855", alpha=0.2, label=f"{ev.nom} {ev.duree:.0f} s")
            ax.set_title(titre, loc="left", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        np.atleast_1d(axes)[-1].set_xlabel("secondes depuis le début de l'enregistrement")
        fig.suptitle(f"{edf.stem} — apnée obstructive à {ev.debut:.0f} s", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / "apnee_obstructive.png", dpi=130)
        plt.close(fig)
        print(f"  apnée : {ev.debut:.1f} s, {ev.duree:.1f} s ({len(apnees)} apnées obstructives)")

    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
