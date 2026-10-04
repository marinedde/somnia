#!/usr/bin/env python3
"""
Horizon 1.3 — Préparer les signaux du scoring humain : deux EEG, les yeux, le menton.

Un technicien score avec trois familles de signaux : les ondes (EEG), les mouvements des yeux
(EOG : rapides en REM, lents à l'endormissement) et le tonus du menton (EMG : effondré en REM).
Avec l'EEG seul, REM, N1 et éveil se ressemblent.

Pour chaque nuit retenue de la cohorte : 5 canaux à 100 Hz, découpés en époques de 30 s.
    0 EEG      C4-A1   125 Hz -> 100
    1 EEG2     C3-A2   125 Hz -> 100   (canal `EEG(sec)` ou `EEG2`)
    2 EOG-G             50 Hz -> 100
    3 EOG-D             50 Hz -> 100
    4 EMG              125 Hz -> 100
Un canal absent est rempli de zéros et signalé dans `present` : le réseau apprend à s'en passer.

Sortie (hors dépôt) : ~/data/shhs/processed_multi/<id>.npz
    signaux (n_epoques, 5, 3000) float16 en µV, stades (n_epoques,) int8, present (5,) bool

Usage : python python_scripts/shhs_prepare_multi.py [--force] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.shhs import lire_annotations  # noqa: E402
from somnia.shhs_prepare import decouper_en_epoques, lire_canal_natif, reechantillonner, trouver_canal  # noqa: E402

RAW = Path(os.environ.get("SHHS_DIR", Path.home() / "data" / "shhs" / "raw"))
PROCESSED = Path(os.environ.get("SHHS_PROCESSED", Path.home() / "data" / "shhs" / "processed"))
OUT = PROCESSED.parent / "processed_multi"
CANAUX = [("EEG", ["EEG"]), ("EEG2", ["EEG(sec)", "EEG2", "EEG 2", "EEG sec"]),
          ("EOG-G", ["EOG(L)", "EOG L", "LOC"]), ("EOG-D", ["EOG(R)", "EOG R", "ROC"]), ("EMG", ["EMG"])]


def main() -> int:
    import mne

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    with open(PROCESSED / "cohorte.csv", encoding="utf-8") as f:
        retenues = [r["enregistrement"] for r in csv.DictReader(f) if r.get("exclue") != "True"][: args.limit]
    edfs = {p.stem: p for p in RAW.rglob("*.edf")}
    xmls = {p.name.replace("-nsrr.xml", ""): p for p in RAW.rglob("*-nsrr.xml")}
    OUT.mkdir(parents=True, exist_ok=True)
    absents = Counter()
    for i, ident in enumerate(retenues, 1):
        cible = OUT / f"{ident}.npz"
        if cible.exists() and not args.force:
            continue
        ann = lire_annotations(xmls[ident])
        noms = mne.io.read_raw_edf(str(edfs[ident]), preload=False, verbose="error").ch_names
        S = np.zeros((ann.n_epoques, len(CANAUX), 3000), dtype=np.float16)
        present = np.zeros(len(CANAUX), dtype=bool)
        for c, (nom, candidats) in enumerate(CANAUX):
            canal = trouver_canal(noms, candidats)
            if canal is None:
                absents[nom] += 1
                continue
            x, fs = lire_canal_natif(edfs[ident], canal)
            ep, _ = decouper_en_epoques(reechantillonner(x, fs), ann.n_epoques)
            S[:, c] = np.clip(ep * 1e6, -3e4, 3e4).astype(np.float16)        # volts -> µV
            present[c] = True
        np.savez_compressed(cible, signaux=S, stades=ann.stades.astype(np.int8), present=present,
                            canaux=np.array([n for n, _ in CANAUX]))
        if i % 25 == 0 or i == len(retenues):
            print(f"  {i}/{len(retenues)} nuits", flush=True)
    print(f"canaux absents (nuits) : {dict(absents) or 'aucun'}")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
