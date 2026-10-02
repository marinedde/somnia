#!/usr/bin/env python3
"""
Tâche 0.2 — Extraction des caractéristiques avec la personne à côté de chaque époque.

Remplace les cellules de découpage du notebook 02. Produit un tableau complet
par tâche, SANS découpage : le découpage est fait ensuite, par personne, à partir
du fichier data/splits/physionet_v1.json (python_scripts/make_split.py).

Usage :
    python python_scripts/prepare_features.py            # EEG + ECG
    python python_scripts/prepare_features.py --task eeg

Sorties :
    data/processed/eeg_features.npz   X (n,16) y personne enregistrement feature_names
    data/processed/ecg_features.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.physionet import STAGE_NAMES, extraire_ecg, extraire_eeg  # noqa: E402

RAW_EEG = ROOT / "data" / "raw"
RAW_ECG = ROOT / "data" / "raw_apnea"
OUT = ROOT / "data" / "processed"


def _resume(nom: str, t: dict, noms_classes: dict) -> None:
    pers = Counter(t["personne"])
    print(f"\n{nom} : {len(t['y']):,} époques, {len(set(t['enregistrement']))} enregistrements, "
          f"{len(pers)} personnes")
    for code, n in sorted(Counter(t["y"].tolist()).items()):
        print(f"   {noms_classes[code]:6s} : {n:6,}  ({n / len(t['y']) * 100:4.1f} %)")
    print("   époques par personne : min %d, médiane %d, max %d"
          % (min(pers.values()), int(np.median(list(pers.values()))), max(pers.values())))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extraction des caractéristiques PhysioNet")
    parser.add_argument("--task", choices=["all", "eeg", "ecg"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    if args.task in ("all", "eeg"):
        t = extraire_eeg(RAW_EEG)
        np.savez_compressed(OUT / "eeg_features.npz", **t)
        _resume("EEG (Sleep-EDF)", t, STAGE_NAMES)
    if args.task in ("all", "ecg"):
        t = extraire_ecg(RAW_ECG)
        np.savez_compressed(OUT / "ecg_features.npz", **t)
        _resume("ECG (Apnea-ECG)", t, {0: "Normal", 1: "Apnée"})
    print(f"\nFichiers écrits dans {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
