#!/usr/bin/env python3
"""
Étape 3 — Les 16 caractéristiques de Somnia sur la cohorte SHHS, avec la personne à côté.

Mêmes extracteurs que PhysioNet et que l'API (`app/feature_extractor.py`, `app/ecg_features.py`),
donc mêmes conventions d'unités :
  - EEG : l'extracteur attend des VOLTS et multiplie par 1e6. Les .npz SHHS sont en µV -> on divise par 1e6.
  - ECG : l'extracteur a été entraîné sur Apnea-ECG en mV. Les .npz SHHS sont en mV -> tels quels.

Deux tableaux :
  - EEG : une ligne par époque de 30 s (toutes les époques, éveil compris : une vraie nuit en contient).
  - ECG : une ligne par fenêtre de 60 s = deux époques consécutives (2k, 2k+1), pour rester
    comparable à Apnea-ECG (minutes). Étiquette = 1 si l'une des deux époques est « apnée ».
    Seules les fenêtres dont les DEUX époques sont du sommeil sont gardées (décision : une apnée
    se définit pendant le sommeil).

Chaque ligne porte : personne, enregistrement, ensemble (train/val/test) et un drapeau
« aberrant » (saturation EEG ou ECG, voir shhs_eda.py) pour mesurer le poids de ces époques.

Sorties (hors dépôt) : ~/data/shhs/processed/features_eeg.npz, features_ecg.npz
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ecg_features import ECGFeatureExtractor  # noqa: E402
from app.feature_extractor import FeatureExtractor  # noqa: E402
from somnia.shhs_prepare import charger_nuit  # noqa: E402

PROCESSED = Path(os.environ.get("SHHS_PROCESSED", Path.home() / "data" / "shhs" / "processed"))
SPLIT = ROOT / "data" / "splits" / "shhs_v1.json"
EEG_SAT_UV, ECG_SAT_MV = 120.0, 1.24


def main() -> int:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))["taches"]["shhs"]
    ensemble_de = {p: k for k, v in split.items() for p in v}
    fx_eeg = FeatureExtractor(fs=100, expected_len=3000)
    fx_ecg = ECGFeatureExtractor(fs=100, expected_len=6000)

    eeg_parts = {k: [] for k in ("X", "y", "personne", "enregistrement", "ensemble", "aberrant")}
    ecg_parts = {k: [] for k in ("X", "y", "personne", "enregistrement", "ensemble", "aberrant")}
    fichiers = sorted(PROCESSED.glob("shhs1-*.npz"))
    for i, f in enumerate(fichiers, 1):
        n = charger_nuit(f)
        pers, ident = n["meta"]["personne"], n["meta"]["enregistrement"]
        ens = ensemble_de.get(pers)
        if ens is None:
            print(f"  {ident} : absent du découpage, ignoré")
            continue
        eeg, ecg, stades, apnee = n["eeg"], n["ecg"], n["stades"], n["apnee"]
        sat_eeg = (np.abs(eeg) >= EEG_SAT_UV).mean(axis=1) > 0.05
        sat_ecg = (np.abs(ecg) >= ECG_SAT_MV).mean(axis=1) > 0.05
        aberr = sat_eeg | sat_ecg

        # EEG : toutes les époques scorées
        ok = stades >= 0
        eeg_parts["X"].append(fx_eeg.transform(eeg[ok] / 1e6))
        eeg_parts["y"].append(stades[ok])
        eeg_parts["aberrant"].append(aberr[ok])
        for k, v in (("personne", pers), ("enregistrement", ident), ("ensemble", ens)):
            eeg_parts[k].append(np.full(ok.sum(), v))

        # ECG : fenêtres de 60 s, les deux époques en sommeil
        m = len(stades) // 2
        s2 = stades[: 2 * m].reshape(m, 2)
        sommeil = np.all(np.isin(s2, [1, 2, 3, 4]), axis=1)
        if sommeil.any():
            fen = ecg[: 2 * m].reshape(m, 6000)[sommeil]
            y2 = apnee[: 2 * m].reshape(m, 2).max(axis=1)[sommeil]
            ab2 = aberr[: 2 * m].reshape(m, 2).any(axis=1)[sommeil]
            ecg_parts["X"].append(fx_ecg.transform(fen))
            ecg_parts["y"].append(y2)
            ecg_parts["aberrant"].append(ab2)
            for k, v in (("personne", pers), ("enregistrement", ident), ("ensemble", ens)):
                ecg_parts[k].append(np.full(sommeil.sum(), v))
        if i % 25 == 0 or i == len(fichiers):
            print(f"  {i}/{len(fichiers)} nuits", flush=True)

    for nom, parts, noms in (("eeg", eeg_parts, FeatureExtractor.feature_names()),
                             ("ecg", ecg_parts, ECGFeatureExtractor.feature_names())):
        t = {k: np.concatenate(v) for k, v in parts.items()}
        t["X"] = np.nan_to_num(t["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        t["feature_names"] = np.asarray(noms)
        np.savez_compressed(PROCESSED / f"features_{nom}.npz", **t)
        print(f"{nom} : {len(t['y']):,} lignes, {len(set(t['personne']))} personnes, "
              f"aberrantes {t['aberrant'].mean():.1%} -> features_{nom}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
