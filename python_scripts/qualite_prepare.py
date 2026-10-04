#!/usr/bin/env python3
"""
Horizon 1.5 — Où la saturation était-elle invalide ? Un masque par seconde, pour chaque nuit.

La préparation respiratoire nettoie la saturation (valeurs hors de 50–100 % remplacées par la
dernière valeur valide) : après elle, on ne voit plus où le capteur était décollé. Ce script
relit le canal brut et garde l'information.

Sortie (hors dépôt) : ~/data/shhs/processed_resp/<id>_sao2_invalide.npy  (n_sec,) booléen.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.resp_net import RESP_DIR, nuits_retenues  # noqa: E402
from somnia.resp import CANAUX_RESP, trouver  # noqa: E402
from somnia.shhs_prepare import lire_canal_natif  # noqa: E402

RAW = Path(os.environ.get("SHHS_DIR", Path.home() / "data" / "shhs" / "raw"))


def main() -> int:
    import mne

    edfs = {p.stem: p for p in RAW.rglob("*.edf")}
    nuits = sorted(nuits_retenues().values())
    for i, ident in enumerate(nuits, 1):
        cible = RESP_DIR / f"{ident}_sao2_invalide.npy"
        if not cible.exists():
            noms = mne.io.read_raw_edf(str(edfs[ident]), preload=False, verbose="error").ch_names
            x, fs = lire_canal_natif(edfs[ident], trouver(noms, CANAUX_RESP["sao2"]))
            brut = x if np.nanmax(x) > 1.5 else x * 100          # même convention que resp_prepare.py
            invalide = (brut < 50) | (brut > 100) | ~np.isfinite(brut)
            pas = int(round(fs))
            n = len(invalide) // pas
            np.save(cible, invalide[: n * pas].reshape(n, pas).mean(axis=1) >= 0.5)
        if i % 50 == 0 or i == len(nuits):
            print(f"  {i}/{len(nuits)} nuits", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
