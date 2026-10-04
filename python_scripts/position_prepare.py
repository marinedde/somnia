#!/usr/bin/env python3
"""
Position du corps, une valeur par seconde, pour chaque nuit de la cohorte respiratoire.

Le canal POSITION de SHHS vaut 0, 1, 2 ou 3 (1 Hz). Sortie (hors dépôt) :
~/data/shhs/processed_resp/<id>_position.npy  (n_sec,) int8, valeurs brutes ; −1 si le canal manque.
Le sens des codes est établi dans python_scripts/position_report.py, pas supposé ici.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.resp_net import RESP_DIR, nuits_retenues  # noqa: E402
from somnia.resp import trouver  # noqa: E402
from somnia.shhs_prepare import lire_canal_natif  # noqa: E402

RAW = Path(os.environ.get("SHHS_DIR", Path.home() / "data" / "shhs" / "raw"))


def main() -> int:
    import mne

    edfs = {p.stem: p for p in RAW.rglob("*.edf")}
    nuits = sorted(nuits_retenues().values())
    absents = 0
    for i, ident in enumerate(nuits, 1):
        cible = RESP_DIR / f"{ident}_position.npy"
        if not cible.exists():
            noms = mne.io.read_raw_edf(str(edfs[ident]), preload=False, verbose="error").ch_names
            canal = trouver(noms, ["POSITION", "POS"])
            if canal is None:
                absents += 1
                np.save(cible, np.full(1, -1, dtype=np.int8)); continue
            x, fs = lire_canal_natif(edfs[ident], canal)
            # MNE suppose des volts pour un canal sans unité ; les codes sont des entiers 0–3
            x = x if np.nanmax(np.abs(x)) >= 1 or np.nanmax(np.abs(x)) == 0 else x * 1e6
            pas = int(round(fs))
            np.save(cible, np.rint(x[::pas]).astype(np.int8))
        if i % 50 == 0 or i == len(nuits):
            print(f"  {i}/{len(nuits)} nuits", flush=True)
    print(f"canal absent : {absents} nuit(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
