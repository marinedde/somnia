#!/usr/bin/env python3
"""
Sorties des réseaux sur les 40 nuits de VALIDATION, calculées une fois et gardées hors dépôt.

Les analyses qui suivent (zones « possibles », index cliniques, intervalles par bootstrap) rejouent
des règles sur les mêmes sorties : inutile de refaire tourner les réseaux à chaque fois.

Sortie : ~/data/shhs/cache_val/<id>.npz
    logits_eeg, logits_multi   (n_epoques, 5)   stades : EEG seul / EEG + yeux + menton, divisés par la température
    proba_v3, proba_v4         (n_sec, 3)       événements : v3 (sommeil du modèle EEG) / v4 (sommeil du modèle multi)
    Ces quatre clés sont la graine 42 ; les mêmes avec le suffixe _s1 et _s2 pour les deux autres graines.
Le test n'est pas touché.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import PROCESSED, personnes_du_split  # noqa: E402
from somnia.deep.multi import MULTI_DIR, VARIANTES, NuitMulti, ReseauMulti, predire_nuit_multi, preparer_signaux  # noqa: E402
from somnia.deep.resp_net import ReseauEvenements, charger_nuits, proba_nuit  # noqa: E402
from somnia.deep.train import appareil  # noqa: E402

CACHE = PROCESSED.parent / "cache_val"
GRAINES = (42, 1, 2)


def suffixe(graine: int) -> str:
    return "" if graine == 42 else f"_s{graine}"


def reseau_stades(variante: str, graine: int):
    m = ReseauMulti(VARIANTES[variante]); m.load_state_dict(torch.load(ROOT / f"models/multi/{variante}_s{graine}.pt", map_location="cpu"))
    T = json.loads((ROOT / f"models/multi/{variante}_s{graine}.json").read_text(encoding="utf-8"))["calibration"]["temperature"]
    return m.eval(), T


def main() -> int:
    dev = appareil()
    CACHE.mkdir(parents=True, exist_ok=True)
    stades = {f"{nom}{suffixe(g)}": reseau_stades(v, g) for g in GRAINES for nom, v in (("eeg", "eeg"), ("multi", "eeg_eog_emg"))}
    out: dict[str, dict] = {}
    for version, sommeil in (("v3", "seq"), ("v4", "multi")):
        val = charger_nuits(personnes_du_split("val"), sommeil=sommeil)
        for g in GRAINES:
            net = ReseauEvenements(canaux=5); net.load_state_dict(torch.load(ROOT / f"models/resp/evenements_{version}_s{g}.pt", map_location="cpu")); net.to(dev)
            for n in val:
                out.setdefault(n.ident, {})[f"proba_{version}{suffixe(g)}"] = proba_nuit(net, n, dev, canal_sommeil=True).astype(np.float16)
    for i, (ident, d) in enumerate(sorted(out.items()), 1):
        with np.load(MULTI_DIR / f"{ident}.npz", allow_pickle=False) as f:
            signaux = f["signaux"]
        for nom, (m, T) in stades.items():
            X = preparer_signaux(signaux, m.canaux)
            d[f"logits_{nom}"] = (predire_nuit_multi(m, NuitMulti(ident, "", X, np.zeros(len(X), dtype=np.int64)), torch.device("cpu")) / T).astype(np.float32)
        np.savez_compressed(CACHE / f"{ident}.npz", **d)
        if i % 10 == 0:
            print(f"  {i}/{len(out)} nuits", flush=True)
    print(f"→ {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
