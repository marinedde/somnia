#!/usr/bin/env python3
"""
Probabilité de sommeil PRÉDITE, à la seconde, pour chaque nuit de la cohorte respiratoire.

Pourquoi : 47 % des fausses propositions du réseau d'événements commencent pendant l'éveil. Le
technicien ne marque presque rien quand le patient est réveillé ; le réseau, lui, ne savait pas
si le patient dormait. On lui donne l'information, telle qu'un outil réel l'aurait : celle du
réseau de stades (EEG), pas celle du technicien.

Sortie : ~/data/shhs/processed_resp/<id>_psommeil.npy  (n_sec,) float16, P(stade ≠ éveil) calibrée.

Réserve : le réseau de stades a été entraîné sur les personnes d'entraînement ; ses prédictions y
sont donc un peu plus propres que sur la validation. C'est noté dans le rapport.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import PROCESSED, normaliser_lot  # noqa: E402
from somnia.deep.model import CNN1D  # noqa: E402
from somnia.deep.resp_net import RESP_DIR, nuits_retenues  # noqa: E402
from somnia.shhs_prepare import charger_nuit  # noqa: E402


def main() -> int:
    modele = CNN1D(5); modele.load_state_dict(torch.load(ROOT / "models/cnn/eeg_f1_s42.pt", map_location="cpu")); modele.eval()
    T = json.loads((ROOT / "models/cnn/eeg_f1_s42.json").read_text(encoding="utf-8"))["calibration"]["temperature"]
    nuits = nuits_retenues()
    for i, (pers, ident) in enumerate(sorted(nuits.items()), 1):
        eeg = charger_nuit(PROCESSED / f"{ident}.npz")["eeg"]
        with torch.no_grad():
            lg = np.concatenate([modele(normaliser_lot(torch.from_numpy(eeg[k:k + 512]).unsqueeze(1))).numpy() / T
                                 for k in range(0, len(eeg), 512)])
        z = lg - lg.max(axis=1, keepdims=True); p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        p_sommeil = 1.0 - p[:, 0]                                    # par époque
        np.save(RESP_DIR / f"{ident}_psommeil.npy", np.repeat(p_sommeil, 30).astype(np.float16))
        if i % 50 == 0 or i == len(nuits):
            print(f"  {i}/{len(nuits)} nuits", flush=True)
    print(f"→ {RESP_DIR}/*_psommeil.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
