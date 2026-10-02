#!/usr/bin/env python3
"""
Étape 5 — Pré-entraînement contrastif de l'encodeur, sans étiquette (tâches 5.4 et 5.5).

Sur les époques des personnes d'ENTRAÎNEMENT seulement : les signaux de validation et de test
ne servent à rien ici, même sans étiquette (roadmap, pièges).

Usage :
    python python_scripts/ssl_pretrain.py --task eeg --petit-lot
    python python_scripts/ssl_pretrain.py --task eeg --steps 4000
    python python_scripts/ssl_pretrain.py --task ecg --steps 4000

Sorties : models/cnn/ssl_{task}_s{seed}_encodeur.pt (hors git), models/cnn/ssl_{task}_s{seed}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import charger_tableau, personnes_du_split  # noqa: E402
from somnia.deep.ssl import petit_lot_contrastif, pre_entrainer  # noqa: E402

OUT = ROOT / "models" / "cnn"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["eeg", "ecg"], required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=192, help="personnes par lot (une époque chacune)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--petit-lot", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nom = f"ssl_{args.task}_s{args.seed}"
    lignes = []

    def journal(msg):
        print(msg, flush=True); lignes.append(msg)

    t0 = time.time()
    train = charger_tableau(args.task, personnes_du_split("train"))
    journal(f"[{nom}] {len(set(train.personne))} personnes, {len(train):,} époques (étiquettes ignorées)")
    if args.petit_lot:
        r = petit_lot_contrastif(args.task, train, graine=args.seed, journal=journal)
        (OUT / f"{nom}_petit_lot.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        print("OK : la perte contrastive descend" if r["ok"] else "ÉCHEC : la perte ne descend pas")
        return 0 if r["ok"] else 1

    encodeur, courbe = pre_entrainer(args.task, train, pas=args.steps, taille_lot=args.batch, graine=args.seed, journal=journal)
    torch.save(encodeur.state_dict(), OUT / f"{nom}_encodeur.pt")
    rapport = {"nom": nom, "tache": args.task, "graine": args.seed, "pas": args.steps, "taille_lot": args.batch,
               "n_personnes": int(len(set(train.personne))), "n_epoques": len(train),
               "date": datetime.now().isoformat(timespec="minutes"), "duree_min": round((time.time() - t0) / 60, 1),
               "perte_debut": courbe[0], "perte_fin": float(sum(courbe[-100:]) / 100),
               "courbe_tous_les_50_pas": courbe[::50], "journal": lignes}
    (OUT / f"{nom}.json").write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    journal(f"→ {OUT / nom}_encodeur.pt ({rapport['duree_min']} min, perte {courbe[0]:.3f} -> {rapport['perte_fin']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
