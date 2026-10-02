#!/usr/bin/env python3
"""
Étape 5 — Que vaut l'encodeur pré-entraîné ? Sonde linéaire (5.6) et affinage (5.7).

Pour une tâche et un encodeur pré-entraîné :
  - sonde linéaire : encodeur GELÉ, régression logistique dessus, avec 10 % puis 100 % des personnes ;
  - affinage : encodeur DÉGELÉ, tout le réseau entraîné (même boucle qu'à l'étape 4), avec 1 %,
    10 % et 100 % des personnes, pour comparer ligne à ligne avec « de zéro ».
Toujours sur la validation ; le test reste fermé.

Usage :
    python python_scripts/ssl_eval.py --task eeg
    python python_scripts/ssl_eval.py --task ecg --fractions 0.1 0.01

Sorties : models/cnn/{task}_f{fraction}_s{seed}_pre.json  (affinage)
          models/cnn/{task}_f{fraction}_s{seed}_sonde.json (sonde linéaire)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import charger_tableau, personnes_du_split, sous_ensemble_de_personnes  # noqa: E402
from somnia.deep.model import Encodeur  # noqa: E402
from somnia.deep.ssl import sonde_lineaire  # noqa: E402
from somnia.deep.train import _metriques, appareil, calibration_croisee, courbe_couverture, entrainer, evaluer  # noqa: E402
from python_scripts.cnn_train import spearman_par_personne  # noqa: E402

OUT = ROOT / "models" / "cnn"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["eeg", "ecg"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    dev = appareil()
    etat = torch.load(OUT / f"ssl_{args.task}_s{args.seed}_encodeur.pt", map_location="cpu")
    val = charger_tableau(args.task, personnes_du_split("val"))
    tout_train = personnes_du_split("train")

    for fraction in sorted(args.fractions):
        pers = sous_ensemble_de_personnes(tout_train, fraction, args.seed)
        train = charger_tableau(args.task, pers)
        base = {"tache": args.task, "fraction": fraction, "graine": args.seed, "n_personnes_train": len(pers),
                "n_exemples_train": len(train), "n_personnes_val": int(len(set(val.personne))), "n_exemples_val": len(val),
                "appareil": dev.type, "pre_entrainement": f"ssl_{args.task}_s{args.seed}"}

        # 5.6 sonde linéaire (encodeur gelé)
        if fraction >= 0.1:
            t0 = time.time()
            enc = Encodeur(); enc.load_state_dict(etat); enc.to(dev)
            proba, y = sonde_lineaire(enc, train, val, dev, args.seed)
            logits = np.log(np.clip(proba, 1e-9, 1))
            r = {**base, "nom": f"{args.task}_f{fraction:g}_s{args.seed}_sonde", "variante": "sonde linéaire (encodeur gelé)",
                 "meilleure_epoque": None, "historique": [], "validation": _metriques(args.task, y, proba),
                 "calibration": calibration_croisee(logits, y, val.personne), "couverture": courbe_couverture(proba, y, args.task),
                 "par_personne": spearman_par_personne({"logits": logits, "y": y, "personnes": val.personne}, args.task),
                 "duree_min": round((time.time() - t0) / 60, 1), "date": datetime.now().isoformat(timespec="minutes")}
            (OUT / f"{r['nom']}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[sonde {fraction:.0%}] " + ", ".join(f"{k} {v:.3f}" for k, v in r["validation"].items()), flush=True)

        # 5.7 affinage (encodeur dégelé)
        t0 = time.time(); lignes = []

        def journal(msg):
            print(msg, flush=True); lignes.append(msg)
        modele, hist = entrainer(args.task, train, val, graine=args.seed, max_epoques=args.epochs, patience=args.patience,
                                 journal=journal, etat_encodeur=etat)
        res = evaluer(modele, args.task, val)
        r = {**base, "nom": f"{args.task}_f{fraction:g}_s{args.seed}_pre", "variante": "pré-entraîné, affiné",
             "meilleure_epoque": hist.meilleure_epoque, "historique": hist.epoques, "validation": res["metriques"],
             "calibration": res["calibration"], "couverture": res["couverture"],
             "par_personne": spearman_par_personne(res, args.task), "duree_min": round((time.time() - t0) / 60, 1),
             "date": datetime.now().isoformat(timespec="minutes"), "journal": lignes}
        torch.save(modele.state_dict(), OUT / f"{r['nom']}.pt")
        (OUT / f"{r['nom']}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[affiné {fraction:.0%}] " + ", ".join(f"{k} {v:.3f}" for k, v in r["validation"].items() if not k.startswith("f1_")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
