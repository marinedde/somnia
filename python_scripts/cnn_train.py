#!/usr/bin/env python3
"""
Étape 4 — Réseau convolutif supervisé sur le signal brut SHHS.

Usage :
    python python_scripts/cnn_train.py --task eeg --petit-lot          # 4.3 : 32 exemples par cœur
    python python_scripts/cnn_train.py --task eeg                       # 4.5 : toute la cohorte
    python python_scripts/cnn_train.py --task ecg --fraction 0.10       # 4.7 : 10 % des personnes étiquetées
    python python_scripts/cnn_train.py --task ecg --fraction 0.01

Règles :
  - entraînement sur les personnes d'ENTRAÎNEMENT de shhs_v1 (ou une fraction d'entre elles),
    arrêt anticipé et toutes les mesures sur la VALIDATION ; le test n'est jamais lu ;
  - pour l'apnée, la mesure clinique : corrélation de Spearman, par personne, entre la part de
    minutes prédites positives et l'index clinique SHHS (ahi_a0h3a, ahi_a0h4 ; covariables.csv).

Sorties (modèles hors git, métriques versionnées) :
    models/cnn/{task}_f{fraction}_s{seed}.pt
    models/cnn/{task}_f{fraction}_s{seed}.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import PROCESSED, charger_tableau, personnes_du_split, sous_ensemble_de_personnes  # noqa: E402
from somnia.deep.train import entrainer, evaluer, petit_lot  # noqa: E402

OUT = ROOT / "models" / "cnn"


def spearman_par_personne(res: dict, tache: str) -> dict | None:
    """Apnée : la part de fenêtres prédites positives par personne, contre l'index annoté et clinique."""
    if tache != "ecg":
        return None
    from scipy.stats import spearmanr

    proba = np.exp(res["logits"] - res["logits"].max(axis=1, keepdims=True)); proba /= proba.sum(axis=1, keepdims=True)
    pred, y, pers = proba.argmax(axis=1), res["y"], res["personnes"]
    cov_path = PROCESSED / "covariables.csv"
    cov = {}
    if cov_path.exists():
        with open(cov_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cov[f"shhs-{r['nsrrid']}"] = r
    est, ann, a3, a4 = [], [], [], []
    for p in sorted(set(pers)):
        m = pers == p
        est.append(pred[m].mean()); ann.append(y[m].mean())
        if p in cov:
            a3.append(float(cov[p]["ahi_a0h3a"] or "nan")); a4.append(float(cov[p]["ahi_a0h4"] or "nan"))
    out = {"n_personnes": len(est), "spearman_part_annotee": float(spearmanr(est, ann).correlation)}
    if len(a3) == len(est):
        out["spearman_ahi_a0h3a"] = float(spearmanr(est, a3, nan_policy="omit").correlation)
        out["spearman_ahi_a0h4"] = float(spearmanr(est, a4, nan_policy="omit").correlation)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["eeg", "ecg"], required=True)
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction des personnes d'entraînement étiquetées")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--petit-lot", action="store_true", help="test du petit lot (32 exemples) et rien d'autre")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nom = f"{args.task}_f{args.fraction:g}_s{args.seed}"
    journal_lignes = []

    def journal(msg):
        print(msg, flush=True); journal_lignes.append(msg)

    t0 = time.time()
    pers_train = sous_ensemble_de_personnes(personnes_du_split("train"), args.fraction, args.seed)
    train = charger_tableau(args.task, pers_train)
    journal(f"[{nom}] {len(pers_train)} personnes d'entraînement, {len(train):,} exemples, chargés en {time.time() - t0:.0f} s")

    if args.petit_lot:
        r = petit_lot(args.task, train, graine=args.seed, journal=journal)
        (OUT / f"{nom}_petit_lot.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        print("OK : le réseau apprend par cœur" if r["ok"] else "ÉCHEC : la perte ne descend pas, il y a un bug")
        return 0 if r["ok"] else 1

    val = charger_tableau(args.task, personnes_du_split("val"))
    journal(f"validation : {len(set(val.personne))} personnes, {len(val):,} exemples")
    modele, hist = entrainer(args.task, train, val, graine=args.seed, max_epoques=args.epochs,
                             patience=args.patience, batch=args.batch, journal=journal)
    res = evaluer(modele, args.task, val)
    journal(f"meilleure époque {hist.meilleure_epoque} | validation : "
            + ", ".join(f"{k} {v:.3f}" for k, v in res["metriques"].items() if not k.startswith("n_")))
    journal(f"calibration : ECE {res['calibration']['ece_avant']:.3f} -> {res['calibration']['ece_apres']:.3f} "
            f"(T = {res['calibration']['temperature']:.2f})")
    pp = spearman_par_personne(res, args.task)
    if pp:
        journal(f"par personne : {pp}")

    torch.save(modele.state_dict(), OUT / f"{nom}.pt")
    np.savez_compressed(OUT / f"{nom}_val_logits.npz", logits=res["logits"], y=res["y"], personnes=res["personnes"])
    rapport = {
        "nom": nom, "tache": args.task, "fraction": args.fraction, "graine": args.seed, "variante": "de zéro", "date": datetime.now().isoformat(timespec="minutes"),
        "n_personnes_train": len(pers_train), "n_exemples_train": len(train),
        "n_personnes_val": int(len(set(val.personne))), "n_exemples_val": len(val),
        "appareil": "mps" if torch.backends.mps.is_available() else "cpu",
        "meilleure_epoque": hist.meilleure_epoque, "historique": hist.epoques,
        "validation": res["metriques"], "calibration": res["calibration"], "couverture": res["couverture"],
        "par_personne": pp, "duree_min": round((time.time() - t0) / 60, 1), "journal": journal_lignes,
    }
    (OUT / f"{nom}.json").write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    journal(f"→ {OUT / nom}.json ({rapport['duree_min']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
