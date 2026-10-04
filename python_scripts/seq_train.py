#!/usr/bin/env python3
"""
Horizon 1.2 — Modèle de séquence pour les stades : encodeur par époque + lecture de la nuit.

Usage :
    python python_scripts/seq_train.py --petit-lot
    python python_scripts/seq_train.py                 # graine 42
    python python_scripts/seq_train.py --seed 1

Règles :
  - personnes d'entraînement de shhs_v1 ; arrêt anticipé et mesures sur la VALIDATION ;
  - le test des stades a déjà été ouvert une fois (docs/RESULTATS_TEST.md) : il n'est pas rouvert
    pour ce modèle. Une nouvelle ouverture demandera une nouvelle cohorte.
  - l'encodeur part des poids du réseau par époque (models/cnn/eeg_f1_s42.pt) s'ils existent.

Sorties : models/cnn/eeg_seq_s{seed}.pt (hors git), models/cnn/eeg_seq_s{seed}.json (versionné,
même format que les autres entraînements : il entre dans docs/RESULTATS_CNN.md).
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
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import personnes_du_split, poids_de_classes  # noqa: E402
from somnia.deep.model import n_parametres  # noqa: E402
from somnia.deep.seq import IGNORER, ReseauSequence, Tranches, accord_eveil_sommeil, charger_nuits_eeg, kappa_par_nuit, predire_nuit  # noqa: E402
from somnia.deep.train import _softmax, appareil, calibration_croisee, courbe_couverture, fixer_graines  # noqa: E402
from somnia.evaluation import metriques_stades  # noqa: E402

OUT = ROOT / "models" / "cnn"
INIT = OUT / "eeg_f1_s42.pt"


def evaluer(modele, val, dev) -> dict:
    logits = [predire_nuit(modele, n, dev) for n in val]
    y = np.concatenate([n.y for n in val]); lg = np.concatenate(logits)
    pers = np.concatenate([np.full(len(n), n.personne) for n in val])
    ok = y >= 0
    proba = _softmax(lg[ok])
    preds = [l.argmax(axis=1) for l in logits]
    kn = kappa_par_nuit(val, preds)
    return {
        "metriques": metriques_stades(y[ok], proba.argmax(axis=1)),
        "eveil_sommeil": accord_eveil_sommeil(y[ok], proba.argmax(axis=1)),
        "kappa_par_nuit": {"mediane": float(np.nanmedian(kn)), "q1": float(np.nanpercentile(kn, 25)),
                           "q3": float(np.nanpercentile(kn, 75)), "min": float(np.nanmin(kn))},
        "calibration": calibration_croisee(lg[ok], y[ok], pers[ok]),
        "couverture": courbe_couverture(proba, y[ok], "eeg"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--longueur", type=int, default=32, help="époques par tranche d'entraînement")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--sans-init", action="store_true", help="encodeur de zéro au lieu des poids du réseau par époque")
    parser.add_argument("--petit-lot", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nom = f"eeg_seq_s{args.seed}"
    lignes = []

    def journal(msg):
        print(msg, flush=True); lignes.append(msg)

    fixer_graines(args.seed)
    dev = appareil()
    t0 = time.time()
    train = charger_nuits_eeg(personnes_du_split("train"))
    journal(f"[{nom}] entraînement : {len(train)} nuits, {sum(len(n) for n in train):,} époques, chargées en {time.time() - t0:.0f} s")
    modele = ReseauSequence().to(dev)
    if INIT.exists() and not args.sans_init:
        etat = {k.replace("encodeur.", "", 1): v for k, v in torch.load(INIT, map_location="cpu").items() if k.startswith("encodeur.")}
        modele.encodeur.load_state_dict(etat)
        journal(f"encodeur initialisé depuis {INIT.name}")
    journal(f"appareil {dev.type} | {n_parametres(modele):,} paramètres | tranches de {args.longueur} époques")
    y_tout = np.concatenate([n.y for n in train]); y_tout = y_tout[y_tout >= 0]
    perte = nn.CrossEntropyLoss(weight=poids_de_classes(y_tout, 5).to(dev), ignore_index=IGNORER)

    if args.petit_lot:
        ds = Tranches(train[:4], args.longueur)
        x = torch.stack([ds[k][0] for k in range(0, 8)]).to(dev); y = torch.stack([ds[k][1] for k in range(0, 8)]).to(dev)
        opt = torch.optim.Adam(modele.parameters(), lr=2e-3); pertes = []
        modele.train()
        for _ in range(300):
            opt.zero_grad(set_to_none=True)
            loss = perte(modele(x).reshape(-1, 5), y.reshape(-1)); loss.backward(); opt.step(); pertes.append(loss.item())
        modele.eval()
        with torch.no_grad():
            p = modele(x).argmax(-1); ok = y != IGNORER
            ex = float((p[ok] == y[ok]).float().mean())
        journal(f"petit lot (8 tranches de {args.longueur} époques) : perte {pertes[0]:.3f} -> {pertes[-1]:.4f}, exactitude {ex:.1%}")
        print("OK : le réseau apprend par cœur" if pertes[-1] < 0.1 * pertes[0] else "ÉCHEC")
        return 0 if pertes[-1] < 0.1 * pertes[0] else 1

    val = charger_nuits_eeg(personnes_du_split("val"))
    journal(f"validation : {len(val)} nuits, {sum(len(n) for n in val):,} époques")
    loader = DataLoader(Tranches(train, args.longueur, entrainement=True, graine=args.seed), batch_size=args.batch, shuffle=True, num_workers=0)
    # L'encodeur est déjà entraîné : pas plus petit pour lui, plus grand pour la séquence et la tête
    opt = torch.optim.AdamW([{"params": modele.encodeur.parameters(), "lr": 3e-4},
                             {"params": list(modele.sequence.parameters()) + list(modele.tete.parameters()), "lr": 1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    historique, meilleur, meilleur_etat, sans_progres, meilleure_epoque = [], -1.0, None, 0, 0
    for ep in range(1, args.epochs + 1):
        modele.train(); te = time.time(); total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = perte(modele(x).reshape(-1, 5), y.reshape(-1)); loss.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 5.0); opt.step()
            total += loss.item() * len(y); n += len(y)
        sched.step()
        r = evaluer(modele, val, dev)
        k = r["metriques"]["kappa"]
        historique.append({"epoque": ep, "perte_train": total / n, "perte_val": float("nan"), "score_val": k,
                           "duree_s": round(time.time() - te, 1), **r["metriques"], "accord_eveil_sommeil": r["eveil_sommeil"]["accord"]})
        journal(f"époque {ep:2d} | perte {total / n:.4f} | val : kappa {k:.4f}, exactitude {r['metriques']['accuracy']:.3f}, "
                f"F1 N1 {r['metriques']['f1_N1']:.3f}, éveil/sommeil {r['eveil_sommeil']['accord']:.3f} | {time.time() - te:.0f} s")
        if k > meilleur + 1e-4:
            meilleur, meilleure_epoque, sans_progres = k, ep, 0
            meilleur_etat = {kk: v.detach().cpu().clone() for kk, v in modele.state_dict().items()}
        else:
            sans_progres += 1
            if sans_progres >= args.patience:
                journal("arrêt anticipé"); break
    modele.load_state_dict(meilleur_etat)
    r = evaluer(modele, val, dev)
    torch.save(modele.state_dict(), OUT / f"{nom}.pt")
    rapport = {"nom": nom, "tache": "eeg", "fraction": 1.0, "graine": args.seed, "variante": "CNN + séquence sur la nuit (bi-GRU)",
               "date": datetime.now().isoformat(timespec="minutes"), "appareil": dev.type,
               "n_personnes_train": len(train), "n_exemples_train": int(sum(len(n) for n in train)),
               "n_personnes_val": len(val), "n_exemples_val": int(sum(len(n) for n in val)),
               "meilleure_epoque": meilleure_epoque, "historique": historique, "validation": r["metriques"],
               "eveil_sommeil": r["eveil_sommeil"], "kappa_par_nuit": r["kappa_par_nuit"],
               "calibration": r["calibration"], "couverture": r["couverture"], "par_personne": None,
               "config": {"longueur": args.longueur, "init_encodeur": not args.sans_init},
               "duree_min": round((time.time() - t0) / 60, 1), "journal": lignes}
    (OUT / f"{nom}.json").write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    m = r["metriques"]
    journal(f"meilleure époque {meilleure_epoque} | validation : exactitude {m['accuracy']:.3f}, kappa {m['kappa']:.3f}, F1 macro {m['f1_macro']:.3f}, "
            f"F1 W {m['f1_Wake']:.3f} N1 {m['f1_N1']:.3f} N2 {m['f1_N2']:.3f} N3 {m['f1_N3']:.3f} REM {m['f1_REM']:.3f}")
    journal(f"éveil / sommeil : accord {r['eveil_sommeil']['accord']:.3f} | kappa par nuit, médiane {r['kappa_par_nuit']['mediane']:.3f} "
            f"({r['kappa_par_nuit']['q1']:.3f}–{r['kappa_par_nuit']['q3']:.3f}) | ECE {r['calibration']['ece_avant']:.3f} → {r['calibration']['ece_apres']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
