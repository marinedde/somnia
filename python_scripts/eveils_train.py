#!/usr/bin/env python3
"""
Horizon 2.1 — Entraîner le réseau de micro-éveils (EEG, second EEG, menton).

Usage :
    python python_scripts/eveils_train.py --petit-lot
    python python_scripts/eveils_train.py --seed 42

Règles : personnes d'entraînement de shhs_v1 ; arrêt anticipé et mesures sur la VALIDATION ; le
test n'est pas touché. Critère d'arrêt : F1 par événement, de bout en bout (propositions limitées
au sommeil PRÉDIT par le réseau de stades). Seuil de décision fixé à 0,5 avant de regarder.

Sorties : models/eveils/eveils_s{seed}.pt (hors git), .json (versionné).
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

from somnia.deep.data import personnes_du_split  # noqa: E402
from somnia.deep.eveils import FenetresEveils, ReseauEveils, charger_nuits_eveils, mesurer_eveils, proba_eveils  # noqa: E402
from somnia.deep.model import n_parametres  # noqa: E402
from somnia.deep.resp_net import RESP_DIR  # noqa: E402
from somnia.deep.train import appareil, fixer_graines  # noqa: E402

OUT = ROOT / "models" / "eveils"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--petit-lot", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nom, lignes = f"eveils_s{args.seed}", []

    def journal(m):
        print(m, flush=True); lignes.append(m)

    fixer_graines(args.seed); dev = appareil(); t0 = time.time()
    train = charger_nuits_eveils(personnes_du_split("train"), sommeil_dir=RESP_DIR)
    part = float(np.mean(np.concatenate([n.y for n in train])))
    journal(f"[{nom}] entraînement : {len(train)} nuits, {part:.1%} des secondes en micro-éveil, chargées en {time.time() - t0:.0f} s")
    modele = ReseauEveils().to(dev)
    journal(f"appareil {dev.type} | {n_parametres(modele):,} paramètres")
    # pondération adoucie (racine), comme pour le réseau d'événements : la classe rare compte plus, sans écraser la précision
    poids = torch.tensor([1.0, float(np.sqrt((1 - part) / part))], dtype=torch.float32, device=dev)
    perte = nn.CrossEntropyLoss(weight=poids, ignore_index=-100)

    if args.petit_lot:
        ds = FenetresEveils(train[:4], par_nuit=4, graine=args.seed)
        x = torch.stack([ds[k][0] for k in range(16)]).to(dev); y = torch.stack([ds[k][1] for k in range(16)]).to(dev)
        opt = torch.optim.Adam(modele.parameters(), lr=2e-3); pertes = []
        modele.train()
        for _ in range(300):
            opt.zero_grad(set_to_none=True); loss = perte(modele(x), y); loss.backward(); opt.step(); pertes.append(loss.item())
        journal(f"petit lot (16 fenêtres) : perte {pertes[0]:.3f} -> {pertes[-1]:.4f}")
        print("OK : le réseau apprend par cœur" if pertes[-1] < 0.1 * pertes[0] else "ÉCHEC")
        return 0 if pertes[-1] < 0.1 * pertes[0] else 1

    val = [n for n in charger_nuits_eveils(personnes_du_split("val"), sommeil_dir=RESP_DIR) if n.p_sommeil is not None]
    journal(f"validation : {len(val)} nuits avec sommeil prédit")
    loader = DataLoader(FenetresEveils(train, graine=args.seed), batch_size=args.batch, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(modele.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    meilleur, etat, sans, meilleure_epoque, historique = -1.0, None, 0, 0, []
    for ep in range(1, args.epochs + 1):
        modele.train(); te = time.time(); total, k = 0.0, 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True); loss = perte(modele(x), y); loss.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 5.0); opt.step()
            total += loss.item(); k += 1
        sched.step()
        probas = [proba_eveils(modele, n, dev) for n in val]
        m = mesurer_eveils(val, probas, sommeil_predit=True)
        historique.append({"epoque": ep, "perte_train": total / k, "f1": m["f1"], "precision": m["precision"], "rappel": m["rappel"]})
        journal(f"époque {ep:2d} | perte {total / k:.4f} | val : F1 {m['f1']:.3f} (précision {m['precision']:.3f}, rappel {m['rappel']:.3f}), "
                f"index Spearman {m['index']['spearman']:.3f} | {time.time() - te:.0f} s")
        if m["f1"] > meilleur + 1e-4:
            meilleur, meilleure_epoque, sans = m["f1"], ep, 0
            etat = {kk: v.detach().cpu().clone() for kk, v in modele.state_dict().items()}
        else:
            sans += 1
            if sans >= args.patience:
                journal("arrêt anticipé"); break
    modele.load_state_dict(etat)
    probas = [proba_eveils(modele, n, dev) for n in val]
    torch.save(modele.state_dict(), OUT / f"{nom}.pt")
    rapport = {"nom": nom, "graine": args.seed, "date": datetime.now().isoformat(timespec="minutes"), "n_nuits_train": len(train), "n_nuits_val": len(val),
               "meilleure_epoque": meilleure_epoque, "historique": historique,
               "validation_bout_en_bout": mesurer_eveils(val, probas, sommeil_predit=True),
               "validation_sommeil_du_technicien": mesurer_eveils(val, probas, sommeil_predit=False),
               "par_seuil": {f"{s:.1f}": {k: v for k, v in mesurer_eveils(val, probas, seuil=s, sommeil_predit=True).items() if k in ("precision", "rappel", "f1")}
                             for s in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)},
               "duree_min": round((time.time() - t0) / 60, 1), "journal": lignes}
    (OUT / f"{nom}.json").write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    b = rapport["validation_bout_en_bout"]
    journal(f"meilleure époque {meilleure_epoque} | de bout en bout : précision {b['precision']:.3f} rappel {b['rappel']:.3f} F1 {b['f1']:.3f} | "
            f"index : Spearman {b['index']['spearman']:.3f}, erreur médiane {b['index']['erreur_absolue_mediane']:.1f}/h, biais {b['index']['biais']:+.1f}/h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
