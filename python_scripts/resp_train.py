#!/usr/bin/env python3
"""
Détection d'événements respiratoires — références et réseau, sur la validation (test fermé).

  --baseline : la référence par désaturation (ODI à 3 % et 4 %), par personne. C'est le chiffre
               à battre pour l'index : chaque événement qui compte cliniquement désature.
  --petit-lot : le réseau doit apprendre 16 fenêtres par cœur.
  (défaut)   : entraînement complet, arrêt anticipé sur le F1 par événement de la validation.

Mesures : par seconde, par événement (recouvrement, puis IoU ≥ 0,3), rappel des apnées et des
hypopnées séparément, et par personne : index estimé contre index annoté (erreur, biais, limites
d'accord) et contre l'index clinique SHHS (Spearman).

Sorties : models/resp/*.json (versionnés), models/resp/*.pt (hors git).
"""

from __future__ import annotations

import argparse
import csv
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

from somnia.deep.data import PROCESSED, personnes_du_split  # noqa: E402
from somnia.deep.model import n_parametres  # noqa: E402
from somnia.deep.resp_net import (  # noqa: E402
    SEUIL_EVENEMENT, FenetresResp, ReseauEvenements, charger_nuits, mesurer_nuits, poids_des_classes, proba_nuit, proba_vers_masque,
)
from somnia.deep.train import appareil, fixer_graines  # noqa: E402
from somnia.resp import FS_RESP, index_de_desaturation, index_par_heure, masque_vers_evenements, sommeil_par_seconde  # noqa: E402

OUT = ROOT / "models" / "resp"


def covariables() -> dict:
    cov = {}
    chemin = PROCESSED / "covariables.csv"
    if chemin.exists():
        with open(chemin, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cov[f"shhs-{r['nsrrid']}"] = r
    return cov


def contre_clinique(detail: list[dict], cov: dict, cle_estime: str = "index_estime") -> dict:
    """Spearman et erreur de l'index estimé contre les index cliniques SHHS."""
    from scipy.stats import spearmanr
    out = {}
    for ref in ("ahi_a0h3a", "ahi_a0h4"):
        est = [d[cle_estime] for d in detail if d["personne"] in cov and cov[d["personne"]][ref] not in ("", None)]
        cli = [float(cov[d["personne"]][ref]) for d in detail if d["personne"] in cov and cov[d["personne"]][ref] not in ("", None)]
        if len(est) > 3:
            est, cli = np.array(est), np.array(cli)
            out[ref] = {"spearman": float(spearmanr(est, cli).correlation),
                        "erreur_absolue_mediane": float(np.median(np.abs(est - cli))), "biais": float(np.mean(est - cli))}
    return out


def reference_desaturation(nuits, cov) -> dict:
    res = {}
    for chute in (3.0, 4.0):
        detail = []
        for n in nuits:
            sommeil = sommeil_par_seconde(n.stades)[: n.n_sec]
            sao2 = n.signaux[3, ::FS_RESP][: n.n_sec]
            ref = masque_vers_evenements(n.y)
            detail.append({"personne": n.personne, "index_estime": index_de_desaturation(sao2, sommeil, chute),
                           "index_reference": index_par_heure(ref, sommeil)})
        est = np.array([d["index_estime"] for d in detail]); ann = np.array([d["index_reference"] for d in detail])
        from scipy.stats import spearmanr
        res[f"odi_{chute:g}"] = {"contre_annote": {"spearman": float(spearmanr(est, ann).correlation),
                                                   "erreur_absolue_mediane": float(np.median(np.abs(est - ann))),
                                                   "biais": float(np.mean(est - ann))},
                                 "contre_clinique": contre_clinique(detail, cov)}
    return res


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--petit-lot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--nom", default=None, help="étiquette de l'expérience (défaut : evenements)")
    parser.add_argument("--canal-sommeil", action="store_true", help="5e canal : probabilité de sommeil prédite par le réseau de stades")
    parser.add_argument("--largeur", type=int, default=64)
    parser.add_argument("--blocs", type=int, default=6, help="nombre de blocs dilatés (dilatations 1, 2, 4, ... )")
    parser.add_argument("--poids", choices=["equilibre", "racine", "aucun"], default="equilibre")
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction des nuits d'entraînement (courbe d'apprentissage)")
    parser.add_argument("--sommeil", choices=["cnn", "seq", "multi"], default="cnn", help="réseau de stades qui fournit le sommeil prédit")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cov = covariables()
    lignes = []

    def journal(msg):
        print(msg, flush=True); lignes.append(msg)

    t0 = time.time()
    val = charger_nuits(personnes_du_split("val"), sommeil=args.sommeil)
    journal(f"validation : {len(val)} nuits, {sum(n.n_sec for n in val) / 3600:.0f} h")

    if args.baseline:
        r = {"date": datetime.now().isoformat(timespec="minutes"), "n_personnes": len(val), **reference_desaturation(val, cov)}
        (OUT / "reference_desaturation.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        for k in ("odi_3", "odi_4"):
            journal(f"{k} : vs index annoté {r[k]['contre_annote']} | vs clinique {r[k]['contre_clinique']}")
        return 0

    fixer_graines(args.seed)
    dev = appareil()
    from somnia.deep.data import sous_ensemble_de_personnes
    train = charger_nuits(sous_ensemble_de_personnes(personnes_du_split("train"), args.fraction, args.seed), sommeil=args.sommeil)
    journal(f"entraînement : {len(train)} nuits, {sum(n.n_sec for n in train) / 3600:.0f} h, chargées en {time.time() - t0:.0f} s")
    canaux = 5 if args.canal_sommeil else 4
    dilatations = tuple(2 ** (i % 6) for i in range(args.blocs))
    modele = ReseauEvenements(canaux=canaux, largeur=args.largeur, dilatations=dilatations).to(dev)
    journal(f"appareil {dev.type} | {n_parametres(modele):,} paramètres | canaux {canaux} | largeur {args.largeur} | dilatations {dilatations} | poids {args.poids}")
    poids = poids_des_classes(train)
    if args.poids == "racine":
        poids = torch.sqrt(poids)
    elif args.poids == "aucun":
        poids = torch.ones_like(poids)
    poids = poids.to(dev)
    journal(f"poids des classes (rien, apnée, hypopnée) : {[round(float(w), 1) for w in poids]}")
    perte = nn.CrossEntropyLoss(weight=poids)

    if args.petit_lot:
        ds = FenetresResp(train[:4], canal_sommeil=args.canal_sommeil)
        # 16 fenêtres qui contiennent des événements
        idx = [k for k in range(len(ds)) if ds[k][1].sum() > 0][:16]
        x = torch.stack([ds[k][0] for k in idx]).to(dev); y = torch.stack([ds[k][1] for k in idx]).to(dev)
        opt = torch.optim.Adam(modele.parameters(), lr=2e-3)
        modele.train(); pertes = []
        for _ in range(400):
            opt.zero_grad(set_to_none=True); loss = perte(modele(x), y); loss.backward(); opt.step(); pertes.append(loss.item())
        modele.eval()
        with torch.no_grad():
            ex = float((modele(x).argmax(1) == y).float().mean())
        journal(f"petit lot (16 fenêtres) : perte {pertes[0]:.3f} -> {pertes[-1]:.4f}, exactitude par seconde {ex:.1%}")
        ok = pertes[-1] < 0.1 * pertes[0]
        print("OK : le réseau apprend par cœur" if ok else "ÉCHEC : la perte ne descend pas")
        return 0 if ok else 1

    loader = DataLoader(FenetresResp(train, entrainement=True, graine=args.seed, canal_sommeil=args.canal_sommeil),
                        batch_size=args.batch, shuffle=True, num_workers=0)
    a_le_sommeil = all(n.p_sommeil is not None for n in val)

    def evaluer():
        masques = [proba_vers_masque(proba_nuit(modele, nv, dev, canal_sommeil=args.canal_sommeil)) for nv in val]
        tout = mesurer_nuits(val, masques)
        som = mesurer_nuits(val, masques, pendant_le_sommeil=True) if a_le_sommeil else None
        return tout, som
    opt = torch.optim.AdamW(modele.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    historique, meilleur, meilleur_etat, sans_progres = [], -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        modele.train(); te = time.time(); total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = perte(modele(x), y); loss.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 5.0); opt.step()
            total += loss.item() * len(y); n += len(y)
        sched.step()
        m, ms = evaluer()
        # Critère d'arrêt : le F1 par événement sur la tâche « pendant le sommeil » quand le sommeil prédit existe
        f1 = (ms or m)["par_evenement"]["recouvrement"]["f1"]
        historique.append({"epoque": ep, "perte_train": total / n, "f1_selection": f1,
                           "f1_tous_evenements": m["par_evenement"]["recouvrement"]["f1"],
                           "f1_seconde": m["par_seconde"]["f1_evenement"], "spearman_index": (ms or m)["par_personne"]["spearman"],
                           "duree_s": round(time.time() - te, 1)})
        journal(f"époque {ep:2d} | perte {total / n:.4f} | val : F1 sommeil {f1:.3f}, F1 tous {m['par_evenement']['recouvrement']['f1']:.3f}, "
                f"Spearman index {(ms or m)['par_personne']['spearman']:.3f} | {time.time() - te:.0f} s")
        if f1 > meilleur + 1e-4:
            meilleur, sans_progres = f1, 0
            meilleur_etat = {k: v.detach().cpu().clone() for k, v in modele.state_dict().items()}
            meilleure_epoque = ep
        else:
            sans_progres += 1
            if sans_progres >= args.patience:
                journal("arrêt anticipé"); break
    modele.load_state_dict(meilleur_etat)
    m, ms = evaluer()
    m["contre_clinique"] = contre_clinique(m["par_personne"]["detail"], cov)
    detail = m["par_personne"].pop("detail")       # identifiants : pas dans le fichier versionné
    if ms is not None:
        ms["contre_clinique"] = contre_clinique(ms["par_personne"]["detail"], cov)
        detail = ms["par_personne"].pop("detail")
    nom = f"{args.nom or 'evenements'}_s{args.seed}"
    torch.save(modele.state_dict(), OUT / f"{nom}.pt")
    np.savez_compressed(OUT / f"{nom}_val_detail.npz", personne=np.array([d["personne"] for d in detail]),
                        index_reference=np.array([d["index_reference"] for d in detail]),
                        index_estime=np.array([d["index_estime"] for d in detail]))
    rapport = {"nom": nom, "graine": args.seed, "date": datetime.now().isoformat(timespec="minutes"), "appareil": dev.type,
               "n_nuits_train": len(train), "n_nuits_val": len(val), "parametres": n_parametres(modele),
               "seuil_evenement": SEUIL_EVENEMENT,
               "config": {"sommeil": args.sommeil, "canal_sommeil": args.canal_sommeil, "largeur": args.largeur, "blocs": args.blocs, "poids": args.poids,
                          "fraction": args.fraction},
               "validation_pendant_le_sommeil": ms,
               "meilleure_epoque": meilleure_epoque, "historique": historique, "validation": m,
               "duree_min": round((time.time() - t0) / 60, 1), "journal": lignes}
    (OUT / f"{nom}.json").write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    journal(f"meilleure époque {meilleure_epoque} | par événement : {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m['par_evenement']['recouvrement'].items()})}")
    journal(f"IoU ≥ 0,3 : F1 {m['par_evenement']['iou_0.3']['f1']:.3f} | rappel apnées {m['par_evenement']['rappel_apnees']:.3f}, hypopnées {m['par_evenement']['rappel_hypopnees']:.3f}")
    journal(f"par personne : Spearman {m['par_personne']['spearman']:.3f}, erreur médiane {m['par_personne']['erreur_absolue_mediane']:.1f}/h, biais {m['par_personne']['biais']:+.1f}/h")
    journal(f"contre l'index clinique : {m['contre_clinique']}")
    if ms is not None:
        e = ms["par_evenement"]
        journal(f"PENDANT LE SOMMEIL (bout en bout) : précision {e['recouvrement']['precision']:.3f} rappel {e['recouvrement']['rappel']:.3f} "
                f"F1 {e['recouvrement']['f1']:.3f} | IoU0,3 F1 {e['iou_0.3']['f1']:.3f} | apnées {e['rappel_apnees']:.3f} hypopnées {e['rappel_hypopnees']:.3f} "
                f"| index Spearman {ms['par_personne']['spearman']:.3f} err {ms['par_personne']['erreur_absolue_mediane']:.1f}/h biais {ms['par_personne']['biais']:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
