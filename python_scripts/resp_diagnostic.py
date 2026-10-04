#!/usr/bin/env python3
"""
Détection d'événements — diagnostic des erreurs sur la validation (jamais le test).

Avant d'améliorer un réseau, regarder où il se trompe :
  1. selon le capteur de flux (NEW AIR ou AIRFLOW) ;
  2. les fausses propositions : pendant l'éveil ? courtes ? près d'un vrai événement ?
  3. les événements manqués : quel type, quelle durée, quel stade ;
  4. l'effet d'un vote de plusieurs graines (moyenne des probabilités) ;
  5. avec --courbe : la courbe d'apprentissage (25 %, 50 %, 100 % des nuits d'entraînement).

Usage :
    python python_scripts/resp_diagnostic.py
    python python_scripts/resp_diagnostic.py --courbe
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import personnes_du_split  # noqa: E402
from somnia.deep.resp_net import RESP_DIR, ReseauEvenements, charger_nuits, mesurer_nuits, proba_nuit  # noqa: E402
from somnia.deep.train import appareil  # noqa: E402
from somnia.resp import apparier, masque_vers_evenements, proba_vers_masque, sommeil_par_seconde  # noqa: E402

MODELES = ROOT / "models" / "resp"


def charger(nom: str, dev):
    m = ReseauEvenements(); m.load_state_dict(torch.load(MODELES / nom, map_location="cpu")); return m.to(dev)


def resume(m: dict) -> str:
    e = m["par_evenement"]
    return (f"précision {e['recouvrement']['precision']:.3f} rappel {e['recouvrement']['rappel']:.3f} F1 {e['recouvrement']['f1']:.3f} "
            f"| IoU0,3 F1 {e['iou_0.3']['f1']:.3f} | apnées {e['rappel_apnees']:.3f} hypopnées {e['rappel_hypopnees']:.3f} "
            f"| index Spearman {m['par_personne']['spearman']:.3f} err {m['par_personne']['erreur_absolue_mediane']:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--courbe", action="store_true")
    args = parser.parse_args()
    dev = appareil()
    val = charger_nuits(personnes_du_split("val"))
    with open(RESP_DIR / "cohorte_resp.csv", encoding="utf-8") as f:
        flux_de = {r["personne"]: r["canal_flux"] for r in csv.DictReader(f)}

    graines = [p.name for p in sorted(MODELES.glob("evenements_s*.pt"))]
    probas = {g: [proba_nuit(charger(g, dev), n, dev) for n in val] for g in graines}
    print(f"validation : {len(val)} nuits ; modèles : {graines}")
    for g in graines:
        print(f"  {g:22s} {resume(mesurer_nuits(val, [proba_vers_masque(p) for p in probas[g]]))}")
    moy = [np.mean([probas[g][i] for g in graines], axis=0) for i in range(len(val))]
    print(f"  {'vote des ' + str(len(graines)) + ' graines':22s} {resume(mesurer_nuits(val, [proba_vers_masque(p) for p in moy]))}")

    ref0 = graines[0]
    masques = [proba_vers_masque(p) for p in probas[ref0]]

    print("\n1. Selon le capteur de flux")
    for capteur in sorted(set(flux_de[n.personne] for n in val)):
        idx = [i for i, n in enumerate(val) if flux_de[n.personne] == capteur]
        print(f"  {capteur:8s} ({len(idx):2d} nuits) {resume(mesurer_nuits([val[i] for i in idx], [masques[i] for i in idx]))}")

    print("\n2. Fausses propositions")
    fp_eveil = fp_total = fp_proche = 0
    durees_fp, durees_vp = [], []
    manques = Counter(); manques_duree, trouves_duree = [], []
    manques_stade, total_stade = Counter(), Counter()
    noms = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", -1: "?"}
    for n, mp in zip(val, masques):
        ref, pred = masque_vers_evenements(n.y), masque_vers_evenements(mp)
        sommeil = sommeil_par_seconde(n.stades)[: n.n_sec]
        ref_mask = n.y > 0
        for e in pred:
            if ref_mask[e.debut:e.fin].any():
                durees_vp.append(e.duree)
            else:
                fp_total += 1; durees_fp.append(e.duree)
                fp_eveil += int(not sommeil[min(e.debut, len(sommeil) - 1)])
                a, b = max(0, e.debut - 60), min(n.n_sec, e.fin + 60)
                fp_proche += int(ref_mask[a:b].any())
        pred_mask = mp > 0
        for e in ref:
            st = noms[int(n.stades[min(e.debut // 30, len(n.stades) - 1)])]
            total_stade[st] += 1
            if pred_mask[e.debut:e.fin].any():
                trouves_duree.append(e.duree)
            else:
                manques[e.classe] += 1; manques_duree.append(e.duree); manques_stade[st] += 1
    print(f"  {fp_total} fausses propositions : {fp_eveil / fp_total:.0%} commencent pendant l'éveil (technicien), "
          f"{fp_proche / fp_total:.0%} sont à moins de 60 s d'un vrai événement")
    print(f"  durée médiane : fausses {np.median(durees_fp):.0f} s, justes {np.median(durees_vp):.0f} s ; "
          f"fausses de moins de 15 s : {np.mean(np.array(durees_fp) < 15):.0%}")

    print("\n3. Événements manqués")
    tot = sum(manques.values())
    print(f"  {tot} manqués : apnées {manques[1]}, hypopnées {manques[2]}")
    print(f"  durée médiane : manqués {np.median(manques_duree):.0f} s, trouvés {np.median(trouves_duree):.0f} s ; "
          f"manqués de moins de 15 s : {np.mean(np.array(manques_duree) < 15):.0%}")
    print("  taux de manqués par stade : " + ", ".join(f"{s} {manques_stade[s] / total_stade[s]:.0%} (sur {total_stade[s]})" for s in ("W", "N1", "N2", "N3", "REM") if total_stade[s]))

    if args.courbe:
        import subprocess
        print("\n5. Courbe d'apprentissage : voir resp_train.py --fraction (lancer séparément)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
