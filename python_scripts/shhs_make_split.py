#!/usr/bin/env python3
"""
Étape 2 — Découpage SHHS par personne, stratifié sur la sévérité, figé dans un fichier.

Stratification : sans cas sévères dans le test, un score ne veut rien dire. La sévérité est
estimée depuis les annotations (époques d'apnée par heure de sommeil) en quatre classes :
    < 5 : normal · 5-15 : légère · 15-30 : modérée · ≥ 30 : sévère
Chaque strate est découpée séparément avec la même graine, puis les parts sont réunies.

En visite 1, une personne = une nuit. Si la visite 2 est ajoutée un jour, l'identifiant de
personne (nsrrid) reste le même : le découpage reste valable.

Usage :
    python python_scripts/shhs_make_split.py --dry-run
    python python_scripts/shhs_make_split.py                # écrit data/splits/shhs_v1.json (hors git)

Le fichier contient des identifiants de participants : il reste hors du dépôt. Seuls les
effectifs (imprimés ici, repris dans le README) sont publiés.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.splits import decouper_par_personne, sauvegarder_decoupage, verifier_decoupage  # noqa: E402

PROCESSED = Path(os.environ.get("SHHS_PROCESSED", Path.home() / "data" / "shhs" / "processed"))
SEUILS = [(5, "normal"), (15, "legere"), (30, "moderee"), (float("inf"), "severe")]


def severite(iah: float) -> str:
    for seuil, nom in SEUILS:
        if iah < seuil:
            return nom
    return "severe"


def decouper_stratifie(personne_vers_strate: dict[str, str], graine: int,
                       part_val: float, part_test: float) -> dict[str, list[str]]:
    par_strate = defaultdict(list)
    for p, s in personne_vers_strate.items():
        par_strate[s].append(p)
    decoupage = {"train": [], "val": [], "test": []}
    for s in sorted(par_strate):
        ids = par_strate[s]
        if len(ids) < 3:                      # strate trop petite : tout en entraînement, on le dit
            print(f"  strate {s} : {len(ids)} personne(s) seulement -> entraînement")
            decoupage["train"] += sorted(ids)
            continue
        d = decouper_par_personne(ids, graine, part_val, part_test)
        for k in decoupage:
            decoupage[k] += d[k]
    decoupage = {k: sorted(v) for k, v in decoupage.items()}
    verifier_decoupage(decoupage)
    return decoupage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default=str(PROCESSED))
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--graine", type=int, default=42)
    parser.add_argument("--part-val", type=float, default=0.15)
    parser.add_argument("--part-test", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cohorte = Path(args.processed).expanduser() / "cohorte.csv"
    with open(cohorte, encoding="utf-8") as f:
        lignes = [r for r in csv.DictReader(f) if r.get("exclue") != "True"]
    if not lignes:
        sys.exit("aucune nuit retenue dans cohorte.csv")

    # Une personne peut avoir plusieurs nuits : sévérité = moyenne de ses nuits
    iah_par_personne = defaultdict(list)
    for r in lignes:
        h = float(r["temps_sommeil_h"])
        iah_par_personne[r["personne"]].append(int(r["n_apnee_epoques"]) / h if h > 0 else 0.0)
    strates = {p: severite(sum(v) / len(v)) for p, v in iah_par_personne.items()}

    print(f"{len(lignes)} nuits retenues, {len(strates)} personnes")
    print("  sévérité (époques d'apnée / h de sommeil) :", dict(Counter(strates.values())))
    dec = decouper_stratifie(strates, args.graine, args.part_val, args.part_test)
    for k, v in dec.items():
        print(f"  {k:5s} : {len(v):4d} personnes  {dict(Counter(strates[p] for p in v))}")

    if args.dry_run:
        print("[dry-run] rien écrit")
        return 0
    contenu = {
        "nom": f"shhs_v{args.version}", "cree_le": date.today().isoformat(), "graine": args.graine,
        "parts": {"val": args.part_val, "test": args.part_test},
        "stratification": "sévérité estimée depuis les annotations : <5, 5-15, 15-30, ≥30 époques d'apnée / h",
        "regle": "découpage sur l'identifiant de la PERSONNE (nsrrid), visites confondues",
        "effectifs": {k: len(v) for k, v in dec.items()},
        "taches": {"shhs": dec},
    }
    chemin = ROOT / "data" / "splits" / f"{contenu['nom']}.json"
    sauvegarder_decoupage(chemin, contenu)
    print(f"Écrit : {chemin.relative_to(ROOT)} (hors git)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
