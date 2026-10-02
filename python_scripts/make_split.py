#!/usr/bin/env python3
"""
Crée le fichier de découpage par personne pour PhysioNet (tâche 0.5).

À lancer UNE fois. Le fichier est versionné ; pour changer la cohorte, on crée
une v2, on ne modifie pas la v1.

Usage :
    python python_scripts/make_split.py                 # écrit data/splits/physionet_v1.json
    python python_scripts/make_split.py --version 2 --graine 7
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.physionet import charger_tableau  # noqa: E402
from somnia.splits import decouper_par_personne, sauvegarder_decoupage  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--graine", type=int, default=42)
    parser.add_argument("--part-val", type=float, default=0.15)
    parser.add_argument("--part-test", type=float, default=0.15)
    args = parser.parse_args()

    contenu = {
        "nom": f"physionet_v{args.version}",
        "cree_le": date.today().isoformat(),
        "graine": args.graine,
        "parts": {"val": args.part_val, "test": args.part_test},
        "regle": "découpage sur l'identifiant de la PERSONNE (somnia.subjects.identifiant_personne)",
        "taches": {},
    }
    for tache in ("eeg", "ecg"):
        t = charger_tableau(PROCESSED / f"{tache}_features.npz")
        dec = decouper_par_personne(t["personne"], args.graine, args.part_val, args.part_test)
        contenu["taches"][tache] = dec
        print(f"{tache} : " + ", ".join(f"{k} = {len(v)} personnes" for k, v in dec.items()))

    chemin = ROOT / "data" / "splits" / f"{contenu['nom']}.json"
    sauvegarder_decoupage(chemin, contenu)
    print(f"Écrit : {chemin.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
