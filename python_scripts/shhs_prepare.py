#!/usr/bin/env python3
"""
Étape 2 — Prépare toutes les nuits SHHS téléchargées : un .npz par nuit + diagramme de cohorte.

Relançable : les nuits déjà préparées sont sautées (sauf --force). À lancer pendant ou après
le téléchargement.

Usage :
    python python_scripts/shhs_prepare.py                 # ~/data/shhs/raw -> ~/data/shhs/processed
    python python_scripts/shhs_prepare.py --force
    python python_scripts/shhs_prepare.py --limit 9       # pour tester

Sorties (hors dépôt, identifiants de participants) :
    ~/data/shhs/processed/<id>.npz
    ~/data/shhs/processed/cohorte.csv      une ligne par nuit, retenue ou exclue, avec le motif
    ~/data/shhs/processed/cohorte.json     effectifs du diagramme de cohorte (publiable)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.shhs_prepare import preparer_nuit, resume_vers_dict  # noqa: E402

RAW = Path(os.environ.get("SHHS_DIR", Path.home() / "data" / "shhs" / "raw"))
PROCESSED = Path(os.environ.get("SHHS_PROCESSED", Path.home() / "data" / "shhs" / "processed"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(RAW))
    parser.add_argument("--out", default=str(PROCESSED))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw, out = Path(args.raw).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    edfs = sorted(raw.rglob("*.edf"))[: args.limit]
    xmls = {p.name.replace("-nsrr.xml", ""): p for p in raw.rglob("*-nsrr.xml")}

    # Résumés déjà connus (relance) : on les recharge pour ne pas perdre les motifs d'exclusion
    csv_path = out / "cohorte.csv"
    anciens = {}
    if csv_path.exists() and not args.force:
        with open(csv_path, encoding="utf-8") as f:
            anciens = {r["enregistrement"]: r for r in csv.DictReader(f)}

    lignes, sans_xml = [], []
    for i, edf in enumerate(edfs, 1):
        ident = edf.stem
        if ident not in xmls:
            sans_xml.append(ident)
            continue
        cible = out / f"{ident}.npz"
        if not args.force and ident in anciens and (cible.exists() or anciens[ident].get("exclue") == "True"):
            lignes.append(anciens[ident])
            continue
        logging.info("[%d/%d] %s", i, len(edfs), ident)
        r = resume_vers_dict(preparer_nuit(edf, xmls[ident], cible))
        lignes.append({k: ("" if v is None else v) for k, v in r.items()})

    if sans_xml:
        logging.warning("%d EDF sans XML (téléchargement en cours ?) : %s", len(sans_xml), sans_xml[:5])

    colonnes = ["enregistrement", "personne", "n_epoques", "temps_sommeil_h", "n_apnee_epoques",
                "part_ecg_plat", "canal_eeg", "canal_ecg", "exclue", "motif_exclusion"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colonnes)
        w.writeheader()
        for l in lignes:
            w.writerow({c: l.get(c, "") for c in colonnes})

    retenues = [l for l in lignes if str(l.get("exclue")) != "True"]
    motifs = Counter(str(l.get("motif_exclusion", "")).split(" (")[0].split(" sur ")[0]
                     for l in lignes if str(l.get("exclue")) == "True")
    cohorte = {
        "nuits_telechargees": len(edfs),
        "sans_annotations": len(sans_xml),
        "exclues": dict(motifs),
        "retenues": len(retenues),
        "personnes_retenues": len({l["personne"] for l in retenues}),
        "epoques_retenues": int(sum(int(l["n_epoques"]) for l in retenues)),
        "epoques_apnee": int(sum(int(l["n_apnee_epoques"]) for l in retenues)),
    }
    (out / "cohorte.json").write_text(json.dumps(cohorte, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDiagramme de cohorte")
    print(f"  Nuits téléchargées            n = {cohorte['nuits_telechargees']}")
    print(f"    sans annotations            n = {cohorte['sans_annotations']}")
    for m, n in motifs.items():
        print(f"    retirées : {m:28s} n = {n}")
    print(f"  Nuits retenues                n = {cohorte['retenues']}  "
          f"({cohorte['personnes_retenues']} personnes, {cohorte['epoques_retenues']:,} époques, "
          f"{cohorte['epoques_apnee']:,} avec apnée)")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
