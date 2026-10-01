#!/usr/bin/env python3
"""
Somnia — Téléchargement d'un sous-ensemble SHHS depuis le NSRR.

Remplace le brouillon ~/Downloads/03_download_shhs.py (juin 2026) :
  - le jeton est lu dans la variable d'environnement NSRR_TOKEN, jamais écrit ici ;
  - la destination est ~/data/shhs/raw (hors iCloud, hors dépôt git), modifiable
    par la variable SHHS_DIR ;
  - le script refuse d'écrire dans le dépôt ou dans un dossier iCloud.

Usage :
    export NSRR_TOKEN="..."                 # https://sleepdata.org/token
    python python_scripts/shhs_download.py --dry-run
    python python_scripts/shhs_download.py                      # pilote : ~10 nuits
    python python_scripts/shhs_download.py --pattern "*-20000*"  # ~100 nuits

Prérequis : pip install sleepecg   (déjà dans requirements-research.txt)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = Path.home() / "data" / "shhs" / "raw"

# Sous-dossiers NSRR (visite 1). Les XML "nsrr" contiennent stades ET événements.
SUBFOLDERS = {
    "edf": "polysomnography/edfs/{visit}",
    "xml": "polysomnography/annotations-events-nsrr/{visit}",
}


def _check_destination(dest: Path) -> None:
    resolved = dest.resolve()
    if "Mobile Documents" in str(resolved) or "iCloud" in str(resolved):
        sys.exit(f"Refus : {resolved} est synchronisé iCloud. Utilise ~/data/shhs.")
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return  # hors du dépôt : OK
    sys.exit(f"Refus : {resolved} est dans le dépôt git. Utilise ~/data/shhs.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Télécharge un sous-ensemble SHHS")
    parser.add_argument("--pattern", default="*-200000*",
                        help='motif de nom de fichier (défaut "*-200000*" ≈ 10 nuits)')
    parser.add_argument("--visit", default="shhs1", choices=["shhs1", "shhs2"])
    parser.add_argument("--dest", default=os.environ.get("SHHS_DIR", str(DEFAULT_DIR)),
                        help="dossier de destination (défaut : $SHHS_DIR ou ~/data/shhs/raw)")
    parser.add_argument("--only", choices=["edf", "xml"], default=None,
                        help="ne télécharger qu'un type de fichier")
    parser.add_argument("--dry-run", action="store_true",
                        help="affiche ce qui serait fait, sans rien télécharger")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser()
    _check_destination(dest)

    token = os.environ.get("NSRR_TOKEN")
    if not token and not args.dry_run:
        sys.exit("NSRR_TOKEN absent. Fais : export NSRR_TOKEN=\"...\" (jeton sur sleepdata.org/token)")

    kinds = [args.only] if args.only else ["edf", "xml"]
    plan = [(k, SUBFOLDERS[k].format(visit=args.visit)) for k in kinds]

    print(f"Destination : {dest}")
    print(f"Visite      : {args.visit}   Motif : {args.pattern}")
    for kind, sub in plan:
        print(f"  - {kind:3s} ← shhs/{sub}")
    if args.dry_run:
        print("[dry-run] rien téléchargé." + ("" if token else " (NSRR_TOKEN non défini)"))
        return 0

    from sleepecg import download_nsrr, set_nsrr_token  # import tardif : dry-run sans sleepecg

    dest.mkdir(parents=True, exist_ok=True)
    set_nsrr_token(token)
    for kind, sub in plan:
        print(f"\nTéléchargement {kind} …")
        download_nsrr(db_slug="shhs", subfolder=sub, pattern=args.pattern, data_dir=str(dest))

    n_edf = len(list(dest.rglob("*.edf")))
    n_xml = len(list(dest.rglob("*-nsrr.xml")))
    print(f"\nTerminé. EDF : {n_edf}   XML : {n_xml}   dans {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
