#!/usr/bin/env python3
"""
Tâches 1.6 et 1.10 — Regarder les fichiers SHHS avant d'écrire la moindre chaîne de traitement.

Pour chaque enregistrement de ~/data/shhs/raw :
  - l'en-tête EDF lu directement (sans MNE) : chaque canal avec SA fréquence d'échantillonnage,
    son unité, la durée ;
  - les annotations : époques par stade, événements par type, index d'apnées-hypopnées estimé.

Usage :
    python python_scripts/shhs_inspect.py                 # tableau + canaux du premier fichier
    python python_scripts/shhs_inspect.py --canaux        # canaux de TOUS les fichiers
    python python_scripts/shhs_inspect.py --csv docs/pilote_shhs.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.shhs import lire_annotations, resume  # noqa: E402

DEFAULT_DIR = Path(os.environ.get("SHHS_DIR", Path.home() / "data" / "shhs" / "raw"))


def entete_edf(chemin: Path) -> dict:
    """Lit l'en-tête EDF (format texte à largeur fixe) : la fréquence de CHAQUE canal.

    MNE ramène tous les canaux à une fréquence commune ; ici on lit ce que dit le fichier.
    Spécification : 256 octets d'en-tête général, puis 256 octets par signal répartis en champs.
    """
    with open(chemin, "rb") as f:
        h = f.read(256)
        duree_record = float(h[244:252])
        n_records = int(h[236:244])
        ns = int(h[252:256])
        champs = {"label": 16, "transducer": 80, "unit": 8, "phys_min": 8, "phys_max": 8,
                  "dig_min": 8, "dig_max": 8, "prefilter": 80, "n_samples": 8, "reserved": 32}
        valeurs = {}
        for nom, largeur in champs.items():
            bloc = f.read(largeur * ns)
            valeurs[nom] = [bloc[i * largeur:(i + 1) * largeur].decode("latin-1").strip() for i in range(ns)]
    canaux = []
    for i in range(ns):
        n = int(valeurs["n_samples"][i])
        canaux.append({
            "label": valeurs["label"][i],
            "fs_hz": n / duree_record if duree_record else None,
            "unite": valeurs["unit"][i],
            "phys": f"{valeurs['phys_min'][i]}..{valeurs['phys_max'][i]}",
            "prefiltre": valeurs["prefilter"][i][:40],
        })
    return {"duree_h": n_records * duree_record / 3600, "n_records": n_records,
            "duree_record_s": duree_record, "canaux": canaux}


def afficher_canaux(chemin: Path) -> None:
    e = entete_edf(chemin)
    print(f"\n{chemin.name} — {e['duree_h']:.2f} h, {e['n_records']} blocs de {e['duree_record_s']} s")
    print(f"  {'canal':18s} {'Hz':>7s}  {'unité':6s} {'plage physique':22s} préfiltre")
    for c in e["canaux"]:
        print(f"  {c['label']:18s} {c['fs_hz']:7.1f}  {c['unite']:6s} {c['phys']:22s} {c['prefiltre']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--canaux", action="store_true", help="afficher les canaux de tous les fichiers")
    parser.add_argument("--csv", default=None, help="écrire le tableau du pilote dans ce fichier")
    args = parser.parse_args()

    racine = Path(args.dir).expanduser()
    edfs = sorted(racine.rglob("*.edf"))
    xmls = {p.name.replace("-nsrr.xml", ""): p for p in racine.rglob("*-nsrr.xml")}
    if not edfs:
        print(f"Aucun EDF sous {racine}. Lance d'abord python_scripts/shhs_download.py.")
        return 1
    print(f"{len(edfs)} EDF, {len(xmls)} XML sous {racine}")

    # 1. Canaux et fréquences
    for p in (edfs if args.canaux else edfs[:1]):
        afficher_canaux(p)
    if not args.canaux:
        # Les montages diffèrent-ils d'un fichier à l'autre ? On compare les signatures.
        signatures = Counter(tuple((c["label"], c["fs_hz"]) for c in entete_edf(p)["canaux"]) for p in edfs)
        print(f"\n{len(signatures)} montage(s) distinct(s) sur {len(edfs)} fichiers"
              + ("" if len(signatures) == 1 else " — relance avec --canaux pour voir les différences"))

    # 2. Tableau du pilote
    lignes = []
    for p in edfs:
        ident = p.stem
        ligne = {"enregistrement": ident, "duree_edf_h": round(entete_edf(p)["duree_h"], 2)}
        if ident in xmls:
            ligne.update(resume(lire_annotations(xmls[ident])))
        else:
            ligne["annotations"] = "manquantes"
        lignes.append(ligne)

    colonnes = ["enregistrement", "duree_edf_h", "duree_h", "temps_sommeil_h", "n_Wake", "n_N1", "n_N2",
                "n_N3", "n_REM", "n_Inconnu", "n_apnees_hypopnees", "iah_estime"]
    autres = sorted({k for l in lignes for k in l} - set(colonnes))
    colonnes += autres
    print("\n" + " | ".join(colonnes))
    for l in lignes:
        print(" | ".join(str(l.get(c, "")) for c in colonnes))

    if args.csv:
        import csv
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=colonnes)
            w.writeheader()
            w.writerows(lignes)
        print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
