#!/usr/bin/env python3
"""
Détection d'événements — préparation : pour chaque nuit retenue de la cohorte, les quatre
canaux respiratoires à 10 Hz et les étiquettes à la seconde.

Entrée  : ~/data/shhs/raw (EDF + XML), ~/data/shhs/processed/cohorte.csv (les 272 nuits retenues).
Sortie  : ~/data/shhs/processed_resp/<id>.npz
            signaux  (4, n_sec*10) float32 : flux, thorax, abdomen (unités d'origine), SaO2 nettoyée (%)
            y        (n_sec,) int8          : 0 rien, 1 apnée, 2 hypopnée
            stades   (n_epoques,) int8
            meta
          ~/data/shhs/processed_resp/cohorte_resp.csv (motifs d'exclusion, part de SaO2 invalide)

Règles d'exclusion supplémentaires, écrites avant de regarder les résultats :
  - un des quatre canaux absent ;
  - saturation invalide (capteur décollé) sur plus de 30 % de la nuit.

Usage : python python_scripts/resp_prepare.py [--force] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.resp import CANAUX_RESP, FS_RESP, ORDRE_CANAUX, etiquettes_par_seconde, nettoyer_sao2, trouver, vers_10hz  # noqa: E402
from somnia.shhs import lire_annotations  # noqa: E402
from somnia.shhs_prepare import lire_canal_natif  # noqa: E402
from somnia.subjects import identifiant_personne  # noqa: E402

RAW = Path(os.environ.get("SHHS_DIR", Path.home() / "data" / "shhs" / "raw"))
PROCESSED = Path(os.environ.get("SHHS_PROCESSED", Path.home() / "data" / "shhs" / "processed"))
OUT = PROCESSED.parent / "processed_resp"
SAO2_INVALIDE_MAX = 0.30


def preparer(edf: Path, xml: Path, cible: Path) -> dict:
    import mne

    ident = edf.stem
    base = {"enregistrement": ident, "personne": identifiant_personne(edf), "exclue": False, "motif": "",
            "canal_flux": "", "part_sao2_invalide": "", "n_secondes": 0, "n_apnees": 0, "n_hypopnees": 0}
    ann = lire_annotations(xml)
    noms = mne.io.read_raw_edf(str(edf), preload=False, verbose="error").ch_names
    canaux = {k: trouver(noms, CANAUX_RESP[k]) for k in ORDRE_CANAUX}
    manquants = [k for k, v in canaux.items() if v is None]
    if manquants:
        return {**base, "exclue": True, "motif": f"canal absent : {','.join(manquants)}"}
    n_sec = int(ann.n_epoques * ann.duree_epoque)
    signaux = []
    part_invalide = 0.0
    for k in ORDRE_CANAUX:
        x, fs = lire_canal_natif(edf, canaux[k])
        if k == "sao2":
            # MNE suppose des volts pour un canal sans unité : la saturation est déjà en %
            brut = x if np.nanmax(x) > 1.5 else x * 100
            part_invalide = float(((brut < 50) | (brut > 100)).mean())
            x = nettoyer_sao2(brut)
        x = vers_10hz(x, fs)
        total = n_sec * FS_RESP
        x = x[:total] if len(x) >= total else np.concatenate([x, np.full(total - len(x), x[-1] if len(x) else 0.0)])
        signaux.append(x.astype(np.float32))
    if part_invalide > SAO2_INVALIDE_MAX:
        return {**base, "exclue": True, "motif": f"SaO2 invalide sur {part_invalide:.0%} de la nuit",
                "canal_flux": canaux["flux"], "part_sao2_invalide": round(part_invalide, 3)}
    y = etiquettes_par_seconde(ann, n_sec)
    # flux et ceintures : en « volts » MNE (unités arbitraires) -> remis à une échelle lisible
    S = np.stack(signaux)
    S[:3] *= 1e6
    cible.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cible, signaux=S.astype(np.float32), y=y, stades=ann.stades.astype(np.int8),
                        meta=np.array([ident, base["personne"], canaux["flux"], str(FS_RESP)]))
    ref = [e for e in ann.evenements if e.nom.lower() in ("obstructive apnea", "central apnea", "mixed apnea", "hypopnea")]
    return {**base, "canal_flux": canaux["flux"], "part_sao2_invalide": round(part_invalide, 3), "n_secondes": n_sec,
            "n_apnees": sum(1 for e in ref if e.nom.lower() != "hypopnea"), "n_hypopnees": sum(1 for e in ref if e.nom.lower() == "hypopnea")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    with open(PROCESSED / "cohorte.csv", encoding="utf-8") as f:
        retenues = [r["enregistrement"] for r in csv.DictReader(f) if r.get("exclue") != "True"][: args.limit]
    edfs = {p.stem: p for p in RAW.rglob("*.edf")}
    xmls = {p.name.replace("-nsrr.xml", ""): p for p in RAW.rglob("*-nsrr.xml")}
    OUT.mkdir(parents=True, exist_ok=True)
    lignes = []
    for i, ident in enumerate(retenues, 1):
        cible = OUT / f"{ident}.npz"
        try:
            r = preparer(edfs[ident], xmls[ident], cible) if (args.force or not cible.exists()) else None
        except Exception as e:
            r = {"enregistrement": ident, "personne": identifiant_personne(ident), "exclue": True,
                 "motif": f"erreur de lecture ({type(e).__name__})", "canal_flux": "", "part_sao2_invalide": "",
                 "n_secondes": 0, "n_apnees": 0, "n_hypopnees": 0}
        if r is None:   # déjà préparée : on relit le résumé précédent plus bas
            continue
        lignes.append(r)
        if i % 25 == 0 or i == len(retenues):
            print(f"  {i}/{len(retenues)} nuits", flush=True)

    chemin_csv = OUT / "cohorte_resp.csv"
    anciens = {}
    if chemin_csv.exists() and not args.force:
        with open(chemin_csv, encoding="utf-8") as f:
            anciens = {r["enregistrement"]: r for r in csv.DictReader(f)}
    for l in lignes:
        anciens[l["enregistrement"]] = {k: str(v) for k, v in l.items()}
    colonnes = ["enregistrement", "personne", "exclue", "motif", "canal_flux", "part_sao2_invalide", "n_secondes", "n_apnees", "n_hypopnees"]
    with open(chemin_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colonnes); w.writeheader()
        for k in sorted(anciens):
            w.writerow({c: anciens[k].get(c, "") for c in colonnes})

    tous = list(anciens.values())
    gardees = [l for l in tous if l["exclue"] != "True"]
    motifs = Counter(l["motif"].split(" sur ")[0].split(" (")[0] for l in tous if l["exclue"] == "True")
    print(f"\nNuits de la cohorte : {len(tous)} | retenues pour les événements : {len(gardees)} | exclues : {dict(motifs)}")
    print(f"canal de flux : {dict(Counter(l['canal_flux'] for l in gardees))}")
    print(f"événements annotés : {sum(int(l['n_apnees']) for l in gardees):,} apnées, {sum(int(l['n_hypopnees']) for l in gardees):,} hypopnées")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
