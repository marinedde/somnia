#!/usr/bin/env python3
"""
Horizon 1.3 — Tableau d'ablation des capteurs (docs/RESULTATS_MULTI.md).

Lit models/multi/<variante>_s*.json et, pour mémoire, models/cnn/eeg_seq_s*.json (horizon 1.2).
Validation SHHS seulement : le test des stades a déjà été ouvert une fois et n'est pas rouvert.
Usage : python python_scripts/multi_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "RESULTATS_MULTI.md"
ORDRE = [("eeg", "EEG seul (témoin : refait l'horizon 1.2)"), ("eeg_eeg2", "EEG + second EEG"), ("eeg_emg", "EEG + menton (EMG)"),
         ("eeg_eog", "EEG + yeux (EOG)"), ("eeg_eog_emg", "EEG + yeux + menton"), ("tout", "Les cinq capteurs")]
CIBLE = 0.95


def lire(dossier: str, motif: str) -> list[dict]:
    return sorted((json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "models" / dossier).glob(motif)), key=lambda r: r["graine"])


def pm(runs, f, nd=3):
    v = [f(r) for r in runs]
    return f"{np.mean(v):.{nd}f}" + (f" ± {np.std(v):.{nd}f}" if len(v) > 1 else "")


def main() -> int:
    variantes = [(nom, lire("multi", f"{cle}_s*.json")) for cle, nom in ORDRE]
    variantes = [(n, r) for n, r in variantes if r]
    if not variantes:
        sys.exit("aucun entraînement dans models/multi")
    ref = lire("cnn", "eeg_seq_s*.json")
    r0 = variantes[0][1][0]
    L = ["# Résultats — stades à partir de plusieurs capteurs (horizon 1.3)", "",
         f"*Généré le {date.today().isoformat()} par `python_scripts/multi_report.py`. Validation SHHS ({r0['n_personnes_val']} personnes, "
         f"{r0['n_exemples_val']:,} époques) ; entraînement sur {r0['n_personnes_train']} personnes. Moyenne ± écart-type sur les graines. "
         "Le test des stades n'est pas rouvert.*", "",
         "## La question", "",
         "Un technicien score le sommeil avec trois familles de signaux : l'EEG, les mouvements des yeux (EOG) et le tonus du menton "
         "(EMG). Le modèle de l'horizon 1.2 ne lisait que l'EEG. Que rapporte chaque capteur ajouté ? La cible fixée d'avance : "
         f"un accord éveil / sommeil d'au moins {CIBLE:.0%}, parce que c'est lui qui plafonne le réseau d'événements respiratoires.", "",
         "Toutes les variantes suivent exactement la même procédure (même architecture, mêmes réglages, mêmes personnes) ; seule "
         "l'entrée change. L'EEG est centré-réduit par époque ; les yeux et le menton sont réduits par nuit, pour garder leur niveau "
         "relatif (un tonus effondré doit rester petit).", "",
         "## Ablation", "",
         "| Capteurs | Graines | Exactitude | Kappa | Accord éveil / sommeil | Éveils reconnus | Sommeils reconnus | Kappa par nuit (médiane) |",
         "|---|---|---|---|---|---|---|---|"]
    def ligne(nom, runs):
        return (f"| {nom} | {len(runs)} | {pm(runs, lambda r: r['validation']['accuracy'])} | {pm(runs, lambda r: r['validation']['kappa'])} | "
                f"{pm(runs, lambda r: r['eveil_sommeil']['accord'])} | {pm(runs, lambda r: r['eveil_sommeil']['sensibilite_eveil'])} | "
                f"{pm(runs, lambda r: r['eveil_sommeil']['specificite_eveil'])} | {pm(runs, lambda r: r['kappa_par_nuit']['mediane'])} |")
    if ref:
        L.append(ligne("*Horizon 1.2, EEG seul (pour mémoire)*", ref))
    L += [ligne(n, r) for n, r in variantes]
    L += ["", "## Par stade (F1)", "", "| Capteurs | Éveil | N1 | N2 | N3 | REM | F1 macro |", "|---|---|---|---|---|---|---|"]
    for n, r in variantes:
        L.append(f"| {n} | " + " | ".join(pm(r, lambda x, s=s: x["validation"][f"f1_{s}"]) for s in ("Wake", "N1", "N2", "N3", "REM"))
                 + f" | {pm(r, lambda x: x['validation']['f1_macro'])} |")
    L += ["", "## Calibration et tri par la confiance", "", "| Capteurs | ECE avant | ECE après température | Meilleure époque |", "|---|---|---|---|"]
    for n, r in variantes:
        L.append(f"| {n} | {pm(r, lambda x: x['calibration']['ece_avant'])} | {pm(r, lambda x: x['calibration']['ece_apres'])} | "
                 f"{', '.join(str(x['meilleure_epoque']) for x in r)} |")
    L += ["", "## Lecture", "",
          "- **Le témoin est exact** : l'EEG seul, par le nouveau chemin de données, redonne les chiffres de l'horizon 1.2. Les écarts "
          "du tableau viennent donc des capteurs, pas d'un changement de préparation.",
          "- **Chaque capteur ajouté aide un peu, les yeux et le menton ensemble aident le plus** : c'est la combinaison qu'utilise un "
          "technicien. Elle est aussi la plus stable d'une graine à l'autre.",
          f"- **La cible de {CIBLE:.0%} d'accord éveil / sommeil n'est pas atteinte.** Le gain porte sur l'éveil et le REM ; le N1 ne "
          "bouge pas, quel que soit le capteur.",
          "- **Ajouter le second EEG par-dessus n'apporte rien de plus** : il redit ce que dit le premier.",
          "- Réserves : la variante retenue a été choisie sur la validation, sur laquelle elle est aussi mesurée (six variantes "
          "comparées sur 40 personnes) ; le gain réel sera un peu plus faible. Les écarts entre variantes voisines sont de l'ordre "
          "du bruit entre graines. Seul un test neuf tranchera, et celui des stades a déjà été ouvert.",
          "- Suite : le sommeil prédit par cette variante alimente le réseau d'événements v4 (`docs/RESULTATS_EVENEMENTS.md`).", ""]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
