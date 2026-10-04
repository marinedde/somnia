#!/usr/bin/env python3
"""Horizon 2.1 — Tableau des micro-éveils (docs/RESULTATS_EVEILS.md). Lit models/eveils/*.json. Validation seulement."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "RESULTATS_EVEILS.md"


def main() -> int:
    R = sorted((json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "models" / "eveils").glob("eveils_s*.json")), key=lambda r: r["graine"])
    if not R:
        sys.exit("aucun entraînement dans models/eveils")
    pm = lambda f, nd=2: f"{np.mean([f(r) for r in R]):.{nd}f} ± {np.std([f(r) for r in R]):.{nd}f}"
    L = ["# Résultats — détection des micro-éveils (horizon 2.1)", "",
         f"*Généré le {date.today().isoformat()} par `python_scripts/eveils_report.py`. Entraînement : {R[0]['n_nuits_train']} nuits ; validation : "
         f"{R[0]['n_nuits_val']} nuits, {R[0]['validation_bout_en_bout']['n_reference']:,} micro-éveils du technicien pendant le sommeil. "
         f"{len(R)} graines. Le test n'est pas touché.*", "",
         "## Ce qui est mesuré", "",
         "Le réseau lit l'EEG, le second EEG et le menton à 100 Hz, par fenêtres de 2 minutes, et rend pour chaque seconde la probabilité d'un "
         "micro-éveil. Les secondes au-dessus de 0,5 sont regroupées en événements d'au moins 3 s. Un événement proposé est juste s'il recouvre "
         "un micro-éveil du technicien (appariement un à un).", "",
         "| Mesure | De bout en bout (sommeil prédit) | Avec le sommeil du technicien |", "|---|---|---|"]
    for nom, cle in (("Précision", "precision"), ("Rappel", "rappel"), ("F1 par événement", "f1")):
        L.append(f"| {nom} | {pm(lambda r: r['validation_bout_en_bout'][cle])} | {pm(lambda r: r['validation_sommeil_du_technicien'][cle])} |")
    for nom, cle, nd in (("Index par personne : Spearman", "spearman", 2), ("Index : erreur absolue médiane (/h)", "erreur_absolue_mediane", 1), ("Index : biais (/h)", "biais", 1)):
        L.append(f"| {nom} | {pm(lambda r: r['validation_bout_en_bout']['index'][cle], nd)} | {pm(lambda r: r['validation_sommeil_du_technicien']['index'][cle], nd)} |")
    L += ["", f"Index médian du technicien : {R[0]['validation_bout_en_bout']['index']['mediane_reference']:.0f} micro-éveils par heure de sommeil.", "",
          "## Le seuil : précision contre rappel (de bout en bout)", "",
          "| Seuil sur P(micro-éveil) | Précision | Rappel | F1 |", "|---|---|---|---|"]
    for s in R[0]["par_seuil"]:
        L.append(f"| {s}{' (retenu, fixé d’avance)' if s == '0.5' else ''} | " + " | ".join(f"{np.mean([r['par_seuil'][s][k] for r in R]):.2f}" for k in ("precision", "rappel", "f1")) + " |")
    L += ["", "## Lecture", "",
          "- **Quatre micro-éveils sur cinq sont retrouvés, et six propositions sur dix sont justes.** C'est une première version, sans réglage.",
          "- **L'écart entre les deux colonnes vient du réseau de stades** : avec le sommeil du technicien, la précision monte de près de dix points. "
          "Les fausses propositions sont en grande partie des éveils francs que le réseau de stades a pris pour du sommeil.",
          "- **L'index par personne est surestimé de quelques unités par heure** et ordonne les personnes moyennement. Il ne remplace pas un comptage.",
          "- **Repère** : les micro-éveils sont l'événement sur lequel les scoreurs humains s'accordent le moins. Une partie des « fausses » "
          "propositions sont probablement des micro-éveils qu'un autre technicien aurait marqués. Non vérifiable ici, faute de double scoring.",
          "- **Dans la sortie par nuit**, les micro-éveils sont proposés avec leur confiance et leur index, mais ne sont pas ajoutés à la file de "
          "relecture : à ce niveau de précision, ils alourdiraient la relecture sans la guider.",
          "- **Pistes** : donner au réseau le sommeil prédit et les événements respiratoires (un micro-éveil suit souvent une apnée), lier chaque "
          "hypopnée à son micro-éveil pour l'index `ahi_a0h3a`, et allonger le contexte.", ""]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
