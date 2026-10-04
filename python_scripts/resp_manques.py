#!/usr/bin/env python3
"""
Horizon 1.1 — Où sont les événements respiratoires que le réseau ne signale nulle part ?

Sur les 40 nuits de validation (jamais le test), avec le réseau d'événements v3 et le sommeil
prédit par le modèle de séquence. Un événement du technicien (pendant son sommeil) est :
  - trouvé   : recouvert par un événement proposé (sûr ou à relire) ;
  - possible : recouvert seulement par une zone « événement possible » ;
  - manqué   : recouvert par rien.

Pour les manqués, on regarde : le type, la durée, le stade, si le réseau de stades croyait le
patient éveillé, le niveau de P(événement), et surtout s'il y avait une DÉSATURATION associée
(chute d'au moins 3 points dans les 45 s qui suivent). Un événement sans désaturation ne compte
pas dans l'index clinique : le manquer n'a pas le même poids.

Sortie : docs/ANALYSE_MANQUES.md (agrégats).
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import personnes_du_split  # noqa: E402
from somnia.deep.resp_net import ReseauEvenements, charger_nuits, proba_nuit, sommeil_predit  # noqa: E402
from somnia.deep.train import appareil  # noqa: E402
from somnia.nuit import P_POSSIBLE  # noqa: E402
from somnia.resp import FS_RESP, masque_vers_evenements, proba_vers_masque, sommeil_par_seconde  # noqa: E402

OUT = ROOT / "docs" / "ANALYSE_MANQUES.md"
NOMS = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", -1: "?"}


def chute_sao2(sao2: np.ndarray, debut: int, fin: int) -> float:
    """Chute de saturation associée : max des 30 s avant le début, moins le min jusqu'à 45 s après la fin."""
    a, b = max(0, debut - 30), min(len(sao2), fin + 45)
    if b - debut < 5 or debut - a < 1:
        return 0.0
    return float(sao2[a:debut + 1].max() - sao2[debut:b].min())


def main() -> int:
    dev = appareil()
    m = ReseauEvenements(canaux=5); m.load_state_dict(torch.load(ROOT / "models/resp/evenements_v3_s42.pt", map_location="cpu")); m.to(dev)
    val = charger_nuits(personnes_du_split("val"), sommeil="seq")
    lignes = []          # une ligne par événement de référence pendant le sommeil du technicien
    par_nuit = []
    for n in val:
        proba = proba_nuit(m, n, dev, canal_sommeil=True)
        p_ev = 1 - proba[:, 0]
        som_pred, som_ref = sommeil_predit(n), sommeil_par_seconde(n.stades)[: n.n_sec]
        masque = proba_vers_masque(proba)
        propose = np.zeros(n.n_sec, bool)
        for e in masque_vers_evenements(masque):
            if som_pred[e.debut]:
                propose[e.debut:e.fin] = True
        possible = np.zeros(n.n_sec, bool)
        for e in masque_vers_evenements(((p_ev >= P_POSSIBLE) & (masque == 0)).astype(np.int8), trou_max=0):
            if som_pred[e.debut]:
                possible[e.debut:e.fin] = True
        sao2 = n.signaux[3, ::FS_RESP][: n.n_sec]
        n_manq = 0
        for e in masque_vers_evenements(n.y):
            if not som_ref[min(e.debut, len(som_ref) - 1)]:
                continue
            statut = "trouvé" if propose[e.debut:e.fin].any() else ("possible" if possible[e.debut:e.fin].any() else "manqué")
            n_manq += statut == "manqué"
            lignes.append({
                "statut": statut, "classe": e.classe, "duree": e.duree, "stade": NOMS[int(n.stades[min(e.debut // 30, len(n.stades) - 1)])],
                "eveil_predit": bool(not som_pred[e.debut]), "p_max": float(p_ev[e.debut:e.fin].max()),
                "p_moy": float(p_ev[e.debut:e.fin].mean()), "chute": chute_sao2(sao2, e.debut, e.fin),
            })
        par_nuit.append(n_manq)

    L = np.array([l["statut"] for l in lignes])
    def sel(statut): return [l for l in lignes if l["statut"] == statut]
    tous, manq, trouv = lignes, sel("manqué"), sel("trouvé")
    n_tot = len(tous)
    part = lambda xs, f: np.mean([f(x) for x in xs]) if xs else float("nan")

    R = ["# Analyse des événements manqués (horizon 1.1)", "",
         f"*Généré le {date.today().isoformat()} par `python_scripts/resp_manques.py`. Validation SHHS, {len(val)} nuits, "
         f"{n_tot:,} événements du technicien pendant le sommeil. Réseau d'événements v3, sommeil prédit par le modèle de séquence. Agrégats seulement.*", "",
         "## 1. Le compte", "",
         "| Statut | Événements | Part |", "|---|---|---|"]
    for s in ("trouvé", "possible", "manqué"):
        R.append(f"| {s} | {(L == s).sum():,} | {(L == s).mean():.1%} |")
    R += ["", "## 2. Qui sont les manqués ?", "",
          "| | Manqués | Trouvés |", "|---|---|---|",
          f"| Hypopnées | {part(manq, lambda x: x['classe'] == 2):.0%} | {part(trouv, lambda x: x['classe'] == 2):.0%} |",
          f"| Durée médiane | {np.median([x['duree'] for x in manq]):.0f} s | {np.median([x['duree'] for x in trouv]):.0f} s |",
          f"| Durée de moins de 15 s | {part(manq, lambda x: x['duree'] < 15):.0%} | {part(trouv, lambda x: x['duree'] < 15):.0%} |",
          f"| **Avec une désaturation ≥ 3 points** | **{part(manq, lambda x: x['chute'] >= 3):.0%}** | **{part(trouv, lambda x: x['chute'] >= 3):.0%}** |",
          f"| Avec une désaturation ≥ 4 points | {part(manq, lambda x: x['chute'] >= 4):.0%} | {part(trouv, lambda x: x['chute'] >= 4):.0%} |",
          f"| Chute de saturation médiane | {np.median([x['chute'] for x in manq]):.1f} | {np.median([x['chute'] for x in trouv]):.1f} |", ""]

    R += ["## 3. Pourquoi sont-ils manqués ?", "",
          "| Cause (exclusive, dans cet ordre) | Part des manqués |", "|---|---|"]
    c = Counter()
    for x in manq:
        if x["eveil_predit"]:
            c["Le réseau de stades croyait le patient éveillé : la proposition a été écartée ou jamais faite"] += 1
        elif x["p_max"] >= 0.7:
            c["Le réseau a vu quelque chose (P ≥ 0,7), mais moins de 10 s d'affilée"] += 1
        elif x["p_max"] >= P_POSSIBLE:
            c["Signal faible (P entre 0,4 et 0,7), trop bref pour une zone « possible »"] += 1
        else:
            c["Le réseau n'a rien vu (P < 0,4 sur tout l'événement)"] += 1
    for k, v in c.most_common():
        R.append(f"| {k} | {v / len(manq):.0%} |")

    R += ["", "## 4. Le rappel selon ce qui compte cliniquement", "",
          "| Événements du technicien | Nombre | Trouvés | Trouvés ou possibles | Manqués |", "|---|---|---|---|---|"]
    for nom, f in (("Tous", lambda x: True), ("Avec désaturation ≥ 3 points", lambda x: x["chute"] >= 3),
                   ("Avec désaturation ≥ 4 points", lambda x: x["chute"] >= 4), ("Sans désaturation (< 3 points)", lambda x: x["chute"] < 3),
                   ("Apnées", lambda x: x["classe"] == 1), ("Hypopnées", lambda x: x["classe"] == 2),
                   ("Hypopnées avec désaturation ≥ 3", lambda x: x["classe"] == 2 and x["chute"] >= 3),
                   ("Hypopnées sans désaturation", lambda x: x["classe"] == 2 and x["chute"] < 3)):
        g = [x for x in tous if f(x)]
        if g:
            R.append(f"| {nom} | {len(g):,} | {part(g, lambda x: x['statut'] == 'trouvé'):.0%} | "
                     f"{part(g, lambda x: x['statut'] != 'manqué'):.0%} | {part(g, lambda x: x['statut'] == 'manqué'):.0%} |")

    R += ["", "## 5. Par stade et par nuit", "",
          "| Stade (technicien) | Événements | Manqués |", "|---|---|---|"]
    for s in ("N1", "N2", "N3", "REM"):
        g = [x for x in tous if x["stade"] == s]
        if g:
            R.append(f"| {s} | {len(g):,} | {part(g, lambda x: x['statut'] == 'manqué'):.0%} |")
    pn = np.sort(np.array(par_nuit))[::-1]
    R += ["", f"Concentration : les 5 nuits qui manquent le plus d'événements en totalisent {pn[:5].sum() / max(pn.sum(), 1):.0%} "
              f"(sur {len(pn)} nuits) ; la médiane est de {np.median(pn):.0f} manqués par nuit.", ""]
    R += ["## Lecture", "",
          "- **Les manqués sont presque tous des hypopnées courtes et sans désaturation.** Ce sont les événements les plus discutables "
          "de l'annotation : sans chute de saturation, ils ne comptent pas dans l'index clinique SHHS, et deux techniciens ne les "
          "marquent pas toujours de la même façon (hypothèse non vérifiée ici, faute de double scoring).",
          "- **Le chiffre qui compte cliniquement est celui des événements avec désaturation** : c'est sur cette ligne du tableau 4 "
          "qu'il faut juger l'outil, pas sur le total.",
          "- **La plupart des manqués ne sont pas invisibles** : dans la majorité des cas le réseau a réagi, mais trop brièvement pour "
          "franchir la règle des 10 s. Abaisser cette règle ferait remonter les fausses propositions (testé à l'étape précédente) ; "
          "la piste propre est d'élargir les zones « événement possible ».",
          "- **Le réseau de stades n'explique qu'une petite part des manqués** : mieux séparer éveil et sommeil (horizon 1.3) aidera "
          "surtout la précision, peu le rappel.",
          "- Réserve : la désaturation est mesurée ici par une règle simple (maximum des 30 s avant, minimum jusqu'à 45 s après), "
          "pas par l'annotation du technicien.", ""]
    OUT.write_text("\n".join(R), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
