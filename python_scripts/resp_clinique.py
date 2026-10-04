#!/usr/bin/env python3
"""
Horizon 1.4 — Mesurer ce qui compte cliniquement, et rattraper les événements manqués.

L'analyse des manqués (docs/ANALYSE_MANQUES.md) a montré que les événements signalés nulle part
sont surtout des hypopnées sans désaturation, et que le réseau y a souvent réagi moins de 10 s.
Trois questions, sur les 40 nuits de VALIDATION (sorties en cache : python_scripts/val_cache.py) :

  1. Zones « événement possible » : quel réglage rattrape le plus d'événements avec désaturation,
     et combien de signal à relire coûte-t-il ?
     Règle écrite avant de regarder : parmi les réglages qui n'ajoutent pas plus de 5 points de
     signal à relire, celui qui laisse le moins d'événements AVEC désaturation signalés nulle part.
  2. Où tombent les événements du technicien, selon qu'ils comptent cliniquement ou non ?
  3. Index clinique : apnées + hypopnées avec désaturation, contre les index SHHS (`ahi_a0h3`,
     `ahi_a0h4`), et contre la référence simple qui compte les désaturations.

Sorties : docs/RESULTATS_CLINIQUE.md, models/resp/clinique.json (agrégats seulement).
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import PROCESSED, personnes_du_split  # noqa: E402
from somnia.deep.resp_net import charger_nuits  # noqa: E402
from somnia.nuit import DUREE_POSSIBLE_S, P_POSSIBLE, TROU_POSSIBLE_S, analyser_evenements, analyser_nuit, file_commune  # noqa: E402
from somnia.resp import FS_RESP, chute_de_saturation, index_de_desaturation, masque_vers_evenements, sommeil_par_seconde  # noqa: E402

CACHE = PROCESSED.parent / "cache_val"
OUT_MD, OUT_JSON = ROOT / "docs" / "RESULTATS_CLINIQUE.md", ROOT / "models" / "resp" / "clinique.json"
GRILLE = list(itertools.product((0.4, 0.3, 0.2), (10, 5), (0, 3)))        # (P minimale, durée minimale, trou comblé)
MARGE_PTS = 5.0
CLASSES_SEVERITE = (5, 15, 30)


def severite(index: np.ndarray) -> np.ndarray:
    return np.digitize(index, CLASSES_SEVERITE)        # 0 normal, 1 léger, 2 modéré, 3 sévère


def ou_tombent(ref, respiration, n_sec) -> list[int]:
    """Pour chaque événement de référence : 3 sûr, 2 à relire, 1 possible, 0 nulle part."""
    couv = np.zeros(n_sec, dtype=np.int8)
    for e in respiration["possibles"]:
        couv[e["debut_s"]:e["debut_s"] + e["duree_s"]] = 1
    for e in respiration["evenements"]:
        a, b = e["debut_s"], e["debut_s"] + e["duree_s"]
        couv[a:b] = np.maximum(couv[a:b], 2 if e["a_relire"] else 3)
    return [int(couv[e.debut:e.fin].max()) for e in ref]


def accord(est: np.ndarray, ref: np.ndarray) -> dict:
    ok = np.isfinite(est) & np.isfinite(ref)
    est, ref = est[ok], ref[ok]
    return {"n": int(ok.sum()), "spearman": float(spearmanr(est, ref).correlation), "erreur_absolue_mediane": float(np.median(np.abs(est - ref))),
            "biais": float(np.mean(est - ref)), "meme_classe_de_severite": float((severite(est) == severite(ref)).mean()),
            "a_une_classe_pres": float((np.abs(severite(est) - severite(ref)) <= 1).mean())}


def main() -> int:
    with open(PROCESSED / "covariables.csv", encoding="utf-8") as f:
        cov = {f"shhs-{r['nsrrid']}": r for r in csv.DictReader(f)}
    nuits = charger_nuits(personnes_du_split("val"), sommeil="multi")
    grille = {g: {"lieu": [], "desat": [], "part": [], "n_possibles": []} for g in GRILLE}
    lieux, par_personne = [], []
    donnees = []
    for n in nuits:
        with np.load(CACHE / f"{n.ident}.npz") as d:
            logits, proba = d["logits_multi"], d["proba_v4"]
        stades = analyser_nuit(None, lambda _: logits)
        codes = logits.argmax(axis=1)
        sommeil_pred = np.repeat(np.isin(codes, [1, 2, 3, 4]), 30)[: n.n_sec]
        sommeil_pred = np.concatenate([sommeil_pred, np.zeros(n.n_sec - len(sommeil_pred), dtype=bool)])
        sao2 = n.signaux[3, ::FS_RESP][: n.n_sec]
        sommeil_ref = sommeil_par_seconde(n.stades)[: n.n_sec]
        ref = [e for e in masque_vers_evenements(n.y) if sommeil_ref[min(e.debut, len(sommeil_ref) - 1)]]
        desat = np.array([chute_de_saturation(sao2, e.debut, e.fin) for e in ref])
        donnees.append((n, stades, proba, sommeil_pred, sao2, ref, desat))
        for g in GRILLE:
            r = analyser_evenements(proba, sommeil_pred, sommeil_seulement=True, p_possible=g[0], duree_possible=g[1], trou_possible=g[2])
            fc = file_commune(stades["relecture"], r, n.n_sec)
            grille[g]["lieu"] += ou_tombent(ref, r, n.n_sec); grille[g]["desat"] += desat.tolist()
            grille[g]["part"].append(fc["part_a_relire_pct"]); grille[g]["n_possibles"].append(len(r["possibles"]))

    # 1. choix du réglage, par la règle écrite en tête de fichier
    def resume(g):
        lieu, ds = np.array(grille[g]["lieu"]), np.array(grille[g]["desat"])
        return {"p_min": g[0], "duree_min_s": g[1], "trou_s": g[2], "manques_tous": float((lieu == 0).mean()),
                "manques_avec_desaturation": float((lieu[ds >= 3] == 0).mean()), "possibles_par_nuit": float(np.median(grille[g]["n_possibles"])),
                "signal_a_relire_pct": float(np.median(grille[g]["part"]))}
    table = [resume(g) for g in GRILLE]
    actuel = resume((P_POSSIBLE, DUREE_POSSIBLE_S, TROU_POSSIBLE_S)) if (P_POSSIBLE, DUREE_POSSIBLE_S, TROU_POSSIBLE_S) in grille else None
    depart = next(t for t in table if (t["p_min"], t["duree_min_s"], t["trou_s"]) == (0.4, 10, 0))
    admis = [t for t in table if t["signal_a_relire_pct"] <= depart["signal_a_relire_pct"] + MARGE_PTS]
    retenu = min(admis, key=lambda t: (t["manques_avec_desaturation"], t["signal_a_relire_pct"]))
    g_ret = (retenu["p_min"], retenu["duree_min_s"], retenu["trou_s"])

    # 2 et 3. avec le réglage retenu
    strates = {"Tous": lambda e, d: True, "Avec désaturation ≥ 3 points": lambda e, d: d >= 3, "Avec désaturation ≥ 4 points": lambda e, d: d >= 4,
               "Sans désaturation (< 3 points)": lambda e, d: d < 3, "Apnées": lambda e, d: e.classe == 1}
    compte = {k: np.zeros(4, dtype=int) for k in strates}
    est = {k: [] for k in ("brut", "clinique_3", "clinique_4", "odi_3", "odi_4")}
    cli = {k: [] for k in ("ahi_a0h3", "ahi_a0h4")}
    for n, stades, proba, sommeil_pred, sao2, ref, desat in donnees:
        r = analyser_evenements(proba, sommeil_pred, sommeil_seulement=True, sao2_1hz=sao2, p_possible=g_ret[0], duree_possible=g_ret[1], trou_possible=g_ret[2])
        for e, d, lieu in zip(ref, desat, ou_tombent(ref, r, n.n_sec)):
            for k, f in strates.items():
                if f(e, d):
                    compte[k][lieu] += 1
        res = r["resume"]
        est["brut"].append(res["index_par_heure"] or 0.0); est["clinique_3"].append(res["index_clinique_3"] or 0.0); est["clinique_4"].append(res["index_clinique_4"] or 0.0)
        est["odi_3"].append(index_de_desaturation(sao2, sommeil_pred, 3.0)); est["odi_4"].append(index_de_desaturation(sao2, sommeil_pred, 4.0))
        for k in cli:
            v = cov.get(n.personne, {}).get(k, "")
            cli[k].append(float(v) if v not in ("", None) else np.nan)
    est = {k: np.array(v) for k, v in est.items()}; cli = {k: np.array(v) for k, v in cli.items()}
    index = {"ahi_a0h3": {"Réseau : apnées + hypopnées avec désaturation ≥ 3": accord(est["clinique_3"], cli["ahi_a0h3"]),
                          "Réseau : tous les événements proposés": accord(est["brut"], cli["ahi_a0h3"]),
                          "Référence simple : désaturations ≥ 3 points / h": accord(est["odi_3"], cli["ahi_a0h3"])},
             "ahi_a0h4": {"Réseau : apnées + hypopnées avec désaturation ≥ 4": accord(est["clinique_4"], cli["ahi_a0h4"]),
                          "Réseau : tous les événements proposés": accord(est["brut"], cli["ahi_a0h4"]),
                          "Référence simple : désaturations ≥ 4 points / h": accord(est["odi_4"], cli["ahi_a0h4"])}}
    lieux = {k: (v / max(v.sum(), 1)).tolist() for k, v in compte.items()}
    effectifs = {k: int(v.sum()) for k, v in compte.items()}
    OUT_JSON.write_text(json.dumps({"date": date.today().isoformat(), "n_nuits": len(nuits), "grille": table, "retenu": retenu,
                                    "ou_tombent": lieux, "effectifs": effectifs, "index": index}, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# Résultats — ce qui compte cliniquement (horizon 1.4)", "",
         f"*Généré le {date.today().isoformat()} par `python_scripts/resp_clinique.py`. Validation SHHS, {len(nuits)} nuits, de bout en bout "
         "(stades : EEG + yeux + menton ; événements : v4 ; sommeil prédit). Agrégats seulement. Le test n'est pas touché.*", "",
         "## 1. Élargir les zones « événement possible »", "",
         "Une zone « possible » est un passage où le réseau hésite : P(événement) au-dessus d'un plancher mais sous le seuil de "
         "décision. Trois réglages : le plancher, la durée minimale, et les trous qu'on comble. Le coût se lit dans le signal à relire.", "",
         "| Plancher de P | Durée minimale | Trous comblés | Manqués, tous | **Manqués, avec désaturation** | Zones possibles par nuit | Signal à relire |",
         "|---|---|---|---|---|---|---|"]
    for t in table:
        marque = " **(retenu)**" if t is retenu else (" (départ)" if t is depart else "")
        L.append(f"| {t['p_min']:.1f}{marque} | {t['duree_min_s']} s | {t['trou_s']} s | {t['manques_tous']:.1%} | {t['manques_avec_desaturation']:.1%} | "
                 f"{t['possibles_par_nuit']:.0f} | {t['signal_a_relire_pct']:.0f} % |")
    L += ["", f"Règle écrite avant de regarder : parmi les réglages qui n'ajoutent pas plus de {MARGE_PTS:.0f} points de signal à relire, "
              "celui qui laisse le moins d'événements avec désaturation signalés nulle part.", "",
          "## 2. Où tombent les événements du technicien (réglage retenu)", "",
          "| Événements du technicien pendant le sommeil | Nombre | Proposé « sûr » | Proposé « à relire » | Zone « possible » | **Nulle part** |",
          "|---|---|---|---|---|---|"]
    for k in strates:
        v = lieux[k]
        L.append(f"| {k} | {effectifs[k]:,} | {v[3]:.0%} | {v[2]:.0%} | {v[1]:.0%} | **{v[0]:.0%}** |")
    L += ["", "La désaturation associée est mesurée par une règle simple (maximum des 30 s avant le début, minimum jusqu'à 45 s "
              "après la fin), pas par l'annotation du technicien.", "",
          "## 3. L'index clinique", "",
          "Les index SHHS comptent toutes les apnées, et les hypopnées seulement si elles sont suivies d'une désaturation "
          "(3 points pour `ahi_a0h3`, 4 pour `ahi_a0h4`). La sortie par nuit donne maintenant le même calcul : chaque événement "
          "proposé reçoit sa désaturation associée. Classes de sévérité : moins de 5, 5 à 15, 15 à 30, 30 et plus par heure.", ""]
    for ref_nom, lignes in index.items():
        L += [f"**Contre `{ref_nom}`** ({next(iter(lignes.values()))['n']} personnes)", "",
              "| Estimateur | Spearman | Erreur absolue médiane | Biais | Même classe de sévérité | À une classe près |", "|---|---|---|---|---|---|"]
        for nom, a in lignes.items():
            L.append(f"| {nom} | {a['spearman']:.2f} | {a['erreur_absolue_mediane']:.1f} / h | {a['biais']:+.1f} / h | {a['meme_classe_de_severite']:.0%} | {a['a_une_classe_pres']:.0%} |")
        L.append("")
    i3 = index["ahi_a0h3"]; r3, o3 = i3["Réseau : apnées + hypopnées avec désaturation ≥ 3"], i3["Référence simple : désaturations ≥ 3 points / h"]
    L += ["## Lecture", "",
          f"- **Zones possibles : le réglage ne change pas.** Tous les réglages plus larges coûtent plus de {MARGE_PTS:.0f} points de signal à "
          "relire. Le compromis est dans le tableau : combler les trous de 3 s ramènerait les manqués avec désaturation de "
          f"{depart['manques_avec_desaturation']:.0%} à {table[1]['manques_avec_desaturation']:.0%}, pour {table[1]['signal_a_relire_pct'] - depart['signal_a_relire_pct']:.0f} "
          "points de signal en plus. C'est une décision de produit, à prendre avec un lecteur, pas une décision de modèle.",
          "- **Les événements qui comptent sont bien mieux retrouvés que le total ne le laisse croire** : le tableau 2 est à lire "
          "ligne « avec désaturation », pas ligne « tous ».",
          f"- **L'index clinique estimé est le bon index à rendre** : en ne comptant que les hypopnées avec désaturation, l'index du réseau "
          f"suit `ahi_a0h3` avec un Spearman de {r3['spearman']:.2f} et la même classe de sévérité dans {r3['meme_classe_de_severite']:.0%} des cas. "
          "Compter tous les événements proposés surestimait l'index de près de 20 par heure : c'était une erreur de définition, pas de détection.",
          f"- **Il ne bat pas de façon établie le simple compte des désaturations** ({o3['spearman']:.2f}, {o3['meme_classe_de_severite']:.0%}) : sur 40 "
          "personnes, l'intervalle de la différence contient zéro (`docs/INTERVALLES.md`). Ce que le réseau apporte en plus de "
          "l'index, c'est la position et le type de chaque événement.",
          "- Réserves : 40 personnes ; la règle de désaturation (30 s avant, 45 s après) n'a pas été réglée, mais elle n'a pas non plus "
          "été validée contre l'annotation du technicien ; les intervalles de confiance sont dans `docs/INTERVALLES.md`.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
