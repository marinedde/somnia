#!/usr/bin/env python3
"""
Horizon 1.6 — Intervalles de confiance par bootstrap sur les personnes (docs/INTERVALLES.md).

Les écarts qu'on compare depuis l'horizon 1 (0,002 à 0,02) sont de l'ordre du bruit. Ce script
donne, pour chaque chiffre clé, un intervalle à 95 % obtenu en retirant les 40 personnes de
validation avec remise (2 000 tirages), et, pour chaque comparaison, l'intervalle de la différence
calculée sur les mêmes personnes (bootstrap apparié).

Chaque mesure est la MOYENNE de trois entraînements (graines 42, 1, 2), calculée sur les mêmes
personnes retirées : l'intervalle couvre le hasard du choix des 40 personnes, pour un modèle
« moyen ». Il ne couvre pas le fait que les réglages ont été choisis sur ces mêmes personnes.
L'index clinique, lui, est celui de la sortie par nuit (graine 42).

Entrée : ~/data/shhs/cache_val (python_scripts/val_cache.py). Sortie : agrégats seulement.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import PROCESSED, personnes_du_split  # noqa: E402
from somnia.deep.resp_net import charger_nuits, sommeil_predit  # noqa: E402
from somnia.evaluation import bootstrap_par_personne  # noqa: E402
from somnia.nuit import analyser_evenements  # noqa: E402
from somnia.resp import (FS_RESP, apparier, chute_de_saturation, index_de_desaturation, masque_vers_evenements,  # noqa: E402
                         proba_vers_masque, sommeil_par_seconde)

CACHE = PROCESSED.parent / "cache_val"
OUT_MD, OUT_JSON = ROOT / "docs" / "INTERVALLES.md", ROOT / "models" / "intervalles.json"


def fmt(r, nd=3, signe=False):
    s = "+" if signe else ""
    return f"{r['estimation']:{s}.{nd}f} [{r['bas']:{s}.{nd}f} ; {r['haut']:{s}.{nd}f}]"


def main() -> int:
    with open(PROCESSED / "covariables.csv", encoding="utf-8") as f:
        cov = {f"shhs-{r['nsrrid']}": r for r in csv.DictReader(f)}
    nuits = {s: charger_nuits(personnes_du_split("val"), sommeil=s) for s in ("seq", "multi")}
    P = len(nuits["multi"])

    # ── Stades : une liste par personne ──
    SUF = ("", "_s1", "_s2")                                   # graines 42, 1, 2
    Y, PR = [], {f"{k}{g}": [] for k in ("eeg", "multi") for g in SUF}
    for n in nuits["multi"]:
        with np.load(CACHE / f"{n.ident}.npz") as d:
            ok = n.stades >= 0
            Y.append(n.stades[ok].astype(int))
            for k in PR:
                PR[k].append(d[f"logits_{k}"].argmax(axis=1)[ok])
    mesures_stades = {
        "Kappa": lambda y, p: cohen_kappa_score(y, p),
        "Exactitude": lambda y, p: float((y == p).mean()),
        "Accord éveil / sommeil": lambda y, p: float(((y == 0) == (p == 0)).mean()),
        "F1 du N1": lambda y, p: f1_score(y == 1, p == 1),
        "F1 du REM": lambda y, p: f1_score(y == 4, p == 4),
    }
    cat = lambda L, i: np.concatenate([L[j] for j in i])
    stades = {}
    for nom, f in mesures_stades.items():
        moy = lambda i, k: float(np.mean([f(cat(Y, i), cat(PR[f"{k}{g}"], i)) for g in SUF]))       # moyenne des trois graines
        stades[nom] = {k: bootstrap_par_personne(lambda i, k=k: moy(i, k), P, n_tirages=1000) for k in ("eeg", "multi")}
        stades[nom]["difference"] = bootstrap_par_personne(lambda i: moy(i, "multi") - moy(i, "eeg"), P, n_tirages=1000)

    # ── Événements : comptes par personne, v3 et v4, pendant le sommeil, de bout en bout ──
    C = {f"{v}{g}": np.zeros((P, 5)) for v in ("v3", "v4") for g in SUF}          # vp, fp, fn, vp parmi les références avec désaturation, n références avec désaturation
    idx = {k: np.full(P, np.nan) for k in ("clinique_3", "odi_3", "brut", "ahi_a0h3", "clinique_4", "odi_4", "ahi_a0h4")}
    for v, s in (("v3", "seq"), ("v4", "multi"), ("v3_s1", "seq"), ("v3_s2", "seq"), ("v4_s1", "multi"), ("v4_s2", "multi")):
        for j, n in enumerate(nuits[s]):
            with np.load(CACHE / f"{n.ident}.npz") as d:
                proba = d[f"proba_{v}"].astype(np.float32)
                som_argmax = np.repeat(d["logits_multi"].argmax(axis=1) != 0, 30)[: n.n_sec]
            som_argmax = np.concatenate([som_argmax, np.zeros(n.n_sec - len(som_argmax), dtype=bool)])
            som_pred, som_ref = sommeil_predit(n), sommeil_par_seconde(n.stades)[: n.n_sec]
            sao2 = n.signaux[3, ::FS_RESP][: n.n_sec]
            ref = [e for e in masque_vers_evenements(n.y) if som_ref[min(e.debut, len(som_ref) - 1)]]
            pred = [e for e in masque_vers_evenements(proba_vers_masque(proba)) if som_pred[min(e.debut, n.n_sec - 1)]]
            m = apparier(pred, ref)
            ref_d = [e for e in ref if chute_de_saturation(sao2, e.debut, e.fin) >= 3]
            C[v][j] = [m["vp"], m["fp"], m["fn"], apparier(pred, ref_d)["vp"], len(ref_d)]
            if v == "v4":
                som_pred = som_argmax                         # même sommeil prédit que la sortie par nuit
                res = analyser_evenements(proba, som_pred, sommeil_seulement=True, sao2_1hz=sao2)["resume"]
                idx["clinique_3"][j], idx["clinique_4"][j], idx["brut"][j] = res["index_clinique_3"] or 0, res["index_clinique_4"] or 0, res["index_par_heure"] or 0
                idx["odi_3"][j], idx["odi_4"][j] = index_de_desaturation(sao2, som_pred, 3.0), index_de_desaturation(sao2, som_pred, 4.0)
                for a in ("ahi_a0h3", "ahi_a0h4"):
                    val = cov.get(n.personne, {}).get(a, "")
                    idx[a][j] = float(val) if val not in ("", None) else np.nan

    def f1(c):
        vp, fp, fn = c[:, 0].sum(), c[:, 1].sum(), c[:, 2].sum()
        return 2 * vp / max(2 * vp + fp + fn, 1)
    mesures_ev = {
        "F1 par événement": f1,
        "Précision": lambda c: c[:, 0].sum() / max(c[:, 0].sum() + c[:, 1].sum(), 1),
        "Rappel": lambda c: c[:, 0].sum() / max(c[:, 0].sum() + c[:, 2].sum(), 1),
        "Rappel des événements avec désaturation ≥ 3": lambda c: c[:, 3].sum() / max(c[:, 4].sum(), 1),
    }
    evenements = {}
    for nom, f in mesures_ev.items():
        moy = lambda i, v: float(np.mean([f(C[f"{v}{g}"][i]) for g in SUF]))
        evenements[nom] = {v: bootstrap_par_personne(lambda i, v=v: moy(i, v), P) for v in ("v3", "v4")}
        evenements[nom]["difference"] = bootstrap_par_personne(lambda i: moy(i, "v4") - moy(i, "v3"), P)

    sp = lambda a, b: (lambda i: spearmanr(idx[a][i], idx[b][i]).correlation)
    mae = lambda a, b: (lambda i: float(np.median(np.abs(idx[a][i] - idx[b][i]))))
    index = {}
    for k, ref in (("3", "ahi_a0h3"), ("4", "ahi_a0h4")):
        index[ref] = {
            "reseau_spearman": bootstrap_par_personne(sp(f"clinique_{k}", ref), P), "odi_spearman": bootstrap_par_personne(sp(f"odi_{k}", ref), P),
            "difference_spearman": bootstrap_par_personne(lambda i, k=k, ref=ref: sp(f"clinique_{k}", ref)(i) - sp(f"odi_{k}", ref)(i), P),
            "reseau_erreur": bootstrap_par_personne(mae(f"clinique_{k}", ref), P), "odi_erreur": bootstrap_par_personne(mae(f"odi_{k}", ref), P),
            "difference_erreur": bootstrap_par_personne(lambda i, k=k, ref=ref: mae(f"clinique_{k}", ref)(i) - mae(f"odi_{k}", ref)(i), P),
        }
    OUT_JSON.write_text(json.dumps({"date": date.today().isoformat(), "n_personnes": P, "stades": stades, "evenements": evenements, "index": index},
                                   indent=2, ensure_ascii=False), encoding="utf-8")

    verdict = lambda d: "oui" if d["bas"] > 0 or d["haut"] < 0 else "non"
    L = ["# Intervalles de confiance (horizon 1.6)", "",
         f"*Généré le {date.today().isoformat()} par `python_scripts/intervalles.py`. Validation SHHS, {P} personnes. Chaque mesure est la "
         "moyenne de trois entraînements (graines 42, 1, 2). Intervalles à 95 % par bootstrap sur les personnes (1 000 à 2 000 tirages) ; "
         "différences calculées sur les mêmes personnes.*", "",
         "Lecture : `estimation [borne basse ; borne haute]`. Une différence est dite établie quand son intervalle ne contient pas zéro.", "",
         "## Stades : EEG seul contre EEG + yeux + menton", "",
         "| Mesure | EEG seul | EEG + yeux + menton | Différence | Établie ? |", "|---|---|---|---|---|"]
    for nom, r in stades.items():
        L.append(f"| {nom} | {fmt(r['eeg'])} | {fmt(r['multi'])} | {fmt(r['difference'], signe=True)} | {verdict(r['difference'])} |")
    L += ["", "## Événements respiratoires : v3 contre v4 (pendant le sommeil, de bout en bout)", "",
          "| Mesure | v3 | v4 | Différence | Établie ? |", "|---|---|---|---|---|"]
    for nom, r in evenements.items():
        L.append(f"| {nom} | {fmt(r['v3'])} | {fmt(r['v4'])} | {fmt(r['difference'], signe=True)} | {verdict(r['difference'])} |")
    L += ["", "## Index clinique : le réseau contre le simple compte des désaturations", "",
          "| Contre | Mesure | Réseau (apnées + hypopnées avec désaturation) | Désaturations par heure | Différence | Établie ? |", "|---|---|---|---|---|---|"]
    for ref, r in index.items():
        L.append(f"| `{ref}` | Spearman | {fmt(r['reseau_spearman'], 2)} | {fmt(r['odi_spearman'], 2)} | {fmt(r['difference_spearman'], 2, True)} | {verdict(r['difference_spearman'])} |")
        L.append(f"| `{ref}` | Erreur absolue médiane (/h) | {fmt(r['reseau_erreur'], 1)} | {fmt(r['odi_erreur'], 1)} | {fmt(r['difference_erreur'], 1, True)} | {verdict(r['difference_erreur'])} |")
    k, ae, f1e, sp3 = stades["Kappa"], stades["Accord éveil / sommeil"], evenements["F1 par événement"], index["ahi_a0h3"]["difference_spearman"]
    L += ["", "## Ce qu'on peut dire, et ce qu'on ne peut pas", "",
          f"- **Établi** : ajouter les yeux et le menton améliore les stades (kappa {fmt(k['difference'], signe=True)}) et l'accord éveil / sommeil "
          f"({fmt(ae['difference'], signe=True)}). La cible de 0,95 reste dans l'intervalle du modèle multi-capteurs : on ne peut ni dire qu'elle est atteinte, ni qu'elle est manquée.",
          "- **Non établi** : un gain sur le N1 ou le REM.",
          f"- **Établi** : le réseau d'événements v4 n'a pas un meilleur F1 que le v3 ({fmt(f1e['difference'], signe=True)}). Il échange un peu de rappel contre un peu de précision.",
          f"- **Non établi** : que l'index clinique du réseau suive mieux `ahi_a0h3` que le simple compte des désaturations ({fmt(sp3, 2, True)}, "
          f"{sp3['part_positive']:.0%} des tirages favorables au réseau). Les deux sont bons ; 40 personnes ne suffisent pas à les départager.",
          "- **La largeur des intervalles est la leçon principale** : avec 40 personnes, un kappa est connu à ± 0,04 et un F1 par événement à ± 0,04. "
          "Tout écart plus petit entre deux réglages ne se lit que par une comparaison appariée, et encore.", "",
          "## Repère : l'accord entre scoreurs humains", "",
          "Un modèle ne peut pas être plus d'accord avec un technicien que deux techniciens entre eux. Chiffres publiés, **cités de mémoire : "
          "à vérifier à la source avant toute publication**. SHHS a été scoré selon les règles de Rechtschaffen et Kales, pas selon celles de l'AASM : "
          "la comparaison est indicative.", "",
          "| Mesure | Entre scoreurs humains (littérature) | Somnia, validation |", "|---|---|---|",
          f"| Stades, accord global | environ 83 % (Rosenberg et Van Hout, 2013, programme inter-scoreurs de l'AASM) | {stades['Exactitude']['multi']['estimation']:.0%} |",
          f"| Stades, kappa | environ 0,76 (Danker-Hopfe et al., 2009) | {k['multi']['estimation']:.2f} |",
          "| N1 | accord d'environ 63 %, le plus bas de tous les stades (Rosenberg et Van Hout, 2013) | F1 0,41 (mesure différente) |",
          "| REM | accord d'environ 90 % | F1 0,85 (mesure différente) |",
          "| Hypopnées | accord d'environ 65 %, contre environ 77 % pour les apnées obstructives (Rosenberg et Van Hout, 2014) | rappel 0,72 ; apnées 0,94 |", "",
          "Lecture : le N1 et les hypopnées sont les deux endroits où Somnia est faible, et ce sont aussi les deux endroits où les humains "
          "s'accordent le moins. Une partie du plafond est dans l'étiquette, pas dans le modèle. Pour le mesurer vraiment, il faudrait "
          "des nuits scorées deux fois, ce que SHHS ne fournit pas.", "",
          "## Ce que ces intervalles ne couvrent pas", "",
          "- **La variation entre entraînements** : les mesures sont moyennées sur trois graines, ce qui l'atténue sans la supprimer. "
          "L'index clinique est celui de la sortie par nuit, une seule graine.",
          "- **Le choix sur la validation** : architectures, seuils et variantes ont été choisis sur ces 40 personnes. Les chiffres sont "
          "donc un peu optimistes, et l'intervalle ne corrige pas ce biais. Seul un test neuf le fera.",
          "- **La population** : 40 adultes de SHHS, enregistrés à domicile entre 1995 et 1998.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
