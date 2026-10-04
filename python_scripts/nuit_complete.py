#!/usr/bin/env python3
"""
La sortie par nuit complète (stades + événements + file de relecture commune), mesurée sur
les 40 nuits de VALIDATION de SHHS. Le test n'est pas touché.

Pourquoi SHHS et pas Sleep-EDF : il faut les ceintures et la saturation, que Sleep-EDF n'a pas.
Ce qui est publié : des agrégats sur 40 nuits (docs/RESULTATS_NUIT.md). Ce qui reste local,
hors git : le compte rendu et la figure d'une nuit (docs/figures_pilote/), parce que le tracé
d'un participant n'a pas à sortir de la machine.

Questions auxquelles ce script répond :
  1. Quelle part du signal reste à relire avec la file commune ?
  2. La confiance trie-t-elle bien les événements ? (précision des « sûrs » vs des « à relire »)
  3. Où sont les événements de référence : dans les sûrs, les à-relire, les possibles, ou manqués ?
  4. L'index calculé de bout en bout (sommeil PRÉDIT, événements PRÉDITS) ordonne-t-il les personnes ?

Usage : python python_scripts/nuit_complete.py [--nuit-locale 0]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import PROCESSED, normaliser_lot, personnes_du_split  # noqa: E402
from somnia.deep.model import CNN1D  # noqa: E402
from somnia.deep.resp_net import ReseauEvenements, charger_nuits, proba_nuit  # noqa: E402
from somnia.deep.train import appareil  # noqa: E402
from somnia.nuit import analyser_nuit_complete  # noqa: E402
from somnia.resp import index_par_heure, masque_vers_evenements, sommeil_par_seconde  # noqa: E402
from somnia.shhs_prepare import charger_nuit  # noqa: E402

OUT_MD = ROOT / "docs" / "RESULTATS_NUIT.md"
OUT_JSON = ROOT / "models" / "nuit_complete.json"
LOCAL = ROOT / "docs" / "figures_pilote"


def med_iqr(v, fmt="{:.0f}"):
    v = np.asarray(v, dtype=float)
    return f"{fmt.format(np.median(v))} ({fmt.format(np.percentile(v, 25))}–{fmt.format(np.percentile(v, 75))})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuit-locale", type=int, default=0, help="indice de la nuit dont on garde le détail en local")
    args = parser.parse_args()
    dev = appareil()
    # Stades : encodeur + lecture de la nuit (horizon 1.2). Événements : v3, qui reçoit le sommeil prédit par ce modèle.
    from somnia.deep.seq import NuitEEG, ReseauSequence, predire_nuit
    stades_net = ReseauSequence(); stades_net.load_state_dict(torch.load(ROOT / "models/cnn/eeg_seq_s42.pt", map_location="cpu")); stades_net.eval()
    T = json.loads((ROOT / "models/cnn/eeg_seq_s42.json").read_text(encoding="utf-8"))["calibration"]["temperature"]
    ev_net = ReseauEvenements(canaux=5); ev_net.load_state_dict(torch.load(ROOT / "models/resp/evenements_v3_s42.pt", map_location="cpu")); ev_net.to(dev)

    def predire_stades(eeg_uv):
        nuit = NuitEEG("nuit", "personne", eeg_uv.astype(np.float16), np.zeros(len(eeg_uv), dtype=np.int64))
        return predire_nuit(stades_net, nuit, torch.device("cpu"))

    cov = {}
    if (PROCESSED / "covariables.csv").exists():
        with open(PROCESSED / "covariables.csv", encoding="utf-8") as f:
            cov = {f"shhs-{r['nsrrid']}": r for r in csv.DictReader(f)}

    nuits = charger_nuits(personnes_du_split("val"), sommeil="seq")
    lignes, niveaux = [], {"sur": [0, 0], "a_relire": [0, 0]}          # [justes, total]
    paires = []                                                        # (confiance, juste) de chaque événement proposé
    ou_sont_les_ref = {"sur": 0, "a_relire": 0, "possible": 0, "manque": 0}
    for k, n in enumerate(nuits):
        eeg = charger_nuit(PROCESSED / f"{n.ident}.npz")["eeg"]                      # µV, (n_epoques, 3000)
        proba = proba_nuit(ev_net, n, dev, canal_sommeil=True)
        r = analyser_nuit_complete(eeg, predire_stades, proba, temperature=T)
        res, fc = r["respiration"]["resume"], r["file_commune"]

        # 2. la confiance trie-t-elle ? un événement proposé est « juste » s'il recouvre un événement de référence
        for e in r["respiration"]["evenements"]:
            juste = bool((n.y[e["debut_s"]:e["debut_s"] + e["duree_s"]] > 0).any())
            cle = "a_relire" if e["a_relire"] else "sur"
            niveaux[cle][0] += juste; niveaux[cle][1] += 1
            paires.append((e["confiance"], juste))
        # 3. où tombent les événements de référence
        couv = np.zeros(n.n_sec, dtype=np.int8)                                       # 1 possible, 2 à relire, 3 sûr
        for e in r["respiration"]["possibles"]:
            couv[e["debut_s"]:e["debut_s"] + e["duree_s"]] = np.maximum(couv[e["debut_s"]:e["debut_s"] + e["duree_s"]], 1)
        for e in r["respiration"]["evenements"]:
            v = 2 if e["a_relire"] else 3
            couv[e["debut_s"]:e["debut_s"] + e["duree_s"]] = np.maximum(couv[e["debut_s"]:e["debut_s"] + e["duree_s"]], v)
        sommeil_ref = sommeil_par_seconde(n.stades)[: n.n_sec]
        ref_tous = masque_vers_evenements(n.y)
        ref = [e for e in ref_tous if sommeil_ref[min(e.debut, len(sommeil_ref) - 1)]]      # ceux qui comptent dans l'index
        for e in ref:
            m = int(couv[e.debut:e.fin].max()) if e.fin > e.debut else 0
            ou_sont_les_ref[{3: "sur", 2: "a_relire", 1: "possible", 0: "manque"}[m]] += 1
        # 4. index de bout en bout vs référence (sommeil et événements du technicien)
        lignes.append({
            "personne": n.personne, "index_ref": index_par_heure(ref_tous, sommeil_ref), "index_bout_en_bout": res["index_par_heure"] or 0.0,
            "n_ref": len(ref), "n_proposes": res["n_evenements"], "n_surs": res["n_surs"], "n_a_relire": res["n_a_relire"],
            "n_possibles": res["n_possibles"], "part_stades_a_relire": r["relecture"]["part_a_relire_pct"],
            "signal_a_relire_min": fc["signal_a_relire_min"], "signal_total_min": fc["signal_total_min"], "part_a_relire": fc["part_a_relire_pct"],
            "heures_sommeil_pred": res["heures_de_sommeil"], "heures_sommeil_ref": float(sommeil_ref.sum()) / 3600,
        })
        if k == args.nuit_locale:                                                     # détail d'une nuit : local, hors git
            LOCAL.mkdir(parents=True, exist_ok=True)
            (LOCAL / "nuit_complete_exemple.json").write_text(json.dumps(
                {**{c: r[c] for c in ("statut", "indices")}, "respiration": r["respiration"]["resume"],
                 "file_commune": {c: fc[c] for c in ("n_items", "signal_a_relire_min", "signal_total_min", "part_a_relire_pct")},
                 "premiers_items": fc["items"][:25]}, indent=2, ensure_ascii=False), encoding="utf-8")

    from scipy.stats import spearmanr
    ref_i = np.array([l["index_ref"] for l in lignes]); est_i = np.array([l["index_bout_en_bout"] for l in lignes])
    total_ref = sum(ou_sont_les_ref.values())
    agg = {
        "date": date.today().isoformat(), "n_nuits": len(lignes),
        "signal_a_relire_pct": {"mediane": float(np.median([l["part_a_relire"] for l in lignes])),
                                "q1": float(np.percentile([l["part_a_relire"] for l in lignes], 25)),
                                "q3": float(np.percentile([l["part_a_relire"] for l in lignes], 75))},
        "precision_par_niveau": {k: (v[0] / v[1] if v[1] else None) for k, v in niveaux.items()},
        "effectifs_par_niveau": {k: v[1] for k, v in niveaux.items()},
        "reference_ou": {k: v / total_ref for k, v in ou_sont_les_ref.items()},
        "index_bout_en_bout": {"spearman": float(spearmanr(est_i, ref_i).correlation),
                               "erreur_absolue_mediane": float(np.median(np.abs(est_i - ref_i))), "biais": float(np.mean(est_i - ref_i))},
    }
    conf, juste = np.array([c for c, _ in paires]), np.array([j for _, j in paires])
    n_ref_total = sum(l["n_ref"] for l in lignes)
    agg["compromis_confiance"] = [
        {"confiance_min": s, "part_des_proposes": float((conf >= s).mean()), "precision": float(juste[conf >= s].mean()) if (conf >= s).any() else None,
         "evenements_par_nuit": float((conf >= s).sum() / len(lignes)), "rapport_aux_references": float(juste[conf >= s].sum() / n_ref_total)}
        for s in (0.70, 0.80, 0.85, 0.90, 0.95)]
    for cle in ("ahi_a0h3a", "ahi_a0h4"):
        cli = np.array([float(cov[l["personne"]][cle]) if l["personne"] in cov and cov[l["personne"]][cle] else np.nan for l in lignes])
        ok = np.isfinite(cli)
        if ok.sum() > 3:
            agg["index_bout_en_bout"][f"spearman_{cle}"] = float(spearmanr(est_i[ok], cli[ok]).correlation)
    OUT_JSON.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    p, o, ib = agg["precision_par_niveau"], agg["reference_ou"], agg["index_bout_en_bout"]
    L = ["# Résultats — la sortie par nuit complète", "",
         f"*Généré le {agg['date']} par `python_scripts/nuit_complete.py` sur les {agg['n_nuits']} nuits de validation SHHS. "
         "Agrégats seulement : le détail d'une nuit reste hors dépôt. Le test n'est pas touché.*", "",
         "Pour chaque nuit, deux réseaux tournent : celui des stades (EEG) et celui des événements respiratoires (flux, "
         "ceintures, saturation, et le sommeil prédit par le premier). La sortie réunit l'hypnogramme, les événements proposés "
         "pendant le sommeil prédit avec leur confiance, l'index, et une file de relecture commune. La référence est l'ensemble "
         "des événements que le technicien a marqués pendant le sommeil : ceux qui comptent dans l'index.", "",
         "## 1. Ce qu'il reste à relire", "",
         "| Par nuit, médiane (quartiles) | |", "|---|---|",
         f"| Durée de l'enregistrement | {med_iqr([l['signal_total_min'] for l in lignes])} min |",
         f"| Époques dont le stade est à relire | {med_iqr([l['part_stades_a_relire'] for l in lignes])} % |",
         f"| Événements respiratoires de référence | {med_iqr([l['n_ref'] for l in lignes])} |",
         f"| Événements proposés | {med_iqr([l['n_proposes'] for l in lignes])} dont sûrs {med_iqr([l['n_surs'] for l in lignes])}, à relire {med_iqr([l['n_a_relire'] for l in lignes])} |",
         f"| « Événements possibles » signalés (non proposés) | {med_iqr([l['n_possibles'] for l in lignes])} |",
         f"| **Signal à relire, stades et événements réunis** | **{med_iqr([l['signal_a_relire_min'] for l in lignes])} min, soit {med_iqr([l['part_a_relire'] for l in lignes])} % de la nuit** |", "",
         "Le signal à relire est l'union des passages concernés, avec 30 s de contexte autour de chaque événement. C'est une "
         "durée de signal, pas un temps de lecture : seul un chronomètre avec un lecteur dira le temps gagné.", "",
         "## 2. La confiance trie-t-elle les événements ?", "",
         "| Niveau | Événements proposés | Part qui recouvre un événement du technicien |", "|---|---|---|",
         f"| Sûrs (confiance ≥ 0,85) | {agg['effectifs_par_niveau']['sur']:,} | **{p['sur']:.0%}** |",
         f"| À relire (confiance < 0,85) | {agg['effectifs_par_niveau']['a_relire']:,} | {p['a_relire']:.0%} |", "",
         "Plus on exige de confiance, plus les propositions sont justes, et moins il y en a :", "",
         "| Confiance minimale | Part des événements proposés | Précision | Événements par nuit | Événements justes / événements de référence |",
         "|---|---|---|---|---|",
         *[f"| {c['confiance_min']:.2f} | {c['part_des_proposes']:.0%} | {c['precision']:.0%} | {c['evenements_par_nuit']:.0f} | {c['rapport_aux_references']:.0%} |"
           for c in agg["compromis_confiance"]], "",
         "## 3. Où tombent les événements du technicien", "",
         "| | Part des événements de référence |", "|---|---|",
         f"| Dans un événement proposé comme sûr | {o['sur']:.0%} |",
         f"| Dans un événement proposé à relire | {o['a_relire']:.0%} |",
         f"| Dans un « événement possible » signalé | {o['possible']:.0%} |",
         f"| Nulle part : à trouver par le lecteur | **{o['manque']:.0%}** |", "",
         "## 4. L'index, de bout en bout", "",
         "Sommeil prédit par le réseau de stades, événements prédits par le réseau respiratoire, contre l'index calculé "
         "avec le sommeil et les événements du technicien.", "",
         "| Contre | Spearman | Erreur absolue médiane | Biais |", "|---|---|---|---|",
         f"| Index annoté (technicien) | {ib['spearman']:.2f} | {ib['erreur_absolue_mediane']:.1f} / h | {ib['biais']:+.1f} / h |"]
    for cle in ("ahi_a0h3a", "ahi_a0h4"):
        if f"spearman_{cle}" in ib:
            L.append(f"| Index clinique `{cle}` | {ib[f'spearman_{cle}']:.2f} | — | — |")
    L += ["", f"Temps de sommeil : prédit {med_iqr([l['heures_sommeil_pred'] for l in lignes], '{:.1f}')} h, "
              f"technicien {med_iqr([l['heures_sommeil_ref'] for l in lignes], '{:.1f}')} h.", "",
          "## Lecture", "",
          "- Les événements « sûrs » sont ceux qu'un lecteur validerait d'un coup d'œil ; leur précision dit si on peut lui faire cette promesse.",
          "- La ligne « nulle part » est la plus importante : ce sont les événements que l'outil ne signale d'aucune façon, "
          "et que le lecteur ne verra que s'il relit toute la nuit.",
          "- L'index de bout en bout dépend des deux réseaux : une erreur sur le temps de sommeil se retrouve dans l'index.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
