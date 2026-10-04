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
from somnia.qualite import qualite_nuit  # noqa: E402
from somnia.resp import POSITION_DOS_SHHS, chute_de_saturation, FS_RESP, index_par_heure, masque_vers_evenements, sommeil_par_seconde  # noqa: E402
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
    # Stades : EEG + yeux + menton, lecture de la nuit (horizon 1.3). Événements : v4, qui reçoit le sommeil prédit par ce modèle.
    from somnia.deep.multi import MULTI_DIR, VARIANTES, NuitMulti, ReseauMulti, predire_nuit_multi, preparer_signaux
    stades_net = ReseauMulti(VARIANTES["eeg_eog_emg"]); stades_net.load_state_dict(torch.load(ROOT / "models/multi/eeg_eog_emg_s42.pt", map_location="cpu")); stades_net.eval()
    T = json.loads((ROOT / "models/multi/eeg_eog_emg_s42.json").read_text(encoding="utf-8"))["calibration"]["temperature"]
    ev_net = ReseauEvenements(canaux=5); ev_net.load_state_dict(torch.load(ROOT / "models/resp/evenements_v4_s42.pt", map_location="cpu")); ev_net.to(dev)

    # Repli (horizon 1.5) : si les yeux ou le menton sont inexploitables, le modèle EEG seul, avec sa propre température.
    repli_net = ReseauMulti(VARIANTES["eeg"]); repli_net.load_state_dict(torch.load(ROOT / "models/multi/eeg_s42.pt", map_location="cpu")); repli_net.eval()
    T_repli = json.loads((ROOT / "models/multi/eeg_s42.json").read_text(encoding="utf-8"))["calibration"]["temperature"]
    from somnia.deep.multi import NOMS_CANAUX
    from somnia.deep.resp_net import RESP_DIR

    def predire_avec(net):
        def predire_stades(signaux):                   # (n_epoques, 5, 3000) : les cinq capteurs préparés
            X = preparer_signaux(signaux, net.canaux)
            return predire_nuit_multi(net, NuitMulti("nuit", "personne", X, np.zeros(len(X), dtype=np.int64)), torch.device("cpu"))
        return predire_stades
    positions = []
    qual = {"refus_stades": 0, "refus_index": 0, "repli_eeg": 0, "ecartes": 0, "resp_inexploitable_pct": []}

    cov = {}
    if (PROCESSED / "covariables.csv").exists():
        with open(PROCESSED / "covariables.csv", encoding="utf-8") as f:
            cov = {f"shhs-{r['nsrrid']}": r for r in csv.DictReader(f)}

    nuits = charger_nuits(personnes_du_split("val"), sommeil="multi")
    lignes, niveaux = [], {"sur": [0, 0], "a_relire": [0, 0]}          # [justes, total]
    paires = []                                                        # (confiance, juste) de chaque événement proposé
    ou_sont_les_ref = {"sur": 0, "a_relire": 0, "possible": 0, "manque": 0}
    for k, n in enumerate(nuits):
        with np.load(MULTI_DIR / f"{n.ident}.npz", allow_pickle=False) as d:
            eeg = d["signaux"]                                                      # µV, (n_epoques, 5 capteurs, 3000)
        proba = proba_nuit(ev_net, n, dev, canal_sommeil=True)
        pos = np.load(RESP_DIR / f"{n.ident}_position.npy")[: n.n_sec]
        dos = np.concatenate([pos == POSITION_DOS_SHHS, np.zeros(n.n_sec - len(pos), dtype=bool)])
        chemin_q = RESP_DIR / f"{n.ident}_sao2_invalide.npy"
        q = qualite_nuit(eeg, NOMS_CANAUX, n.signaux, n.n_sec, np.load(chemin_q) if chemin_q.exists() else None, FS_RESP)
        repli = q["rapport"]["modele_stades"] == "eeg"
        r = analyser_nuit_complete(eeg, predire_avec(repli_net if repli else stades_net), proba, temperature=T_repli if repli else T,
                                   sao2_1hz=n.signaux[3, ::FS_RESP][: n.n_sec], qualite=q, dorsal_sec=dos)
        qual["refus_stades"] += q["rapport"]["refus"]["stades"]; qual["refus_index"] += q["rapport"]["refus"]["index"]; qual["repli_eeg"] += repli
        qual["ecartes"] += r["respiration"]["resume"]["n_ecartes_signal_inexploitable"]
        qual["resp_inexploitable_pct"].append(q["rapport"]["respiration_inexploitable_pct"])
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
        # 5. position : index clinique sur le dos et hors du dos, estimé contre technicien (même règle de désaturation)
        sao2_ref = n.signaux[3, ::FS_RESP][: n.n_sec]
        compte_ref = [e for e in ref if e.classe == 1 or chute_de_saturation(sao2_ref, e.debut, e.fin) >= 3]
        pos_ref = {}
        for nom, zone in (("dorsal", dos), ("non_dorsal", ~dos)):
            h = float((sommeil_ref & zone).sum()) / 3600
            pos_ref[nom] = sum(1 for e in compte_ref if dos[e.debut] == (nom == "dorsal")) / h if h >= 0.5 else None
        positions.append({"ref": pos_ref, "est": {"dorsal": res.get("index_dorsal"), "non_dorsal": res.get("index_non_dorsal")},
                          "dorsal_pct": res.get("sommeil_dorsal_pct"), "positionnel_est": res.get("positionnel"),
                          "positionnel_ref": (pos_ref["dorsal"] >= 2 * pos_ref["non_dorsal"] and pos_ref["dorsal"] >= 5)
                          if pos_ref["dorsal"] is not None and pos_ref["non_dorsal"] is not None else None})
        # 4. index de bout en bout vs référence (sommeil et événements du technicien)
        lignes.append({
            "personne": n.personne, "index_ref": index_par_heure(ref_tous, sommeil_ref), "index_bout_en_bout": res["index_par_heure"] if res["index_par_heure"] is not None else np.nan,
            "index_clinique_3": res["index_clinique_3"] if res["index_clinique_3"] is not None else np.nan,
            "index_clinique_4": res["index_clinique_4"] if res["index_clinique_4"] is not None else np.nan, "index_refuse": res["index_par_heure"] is None,
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
    rendus = [l for l in lignes if not l["index_refuse"]]                       # l'index n'est mesuré que là où l'outil le rend
    ref_i = np.array([l["index_ref"] for l in rendus]); est_i = np.array([l["index_bout_en_bout"] for l in rendus])
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
    agg["qualite"] = {**{k: v for k, v in qual.items() if k != "resp_inexploitable_pct"},
                      "resp_inexploitable_pct_mediane": float(np.median(qual["resp_inexploitable_pct"]))}
    def accord_position(nom):
        v = [(x["est"][nom], x["ref"][nom]) for x in positions if x["est"][nom] is not None and x["ref"][nom] is not None]
        e, r = np.array([a for a, _ in v]), np.array([b for _, b in v])
        return {"n": len(v), "spearman": float(spearmanr(e, r).correlation), "erreur_absolue_mediane": float(np.median(np.abs(e - r)))}
    deux = [x for x in positions if x["positionnel_est"] is not None and x["positionnel_ref"] is not None]
    agg["position"] = {"sommeil_dorsal_pct_mediane": float(np.median([x["dorsal_pct"] for x in positions])),
                       "dorsal": accord_position("dorsal"), "non_dorsal": accord_position("non_dorsal"),
                       "positionnel": {"n": len(deux), "technicien": int(sum(x["positionnel_ref"] for x in deux)), "estime": int(sum(x["positionnel_est"] for x in deux)),
                                       "accord": float(np.mean([x["positionnel_est"] == x["positionnel_ref"] for x in deux])) if deux else None}}
    conf, juste = np.array([c for c, _ in paires]), np.array([j for _, j in paires])
    n_ref_total = sum(l["n_ref"] for l in lignes)
    agg["compromis_confiance"] = [
        {"confiance_min": s, "part_des_proposes": float((conf >= s).mean()), "precision": float(juste[conf >= s].mean()) if (conf >= s).any() else None,
         "evenements_par_nuit": float((conf >= s).sum() / len(lignes)), "rapport_aux_references": float(juste[conf >= s].sum() / n_ref_total)}
        for s in (0.70, 0.80, 0.85, 0.90, 0.95)]
    for cle, col in (("ahi_a0h3", "index_clinique_3"), ("ahi_a0h4", "index_clinique_4")):
        cli = np.array([float(cov[l["personne"]][cle]) if l["personne"] in cov and cov[l["personne"]].get(cle) else np.nan for l in lignes])
        e = np.array([l[col] for l in lignes]); ok = np.isfinite(cli) & np.isfinite(e)
        if ok.sum() > 3:
            agg[col] = {"contre": cle, "spearman": float(spearmanr(e[ok], cli[ok]).correlation),
                        "erreur_absolue_mediane": float(np.median(np.abs(e[ok] - cli[ok]))), "biais": float(np.mean(e[ok] - cli[ok]))}
    for cle in ("ahi_a0h3a", "ahi_a0h4"):
        cli = np.array([float(cov[l["personne"]][cle]) if l["personne"] in cov and cov[l["personne"]][cle] else np.nan for l in rendus])
        ok = np.isfinite(cli)
        if ok.sum() > 3:
            agg["index_bout_en_bout"][f"spearman_{cle}"] = float(spearmanr(est_i[ok], cli[ok]).correlation)
    OUT_JSON.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    p, o, ib = agg["precision_par_niveau"], agg["reference_ou"], agg["index_bout_en_bout"]
    L = ["# Résultats — la sortie par nuit complète", "",
         f"*Généré le {agg['date']} par `python_scripts/nuit_complete.py` sur les {agg['n_nuits']} nuits de validation SHHS. "
         "Agrégats seulement : le détail d'une nuit reste hors dépôt. Le test n'est pas touché.*", "",
         "Pour chaque nuit, deux réseaux tournent : celui des stades (EEG, yeux, menton) et celui des événements respiratoires (flux, "
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
            L.append(f"| Tous les événements proposés, contre l'index clinique `{cle}` | {ib[f'spearman_{cle}']:.2f} | — | — |")
    for col, nom in (("index_clinique_3", "3"), ("index_clinique_4", "4")):
        if col in agg:
            c = agg[col]
            L.append(f"| **Index clinique estimé** (apnées + hypopnées avec désaturation ≥ {nom} points), contre `{c['contre']}` | "
                     f"**{c['spearman']:.2f}** | {c['erreur_absolue_mediane']:.1f} / h | {c['biais']:+.1f} / h |")
    ql = agg["qualite"]
    L += ["", "## 5. Qualité du signal et refus", "",
          "| Sur les nuits de validation | |", "|---|---|",
          f"| Respiration inexploitable, médiane par nuit | {ql['resp_inexploitable_pct_mediane']:.1f} % |",
          f"| Événements proposés puis écartés parce que le signal y était inexploitable | {ql['ecartes']:,} |",
          f"| Nuits où l'outil refuse de rendre un index | {ql['refus_index']} sur {agg['n_nuits']} |",
          f"| Nuits où l'outil refuse de rendre les indices de sommeil | {ql['refus_stades']} sur {agg['n_nuits']} |",
          f"| Nuits où les stades sont calculés avec l'EEG seul (yeux ou menton inexploitables) | {ql['repli_eeg']} sur {agg['n_nuits']} |", "",
          "Règles et mesures : `docs/RESULTATS_QUALITE.md`. L'index ci-dessus n'est mesuré que sur les nuits où l'outil le rend."]
    po = agg["position"]
    L += ["", "## 6. Position du corps", "",
          "L'index clinique (apnées + hypopnées avec désaturation ≥ 3 points) est rendu séparément sur le dos et hors du dos, quand il y a au moins "
          "30 minutes de sommeil dans la position. Référence : mêmes calculs sur les événements et le sommeil du technicien.", "",
          "| | |", "|---|---|",
          f"| Part du sommeil sur le dos, médiane | {po['sommeil_dorsal_pct_mediane']:.0f} % |",
          f"| Index sur le dos, estimé contre technicien ({po['dorsal']['n']} personnes) | Spearman {po['dorsal']['spearman']:.2f}, erreur médiane {po['dorsal']['erreur_absolue_mediane']:.1f} / h |",
          f"| Index hors du dos ({po['non_dorsal']['n']} personnes) | Spearman {po['non_dorsal']['spearman']:.2f}, erreur médiane {po['non_dorsal']['erreur_absolue_mediane']:.1f} / h |",
          f"| Apnée positionnelle (index sur le dos au moins double, et ≥ 5) : technicien / estimé / accord | {po['positionnel']['technicien']} / {po['positionnel']['estime']} / {po['positionnel']['accord']:.0%} sur {po['positionnel']['n']} personnes |"]
    L += ["", f"Temps de sommeil : prédit et analysable {med_iqr([l['heures_sommeil_pred'] for l in lignes], '{:.1f}')} h, "
              f"technicien {med_iqr([l['heures_sommeil_ref'] for l in lignes], '{:.1f}')} h.", "",
          "## Lecture", "",
          "- Les événements « sûrs » sont ceux qu'un lecteur validerait d'un coup d'œil ; leur précision dit si on peut lui faire cette promesse.",
          "- La ligne « nulle part » est la plus importante : ce sont les événements que l'outil ne signale d'aucune façon, "
          "et que le lecteur ne verra que s'il relit toute la nuit.",
          "- L'index à donner au médecin est l'index clinique estimé : il applique la définition des index SHHS (hypopnées "
          "comptées seulement avec désaturation). Détail et comparaison au simple compte des désaturations : `docs/RESULTATS_CLINIQUE.md`.",
          "- L'index de bout en bout dépend des deux réseaux : une erreur sur le temps de sommeil se retrouve dans l'index.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
