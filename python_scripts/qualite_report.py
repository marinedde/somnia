#!/usr/bin/env python3
"""
Horizon 1.5 — Qualité du signal et refus : ce que les règles trouvent, et ce qu'elles valent.

Les seuils de somnia/qualite.py ont été écrits avant de lancer ce script. Trois questions :

  A. Fréquence : sur les nuits d'entraînement et de validation (aucune étiquette utilisée),
     quelle part des époques chaque règle signale-t-elle, par capteur ? Combien de nuits refusées ?
  B. Utilité (40 nuits de validation, sorties en cache) : le réseau se trompe-t-il plus là où le
     signal est signalé ? Le savait-il déjà (confiance basse) ? Que deviennent la relecture et l'index ?
  C. Capteur perdu (simulé) : que vaut le réseau de stades si les yeux ou le menton sont débranchés
     toute la nuit ? Faut-il alors revenir au modèle EEG seul ?

Le test n'est pas touché. Sortie : docs/RESULTATS_QUALITE.md, models/qualite.json (agrégats).
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import PROCESSED, personnes_du_split  # noqa: E402
from somnia.deep.multi import MULTI_DIR, NOMS_CANAUX, VARIANTES, NuitMulti, ReseauMulti, predire_nuit_multi, preparer_signaux  # noqa: E402
from somnia.deep.resp_net import RESP_DIR, charger_nuits  # noqa: E402
from somnia.nuit import analyser_evenements, analyser_nuit, file_commune  # noqa: E402
from somnia.qualite import CAUSES, REFUS_NUIT_PART, par_seconde, qualite_nuit  # noqa: E402
from somnia.resp import FS_RESP, masque_vers_evenements, sommeil_par_seconde  # noqa: E402

CACHE = PROCESSED.parent / "cache_val"
OUT_MD, OUT_JSON = ROOT / "docs" / "RESULTATS_QUALITE.md", ROOT / "models" / "qualite.json"


def qualite_de_la_nuit(n) -> dict:
    """Qualité complète d'une nuit (tête + respiration) au format attendu par analyser_nuit_complete."""
    with np.load(MULTI_DIR / f"{n.ident}.npz", allow_pickle=False) as d:
        signaux = d["signaux"]
    chemin = RESP_DIR / f"{n.ident}_sao2_invalide.npy"
    return qualite_nuit(signaux, NOMS_CANAUX, n.signaux, n.n_sec, np.load(chemin) if chemin.exists() else None, FS_RESP)


def main() -> int:
    with open(PROCESSED / "covariables.csv", encoding="utf-8") as f:
        cov = {f"shhs-{r['nsrrid']}": r for r in csv.DictReader(f)}
    val = charger_nuits(personnes_du_split("val"), sommeil="multi")
    Q = {n.ident: qualite_de_la_nuit(n) for n in val}

    # ── A. fréquence, entraînement + validation ──
    freq = {c: {"inexploitable": [], **{k: [] for k in CAUSES}} for c in NOMS_CANAUX}
    freq.update({c: {"inexploitable": []} for c in ("flux", "thorax", "abdomen", "sao2", "respiration")})
    refus = {"stades": 0, "index": 0, "n": 0}
    for groupe in ("train", "val"):
        for n in (val if groupe == "val" else charger_nuits(personnes_du_split("train"), sommeil="multi")):
            q = Q.get(n.ident) or qualite_de_la_nuit(n)
            for c in NOMS_CANAUX:
                freq[c]["inexploitable"].append(q["tete"][c]["inexploitable"].mean())
                for k in CAUSES:
                    freq[c][k].append(q["tete"][c][k].mean())
            for c in ("flux", "thorax", "abdomen", "sao2"):
                freq[c]["inexploitable"].append(q["resp"][c]["inexploitable"].mean())
            freq["respiration"]["inexploitable"].append(q["resp"]["inexploitable"].mean())
            refus["n"] += 1; refus["stades"] += q["rapport"]["refus"]["stades"]; refus["index"] += q["rapport"]["refus"]["index"]
    resume_freq = {c: {k: {"moyenne": float(np.mean(v)), "mediane": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(np.max(v)),
                           "nuits_au_dela_du_quart": int((np.array(v) > REFUS_NUIT_PART).sum())} for k, v in d.items()} for c, d in freq.items()}

    # ── B. utilité, validation ──
    S = {k: [] for k in ("y", "pred", "conf", "eeg", "eog", "emg", "mouvement")}
    ev = {"bon": [0, 0], "mauvais": [0, 0]}                     # propositions : [justes, total]
    ref_ev = {"bon": [0, 0], "mauvais": [0, 0]}                 # références : [trouvées, total]
    relire = {"sans": [], "avec": []}
    idx = {"sans": [], "avec": [], "ahi": []}
    for n in val:
        q = Q[n.ident]
        with np.load(CACHE / f"{n.ident}.npz") as d:
            logits, proba = d["logits_multi"], d["proba_v4"].astype(np.float32)
        z = logits - logits.max(axis=1, keepdims=True); p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        ok = n.stades >= 0
        S["y"].append(n.stades[ok]); S["pred"].append(p.argmax(axis=1)[ok]); S["conf"].append(p.max(axis=1)[ok])
        S["eeg"].append(q["tete"]["EEG"]["inexploitable"][ok])
        S["eog"].append((q["tete"]["EOG-G"]["inexploitable"] | q["tete"]["EOG-D"]["inexploitable"])[ok])
        S["emg"].append(q["tete"]["EMG"]["inexploitable"][ok])
        S["mouvement"].append(q["tete"]["EEG"]["artefact"][ok])
        sommeil_pred = par_seconde(np.repeat(p.argmax(axis=1) != 0, 1), n.n_sec)
        sao2 = n.signaux[3, ::FS_RESP][: n.n_sec]
        mauvais = q["resp_inexploitable"]
        sommeil_ref = sommeil_par_seconde(n.stades)[: n.n_sec]
        ref = [e for e in masque_vers_evenements(n.y) if sommeil_ref[min(e.debut, len(sommeil_ref) - 1)]]
        for mode in ("sans", "avec"):
            st = analyser_nuit(None, lambda _: logits, inexploitable=q["eeg_inexploitable"] if mode == "avec" else None)
            r = analyser_evenements(proba, sommeil_pred, sommeil_seulement=True, sao2_1hz=sao2, inexploitable_sec=mauvais if mode == "avec" else None)
            fc = file_commune(st["relecture"], r, n.n_sec, resp_inexploitable=mauvais if mode == "avec" else None, sommeil_sec=sommeil_pred)
            relire[mode].append(fc["part_a_relire_pct"]); idx[mode].append(r["resume"]["index_clinique_3"] or 0.0)
            if mode == "sans":
                couv = np.zeros(n.n_sec, dtype=bool)
                for e in r["evenements"]:
                    zone = "mauvais" if mauvais[e["debut_s"]] else "bon"
                    ev[zone][0] += bool((n.y[e["debut_s"]:e["debut_s"] + e["duree_s"]] > 0).any()); ev[zone][1] += 1
                    couv[e["debut_s"]:e["debut_s"] + e["duree_s"]] = True
                for e in ref:
                    zone = "mauvais" if mauvais[e.debut] else "bon"
                    ref_ev[zone][0] += bool(couv[e.debut:e.fin].any()); ref_ev[zone][1] += 1
        v = cov.get(n.personne, {}).get("ahi_a0h3", "")
        idx["ahi"].append(float(v) if v not in ("", None) else np.nan)
    S = {k: np.concatenate(v) for k, v in S.items()}
    juste = S["y"] == S["pred"]

    def bloc(masque):
        return {"n": int(masque.sum()), "part": float(masque.mean()), "exactitude": float(juste[masque].mean()) if masque.any() else None,
                "confiance_mediane": float(np.median(S["conf"][masque])) if masque.any() else None,
                "sous_le_seuil": float((S["conf"][masque] < 0.6).mean()) if masque.any() else None}
    stades = {"EEG exploitable": bloc(~S["eeg"]), "EEG inexploitable": bloc(S["eeg"]),
              "Yeux inexploitables (EEG exploitable)": bloc(S["eog"] & ~S["eeg"]), "Menton inexploitable (EEG exploitable)": bloc(S["emg"] & ~S["eeg"]),
              "Tout exploitable": bloc(~S["eeg"] & ~S["eog"] & ~S["emg"]),
              "EEG : mouvement signalé (information, pas un refus)": bloc(S["mouvement"])}
    stades["EEG : mouvement signalé (information, pas un refus)"]["part_eveil"] = float((S["y"][S["mouvement"]] == 0).mean()) if S["mouvement"].any() else None
    a = np.array(idx["ahi"]); okc = np.isfinite(a)
    index = {m: {"spearman": float(spearmanr(np.array(idx[m])[okc], a[okc]).correlation),
                 "erreur_absolue_mediane": float(np.median(np.abs(np.array(idx[m])[okc] - a[okc])))} for m in ("sans", "avec")}

    # ── C. capteur perdu toute la nuit (simulé) ──
    def reseau(variante):
        m = ReseauMulti(VARIANTES[variante]); m.load_state_dict(torch.load(ROOT / f"models/multi/{variante}_s42.pt", map_location="cpu")); return m.eval()
    multi, seul = reseau("eeg_eog_emg"), reseau("eeg")
    pannes = {"Rien de débranché": (), "Menton débranché": (4,), "Yeux débranchés": (2, 3), "Yeux et menton débranchés": (2, 3, 4)}
    Yc, Pc = [], {k: [] for k in [*pannes, "Modèle EEG seul"]}
    for n in val:
        with np.load(MULTI_DIR / f"{n.ident}.npz", allow_pickle=False) as d:
            signaux = d["signaux"]
        ok = n.stades >= 0; Yc.append(n.stades[ok])
        for nom, coupes in pannes.items():
            s2 = signaux.copy(); s2[:, list(coupes)] = 0
            X = preparer_signaux(s2, multi.canaux)
            Pc[nom].append(predire_nuit_multi(multi, NuitMulti("", "", X, np.zeros(len(X), dtype=np.int64)), torch.device("cpu")).argmax(axis=1)[ok])
        X = preparer_signaux(signaux, seul.canaux)
        Pc["Modèle EEG seul"].append(predire_nuit_multi(seul, NuitMulti("", "", X, np.zeros(len(X), dtype=np.int64)), torch.device("cpu")).argmax(axis=1)[ok])
    yc = np.concatenate(Yc)
    capteur_perdu = {k: {"kappa": float(cohen_kappa_score(yc, np.concatenate(v))),
                         "accord_eveil_sommeil": float(((yc == 0) == (np.concatenate(v) == 0)).mean())} for k, v in Pc.items()}

    prec = lambda c: c[0] / c[1] if c[1] else None
    agg = {"date": date.today().isoformat(), "frequence": resume_freq, "refus": refus, "stades": stades,
           "evenements": {"precision": {k: prec(v) for k, v in ev.items()}, "n_proposes": {k: v[1] for k, v in ev.items()},
                          "rappel": {k: prec(v) for k, v in ref_ev.items()}, "n_references": {k: v[1] for k, v in ref_ev.items()}},
           "signal_a_relire_pct": {m: float(np.median(v)) for m, v in relire.items()}, "index_clinique_3": index, "capteur_perdu": capteur_perdu}
    OUT_JSON.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    pc = lambda x: "—" if x is None else f"{x:.1%}"
    premiere_version = []
    v1p = ROOT / "models" / "qualite_v1.json"
    if v1p.exists():
        v1 = json.loads(v1p.read_text(encoding="utf-8"))
        b1 = v1["stades"]["EEG inexploitable"]
        premiere_version = [
            "**Ce que la première version a donné, et pourquoi elle a été révisée.** Écrite avant de regarder, elle comptait aussi comme "
            "inexploitables les mouvements (tête) et tout capteur respiratoire écrêté. Mesurée sur la validation :", "",
            f"- les {b1['n']:,} époques d'EEG signalées étaient presque toutes de l'éveil avec mouvement, où le réseau est juste à {b1['exactitude']:.0%} "
            f"(contre {v1['stades']['EEG exploitable']['exactitude']:.0%} ailleurs) : la règle envoyait en relecture des époques faciles ;",
            f"- {v1['refus']['index']} nuits sur {v1['refus']['n']} étaient refusées pour l'index, alors que les propositions y étaient aussi justes "
            f"qu'ailleurs ({v1['evenements']['precision']['mauvais']:.0%} contre {v1['evenements']['precision']['bon']:.0%}) ;",
            "- détail par cause : flux écrêté, propositions justes à 75 % (pas un problème) ; deux ceintures écrêtées, 51 % ; saturation invalide, 39 %.", "",
            "La définition a donc été resserrée sur ce qui dégrade vraiment. Les seuils n'ont pas été touchés. **Ce choix a été fait sur les 40 "
            "personnes de validation** : c'est une hypothèse à confirmer sur d'autres nuits, pas un résultat."]
    L = ["# Résultats — qualité du signal et refus (horizon 1.5)", "",
         f"*Généré le {agg['date']} par `python_scripts/qualite_report.py`. Règles et seuils : `somnia/qualite.py`, écrits avant de lancer ce script. "
         "Agrégats seulement. Le test n'est pas touché.*", "",
         "## Les règles", "",
         "Par époque de 30 s et par capteur, relativement à la nuit elle-même : **plat** (écart-type sous 5 % de la médiane de la nuit), "
         "**écrêté** (plus de 5 % des points à l'extrême de la nuit), **mouvement** (écart-type au-delà de 6 fois la médiane). "
         "Saturation : valeurs hors de 50–100 %.", "",
         "Un passage est **inexploitable** quand : un capteur de la tête est plat ou écrêté ; le flux est plat ; les deux ceintures sont "
         f"plates ou écrêtées ; la saturation est invalide. Au-delà de {REFUS_NUIT_PART:.0%} de la nuit, l'outil refuse de rendre les indices de "
         "sommeil (EEG) ou l'index (respiration) ; si ce sont les yeux ou le menton, il passe au modèle EEG seul.", "",
         *premiere_version, "",
         f"## A. Ce que les règles signalent ({refus['n']} nuits d'entraînement et de validation)", "",
         "| Capteur | Époques inexploitables, moyenne | Médiane | 9e décile | Maximum | Nuits au-delà du quart | dont plat / écrêté / artefact (moyennes) |",
         "|---|---|---|---|---|---|---|"]
    for c, d in resume_freq.items():
        i = d["inexploitable"]
        causes = " / ".join(pc(d[k]["moyenne"]) for k in CAUSES) if "plat" in d else "—"
        L.append(f"| {c} | {pc(i['moyenne'])} | {pc(i['mediane'])} | {pc(i['p90'])} | {pc(i['max'])} | {i['nuits_au_dela_du_quart']} | {causes} |")
    L += ["", f"Nuits refusées : {refus['stades']} pour les stades, {refus['index']} pour l'index, sur {refus['n']}. "
              "Rappel : la cohorte a déjà écarté, à la préparation, les nuits sans canal ou avec plus de 30 % de saturation invalide.", "",
          "## B. Les règles trouvent-elles les erreurs ? (40 nuits de validation)", "",
          "**Stades** (EEG + yeux + menton)", "",
          "| Époques | Nombre | Part | Exactitude | Confiance médiane du réseau | Déjà sous le seuil de relecture |", "|---|---|---|---|---|---|"]
    for nom, b in stades.items():
        L.append(f"| {nom} | {b['n']:,} | {pc(b['part'])} | {pc(b['exactitude'])} | {'—' if b['confiance_mediane'] is None else format(b['confiance_mediane'], '.2f')} | {pc(b['sous_le_seuil'])} |")
    e = agg["evenements"]
    L += ["", "**Événements respiratoires** (v4)", "",
          "| Zone | Événements proposés | Part qui recouvre un événement du technicien | Événements du technicien | Part retrouvée |", "|---|---|---|---|---|",
          f"| Respiration exploitable | {e['n_proposes']['bon']:,} | {pc(e['precision']['bon'])} | {e['n_references']['bon']:,} | {pc(e['rappel']['bon'])} |",
          f"| Respiration inexploitable | {e['n_proposes']['mauvais']:,} | {pc(e['precision']['mauvais'])} | {e['n_references']['mauvais']:,} | {pc(e['rappel']['mauvais'])} |", "",
          "**Effet sur la sortie par nuit**", "",
          "| | Sans la qualité | Avec la qualité |", "|---|---|---|",
          f"| Signal à relire, médiane | {agg['signal_a_relire_pct']['sans']:.0f} % | {agg['signal_a_relire_pct']['avec']:.0f} % |",
          f"| Index clinique contre `ahi_a0h3`, Spearman | {index['sans']['spearman']:.3f} | {index['avec']['spearman']:.3f} |",
          f"| Index clinique contre `ahi_a0h3`, erreur absolue médiane | {index['sans']['erreur_absolue_mediane']:.1f} / h | {index['avec']['erreur_absolue_mediane']:.1f} / h |", "",
          "## C. Un capteur débranché toute la nuit (simulé)", "",
          "Le canal est remplacé par des zéros sur toute la nuit, et le réseau multi-capteurs est appliqué tel quel.", "",
          "| Situation | Kappa | Accord éveil / sommeil |", "|---|---|---|"]
    for nom, c in capteur_perdu.items():
        L.append(f"| {nom} | {c['kappa']:.3f} | {c['accord_eveil_sommeil']:.1%} |")
    cp, em = capteur_perdu, stades["EEG : mouvement signalé (information, pas un refus)"]
    L += ["", "## Lecture", "",
          "- **Dans cette cohorte, un capteur vraiment perdu est rare** : elle a déjà été triée à la préparation, et les enregistrements "
          "SHHS ont été contrôlés à l'époque. Les règles serviront surtout sur des nuits venues d'ailleurs.",
          f"- **Un mouvement n'est pas une panne** : {pc(em.get('part_eveil'))} des époques d'EEG « en mouvement » sont de l'éveil, et le réseau y est juste "
          f"à {pc(em['exactitude'])}. C'est une information utile au lecteur, pas un motif de refus.",
          f"- **Là où la respiration est inexploitable, les propositions sont moins justes** ({pc(e['precision']['mauvais'])} contre {pc(e['precision']['bon'])}) : "
          "ne rien y proposer et le dire est le bon comportement. Le coût est du signal à relire en plus.",
          f"- **Un capteur débranché fait plus de mal que son absence** : sans les yeux et le menton, le réseau multi-capteurs tombe à un kappa de "
          f"{cp['Yeux et menton débranchés']['kappa']:.2f}, alors que le modèle EEG seul fait {cp['Modèle EEG seul']['kappa']:.2f}. D'où la règle de repli : "
          "si les yeux ou le menton sont inexploitables sur plus d'un quart de la nuit, les stades sont calculés avec le modèle EEG seul.",
          "- **Ce que le module ne voit pas** : un capteur mal posé mais qui bouge (EEG noyé dans l'ECG ou le secteur), une saturation plausible "
          "mais fausse, une inversion de canaux. Il faudrait des règles spectrales, ou un détecteur appris.",
          "- **Limites** : le refus de l'index compte le temps inexploitable sur toute la nuit, éveil compris, alors que seul le sommeil "
          "compte ; et la comparaison des zones repose sur peu d'événements (une centaine en zone inexploitable).",
          "- **Pas encore fait** : un refus par passage appris sur la confiance du réseau lui-même, et un essai sur des nuits réellement dégradées.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
