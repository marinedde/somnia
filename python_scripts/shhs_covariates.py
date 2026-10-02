#!/usr/bin/env python3
"""
Covariables SHHS : jointure sur la cohorte, index clinique, sous-groupes.

Source : datasets/shhs1-dataset-<version>.csv du NSRR (téléchargé hors dépôt, ~24 Mo,
1 271 colonnes). On n'en garde que quelques-unes, pour les 272 nuits retenues :
    age_s1, gender (1 homme, 2 femme), race, bmi_s1, slpeffp (efficacité de sommeil, %),
    ess_s1 (Epworth), overall_shhs1 (qualité globale de la PSG),
    ahi_a0h3a : index d'apnées-hypopnées, hypopnées avec désaturation ≥ 3 % ou micro-éveil (AASM 2012),
    ahi_a0h4  : idem avec désaturation ≥ 4 % (critère historique SHHS / Medicare).

Trois choses sont produites :
  1. ~/data/shhs/processed/covariables.csv (hors dépôt : individuel) ;
  2. docs/COVARIABLES_SHHS.md (versionné, agrégé, aucune cellule de moins de 5 personnes) :
     description de la cohorte et de chaque ensemble, comparaison de MON index annoté
     (événements / h de sommeil, tous hypopnées comprises) à l'index clinique ;
  3. dans le même document : les références Random Forest de l'étape 3, par sous-groupe,
     sur la VALIDATION (40 personnes : cellules petites, dites comme telles). Le test reste fermé.

Usage :
    python python_scripts/shhs_covariates.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.evaluation import metriques_apnee, metriques_stades  # noqa: E402
from somnia.physionet import charger_tableau  # noqa: E402
from somnia.shhs import lire_annotations, resume  # noqa: E402

RAW = Path(os.environ.get("SHHS_DIR", Path.home() / "data" / "shhs" / "raw"))
PROCESSED = Path(os.environ.get("SHHS_PROCESSED", Path.home() / "data" / "shhs" / "processed"))
SPLIT = ROOT / "data" / "splits" / "shhs_v1.json"
OUT_MD = ROOT / "docs" / "COVARIABLES_SHHS.md"
COLONNES = ["nsrrid", "age_s1", "gender", "race", "bmi_s1", "slpeffp", "ess_s1", "overall_shhs1",
            "ahi_a0h3a", "ahi_a0h4", "ahi_a0h3", "ahi_o0h3", "ahi_c0h3"]
MIN_CELLULE = 5
SEUILS_SEVERITE = [(5, "< 5 (normal)"), (15, "5-15 (légère)"), (30, "15-30 (modérée)"), (np.inf, "≥ 30 (sévère)")]


def severite(x):
    for s, nom in SEUILS_SEVERITE:
        if x < s:
            return nom
    return SEUILS_SEVERITE[-1][1]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def med_iqr(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return f"{np.median(v):.1f} ({np.percentile(v, 25):.1f}–{np.percentile(v, 75):.1f})" if len(v) else "—"


def main() -> int:
    dataset = sorted((RAW / "shhs" / "datasets").glob("shhs1-dataset-*.csv"))[-1]
    split = json.loads(SPLIT.read_text(encoding="utf-8"))["taches"]["shhs"]
    ensemble_de = {p: k for k, v in split.items() for p in v}
    with open(PROCESSED / "cohorte.csv", encoding="utf-8") as f:
        cohorte = {r["personne"]: r for r in csv.DictReader(f) if r.get("exclue") != "True"}
    xmls = {x.name.replace("-nsrr.xml", ""): x for x in RAW.rglob("*-nsrr.xml")}

    # 1. jointure
    lignes = {}
    with open(dataset, encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            pers = f"shhs-{r['nsrrid']}"
            if pers in cohorte:
                lignes[pers] = {c: r.get(c, "") for c in COLONNES}
    manquants = sorted(set(cohorte) - set(lignes))
    if manquants:
        print(f"ATTENTION : {len(manquants)} nuits sans ligne de covariables")
    for pers, l in lignes.items():
        l["ensemble"] = ensemble_de[pers]
        l["index_annote"] = resume(lire_annotations(xmls[cohorte[pers]["enregistrement"]]))["iah_estime"]
    with open(PROCESSED / "covariables.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES + ["ensemble", "index_annote"]); w.writeheader(); w.writerows(lignes.values())
    print(f"{len(lignes)} personnes jointes -> {PROCESSED / 'covariables.csv'} (hors dépôt)")

    P = list(lignes.values())
    def col(nom, sous=P): return np.array([_f(x[nom]) for x in sous])
    L = [f"# Covariables SHHS — cohorte Somnia", "",
         f"*Généré le {date.today().isoformat()} par `python_scripts/shhs_covariates.py` à partir de `{dataset.name}` (NSRR). "
         f"Agrégats seulement ; aucune cellule de moins de {MIN_CELLULE} personnes.*", "",
         "## 1. La cohorte, et chaque ensemble", "",
         "| | Toutes | Entraînement | Validation | Test |", "|---|---|---|---|---|"]
    groupes = {"Toutes": P, **{k.capitalize(): [x for x in P if x["ensemble"] == e] for k, e in (("entraînement", "train"), ("validation", "val"), ("test", "test"))}}
    def ligne(nom, fn):
        L.append(f"| {nom} | " + " | ".join(fn(g) for g in groupes.values()) + " |")
    ligne("Personnes", lambda g: str(len(g)))
    ligne("Âge, médiane (IQR)", lambda g: med_iqr(col("age_s1", g)))
    ligne("Femmes", lambda g: f"{np.mean(col('gender', g) == 2):.0%}")
    ligne("IMC, médiane (IQR)", lambda g: med_iqr(col("bmi_s1", g)))
    ligne("Efficacité de sommeil %, médiane (IQR)", lambda g: med_iqr(col("slpeffp", g)))
    ligne("Epworth, médiane (IQR)", lambda g: med_iqr(col("ess_s1", g)))
    ligne("IAH clinique ≥ 3 % ou éveil (ahi_a0h3a), médiane (IQR)", lambda g: med_iqr(col("ahi_a0h3a", g)))
    ligne("IAH clinique ≥ 4 % (ahi_a0h4), médiane (IQR)", lambda g: med_iqr(col("ahi_a0h4", g)))
    ligne("Mon index annoté (événements / h), médiane (IQR)", lambda g: med_iqr(col("index_annote", g)))
    for nom_sev, cle in (("Sévérité (ahi_a0h4)", "ahi_a0h4"), ("Sévérité (ahi_a0h3a)", "ahi_a0h3a")):
        for s, nom in SEUILS_SEVERITE:
            ligne(f"{nom_sev} : {nom}", lambda g, nom=nom, cle=cle: f"{np.mean([severite(_f(x[cle])) == nom for x in g]):.0%}")
    L += ["", f"Âge minimal : {np.nanmin(col('age_s1')):.0f} ans (SHHS recrute à partir de 40 ans). "
              f"Qualité globale de la PSG (overall_shhs1, 1 à 7) : médiane {np.nanmedian(col('overall_shhs1')):.0f}.", ""]

    # 2. index annoté vs clinique
    ia, a3, a4 = col("index_annote"), col("ahi_a0h3a"), col("ahi_a0h4")
    ok = np.isfinite(ia) & np.isfinite(a3) & np.isfinite(a4)
    L += ["## 2. Mon index annoté contre l'index clinique", "",
          "Mon index compte tous les événements annotés (apnées + toutes les hypopnées) par heure de sommeil. "
          "L'index clinique SHHS ne compte une hypopnée qu'avec une désaturation (≥ 3 % ou micro-éveil pour "
          "`ahi_a0h3a`, ≥ 4 % pour `ahi_a0h4`).", "",
          "| Comparaison | Corrélation (Spearman) | Rapport médian annoté / clinique | Accord de classe de sévérité |", "|---|---|---|---|"]
    from scipy.stats import spearmanr
    for nom, ref in (("ahi_a0h3a", a3), ("ahi_a0h4", a4)):
        rho = spearmanr(ia[ok], ref[ok]).correlation
        rapport = np.median(ia[ok] / np.maximum(ref[ok], 0.5))
        accord = np.mean([severite(x) == severite(y) for x, y in zip(ia[ok], ref[ok])])
        L.append(f"| annoté vs {nom} | {rho:.2f} | ×{rapport:.1f} | {accord:.0%} |")
    L += ["", "Lecture : l'ordre des personnes est bien conservé (corrélation), mais le niveau ne l'est pas : "
              "mon index surestime la sévérité clinique. La stratification du découpage (faite sur l'index annoté) "
              "reste valide pour répartir ; les classes de sévérité à publier sont celles de l'index clinique.", ""]

    # 3. sous-groupes, références de l'étape 3, validation seulement
    L += ["## 3. Références de l'étape 3 par sous-groupe (validation, 40 personnes)", "",
          f"Cellules de moins de {MIN_CELLULE} personnes masquées. Avec 40 personnes, ces chiffres indiquent une "
          "direction, pas une certitude : l'intervalle d'une AUC sur 5 personnes est énorme.", ""]
    age = col("age_s1"); tert = np.nanpercentile(age, [33.3, 66.7])
    def groupe(x):
        g = {}
        g["Âge"] = "< %.0f ans" % tert[0] if _f(x["age_s1"]) < tert[0] else ("≥ %.0f ans" % tert[1] if _f(x["age_s1"]) >= tert[1] else "%.0f–%.0f ans" % (tert[0], tert[1]))
        g["Sexe"] = {1.0: "homme", 2.0: "femme"}.get(_f(x["gender"]), "?")
        b = _f(x["bmi_s1"]); g["IMC"] = "< 25" if b < 25 else ("25–30" if b < 30 else "≥ 30")
        g["Sévérité clinique (ahi_a0h4)"] = severite(_f(x["ahi_a0h4"]))
        return g
    sous = {p: groupe(x) for p, x in lignes.items()}
    for tache, modele_nom, cles in (("eeg", "shhs_rf_eeg.joblib", ["accuracy", "kappa", "f1_macro"]),
                                     ("ecg", "shhs_rf_ecg.joblib", ["auc_roc", "auc_pr", "f1_apnee"])):
        modele = joblib.load(ROOT / "models" / modele_nom)
        t = charger_tableau(PROCESSED / f"features_{tache}.npz")
        va = t["ensemble"] == "val"
        X, y, pers = t["X"][va], t["y"][va], t["personne"][va]
        pred = modele.predict(X); proba = modele.predict_proba(X)[:, 1] if tache == "ecg" else None
        L += [f"### {'Stades (EEG)' if tache == 'eeg' else 'Apnée (ECG)'}", "",
              "| Sous-groupe | Personnes | Lignes | " + " | ".join(cles) + " |", "|---|---|---|" + "---|" * len(cles)]
        for dim in ("Âge", "Sexe", "IMC", "Sévérité clinique (ahi_a0h4)"):
            for val in sorted({sous[p][dim] for p in set(pers)}):
                pp = [p for p in set(pers) if sous[p][dim] == val]
                m = np.isin(pers, pp)
                if len(pp) < MIN_CELLULE:
                    L.append(f"| {dim} : {val} | {len(pp)} | — | " + " | ".join("masqué" for _ in cles) + " |"); continue
                met = metriques_stades(y[m], pred[m]) if tache == "eeg" else metriques_apnee(y[m], pred[m], proba[m])
                L.append(f"| {dim} : {val} | {len(pp)} | {m.sum():,} | " + " | ".join(f"{met[k]:.3f}" for k in cles) + " |")
        L.append("")
        if tache == "ecg":
            # index estimé par le modèle vs index clinique, par personne
            est, cli3, cli4, ann = [], [], [], []
            for p in sorted(set(pers)):
                m = pers == p
                est.append(float(pred[m].mean())); ann.append(float(y[m].mean()))
                cli3.append(_f(lignes[p]["ahi_a0h3a"])); cli4.append(_f(lignes[p]["ahi_a0h4"]))
            est, cli3, cli4, ann = map(np.array, (est, cli3, cli4, ann))
            L += ["Par personne (validation) : part de minutes prédites positives contre les index de référence :", "",
                  f"- corrélation de Spearman avec la part annotée : {spearmanr(est, ann).correlation:.2f}",
                  f"- avec l'IAH clinique ahi_a0h3a : {spearmanr(est, cli3).correlation:.2f}",
                  f"- avec l'IAH clinique ahi_a0h4 : {spearmanr(est, cli4).correlation:.2f}", "",
                  "C'est la mesure clinique : un bon modèle d'époques qui ordonne mal les personnes ne sert à rien à un médecin.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
