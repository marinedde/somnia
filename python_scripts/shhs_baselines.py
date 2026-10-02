#!/usr/bin/env python3
"""
Étape 3 — Les références honnêtes sur SHHS : le chiffre à battre.

Pour chaque tâche (stades EEG, apnée ECG) :
  3.1  référence triviale : prédire la classe majoritaire de l'entraînement ;
  3.2  Random Forest, 16 caractéristiques, entraîné sur les personnes d'ENTRAÎNEMENT SHHS,
       mesuré sur les personnes de VALIDATION. Le test reste fermé jusqu'à la fin du projet ;
  3.3  validation externe : ce même modèle, appliqué tel quel à TOUT PhysioNet ;
  3.4  dans l'autre sens : le modèle PhysioNet (entraîné sur ses personnes d'entraînement),
       appliqué à la validation SHHS.
Plus, pour l'apnée, la mesure qui compte cliniquement : par personne, part de minutes prédites
positives contre part annotée (corrélation et erreur absolue), sur la validation.

Chaque score est aussi donné sans les époques aberrantes (EEG/ECG saturés), pour mesurer leur poids.

Sorties :
    models/shhs_baselines.json      tous les chiffres
    docs/RESULTATS_SHHS.md          le tableau (versionné, effectifs seulement)
    models/shhs_rf_eeg.joblib, models/shhs_rf_ecg.joblib   les deux références (hors git : *.joblib LFS,
                                                            on ne les déploie pas)
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.evaluation import metriques_apnee, metriques_stades, pipeline_rf  # noqa: E402
from somnia.physionet import charger_tableau  # noqa: E402
from somnia.splits import charger_decoupage, masques  # noqa: E402

SHHS = Path(os.environ.get("SHHS_PROCESSED", Path.home() / "data" / "shhs" / "processed"))
PHYSIO = ROOT / "data" / "processed"
SPLIT_PHYSIO = ROOT / "data" / "splits" / "physionet_v1.json"
OUT_JSON = ROOT / "models" / "shhs_baselines.json"
OUT_MD = ROOT / "docs" / "RESULTATS_SHHS.md"
CLES = {"eeg": ["accuracy", "f1_macro", "kappa", "f1_Wake", "f1_N1", "f1_N2", "f1_N3", "f1_REM"],
        "ecg": ["auc_roc", "auc_pr", "f1_apnee", "accuracy"]}
TITRES = {"eeg": "Stades de sommeil (EEG)", "ecg": "Apnée (ECG, fenêtres de 60 s en sommeil)"}


def _mesurer(tache, modele, X, y):
    pred = modele.predict(X)
    if tache == "eeg":
        return metriques_stades(y, pred)
    return metriques_apnee(y, pred, modele.predict_proba(X)[:, 1])


class Majoritaire:
    """Référence triviale : toujours la classe la plus fréquente de l'entraînement."""

    def __init__(self, y):
        self.classe = Counter(np.asarray(y).tolist()).most_common(1)[0][0]
        self.classes_ = sorted(set(np.asarray(y).tolist()))

    def predict(self, X):
        return np.full(len(X), self.classe)

    def predict_proba(self, X):
        p = np.zeros((len(X), 2)); p[:, self.classe if self.classe in (0, 1) else 0] = 1.0
        return p


def index_par_personne(personnes, y, pred, sommeil_h_par_personne=None):
    """Part de fenêtres positives annotées vs prédites, par personne (val). Corrélation et erreur absolue."""
    pers = np.asarray(personnes)
    vrai, est = [], []
    for p in sorted(set(pers)):
        m = pers == p
        vrai.append(float(y[m].mean())); est.append(float(pred[m].mean()))
    vrai, est = np.array(vrai), np.array(est)
    return {
        "n_personnes": int(len(vrai)),
        "correlation": float(np.corrcoef(vrai, est)[0, 1]) if len(vrai) > 2 else float("nan"),
        "erreur_absolue_mediane": float(np.median(np.abs(vrai - est))),
        "erreur_absolue_moyenne": float(np.mean(np.abs(vrai - est))),
    }


def main() -> int:
    res = {"date": date.today().isoformat(), "regle": "test SHHS fermé : toutes les mesures SHHS sont sur la validation"}
    physio_split = charger_decoupage(SPLIT_PHYSIO)["taches"]

    for tache in ("eeg", "ecg"):
        print(f"\n=== {TITRES[tache]} ===")
        s = charger_tableau(SHHS / f"features_{tache}.npz")
        tr, va = s["ensemble"] == "train", s["ensemble"] == "val"
        propre = ~s["aberrant"].astype(bool)
        Xtr, ytr, Xva, yva = s["X"][tr], s["y"][tr], s["X"][va], s["y"][va]
        r = {"n": {"train_personnes": len(set(s["personne"][tr])), "val_personnes": len(set(s["personne"][va])),
                   "train_lignes": int(tr.sum()), "val_lignes": int(va.sum()),
                   "val_lignes_propres": int((va & propre).sum())},
             "classes_val": {str(k): int(v) for k, v in sorted(Counter(yva.tolist()).items())}}

        # 3.1 majoritaire
        maj = Majoritaire(ytr)
        r["majoritaire_val"] = _mesurer(tache, maj, Xva, yva)
        print(f"  3.1 majoritaire (classe {maj.classe}) :", {k: round(r['majoritaire_val'][k], 3) for k in CLES[tache][:3]})

        # 3.2 RF SHHS -> validation SHHS
        rf = pipeline_rf().fit(Xtr, ytr)
        r["rf_shhs_val"] = _mesurer(tache, rf, Xva, yva)
        r["rf_shhs_val_sans_aberrantes"] = _mesurer(tache, rf, s["X"][va & propre], s["y"][va & propre])
        print(f"  3.2 RF SHHS -> val SHHS      :", {k: round(r['rf_shhs_val'][k], 3) for k in CLES[tache][:3]})
        print(f"      idem sans aberrantes     :", {k: round(r['rf_shhs_val_sans_aberrantes'][k], 3) for k in CLES[tache][:3]})
        joblib.dump(rf, ROOT / "models" / f"shhs_rf_{tache}.joblib")

        # 3.3 externe : RF SHHS -> tout PhysioNet
        p = charger_tableau(PHYSIO / f"{tache}_features.npz")
        r["rf_shhs_vers_physionet"] = _mesurer(tache, rf, p["X"], p["y"])
        r["n"]["physionet_personnes"] = len(set(p["personne"])); r["n"]["physionet_lignes"] = int(len(p["y"]))
        print(f"  3.3 RF SHHS -> PhysioNet     :", {k: round(r['rf_shhs_vers_physionet'][k], 3) for k in CLES[tache][:3]})

        # 3.4 externe : RF PhysioNet (train) -> validation SHHS
        mp = masques(physio_split[tache], p["personne"])
        rf_p = pipeline_rf().fit(p["X"][mp["train"]], p["y"][mp["train"]])
        r["rf_physionet_val_physionet"] = _mesurer(tache, rf_p, p["X"][mp["val"]], p["y"][mp["val"]])
        r["rf_physionet_vers_shhs_val"] = _mesurer(tache, rf_p, Xva, yva)
        print(f"  3.4 RF PhysioNet -> val SHHS :", {k: round(r['rf_physionet_vers_shhs_val'][k], 3) for k in CLES[tache][:3]})

        if tache == "ecg":
            r["par_personne_val"] = index_par_personne(s["personne"][va], yva, rf.predict(Xva))
            print("  par personne (val) :", {k: round(v, 3) for k, v in r["par_personne_val"].items()})
        res[tache] = r

    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# Résultats — références SHHS (étape 3)", "",
         f"*Généré le {res['date']} par `python_scripts/shhs_baselines.py`. Ne pas éditer à la main.*", "",
         "Règle : le **test SHHS reste fermé**. Tout ce qui est mesuré sur SHHS l'est sur les 40 personnes de validation. "
         "PhysioNet est utilisé en entier pour la validation externe du modèle SHHS, et par son propre découpage par "
         "personne pour le sens inverse.", ""]
    for tache in ("eeg", "ecg"):
        r = res[tache]; n = r["n"]
        L += [f"## {TITRES[tache]}", "",
              f"SHHS : {n['train_personnes']} personnes / {n['train_lignes']:,} lignes en entraînement, "
              f"{n['val_personnes']} personnes / {n['val_lignes']:,} lignes en validation "
              f"({n['val_lignes_propres']:,} sans époque aberrante). PhysioNet : {n['physionet_personnes']} personnes, "
              f"{n['physionet_lignes']:,} lignes.", "",
              "| Modèle | Entraîné sur | Mesuré sur | " + " | ".join(CLES[tache]) + " |",
              "|---|---|---|" + "---|" * len(CLES[tache])]
        lignes = [("Classe majoritaire", "SHHS train", "SHHS val", "majoritaire_val"),
                  ("Random Forest, 16 caract.", "SHHS train", "SHHS val", "rf_shhs_val"),
                  ("Random Forest, 16 caract.", "SHHS train", "SHHS val, sans aberrantes", "rf_shhs_val_sans_aberrantes"),
                  ("Random Forest, 16 caract.", "SHHS train", "**PhysioNet, tout** (externe)", "rf_shhs_vers_physionet"),
                  ("Random Forest, 16 caract.", "PhysioNet train", "PhysioNet val", "rf_physionet_val_physionet"),
                  ("Random Forest, 16 caract.", "PhysioNet train", "**SHHS val** (externe)", "rf_physionet_vers_shhs_val")]
        for nom, trs, mes, cle in lignes:
            L.append(f"| {nom} | {trs} | {mes} | " + " | ".join(f"{r[cle][k]:.3f}" for k in CLES[tache]) + " |")
        L.append("")
        if tache == "ecg":
            pp = r["par_personne_val"]
            L += [f"Par personne (validation, {pp['n_personnes']} personnes) : part de minutes positives annotée contre "
                  f"prédite, corrélation {pp['correlation']:.2f}, erreur absolue médiane {pp['erreur_absolue_mediane']:.3f} "
                  f"(moyenne {pp['erreur_absolue_moyenne']:.3f}).", ""]
    L += ["## Lecture", "",
          "- La ligne « classe majoritaire » est le plancher : un score qui ne la dépasse pas nettement ne vaut rien.",
          "- Les deux lignes **externe** sont celles qui comptent en santé : un modèle entraîné dans un centre, testé dans un autre.",
          "- Les écarts SHHS → PhysioNet et PhysioNet → SHHS ne sont pas symétriques : populations, appareils, et pour "
          "l'ECG définitions d'étiquettes différentes (minutes annotées contre paires d'époques).", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"\n→ {OUT_JSON.relative_to(ROOT)}\n→ {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
