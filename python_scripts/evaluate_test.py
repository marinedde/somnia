#!/usr/bin/env python3
"""
OUVERTURE DU TEST SHHS — une seule fois, à la fin du projet (3 octobre 2026).

La liste des modèles évalués est écrite ICI, avant d'exécuter quoi que ce soit, et ne sera pas
modifiée après avoir vu les chiffres. Si un modèle manque, la ligne est « non disponible ».
Le script refuse de tourner une seconde fois tant que docs/RESULTATS_TEST.md existe.

Modèles déclarés :
  stades : classe majoritaire ; Random Forest SHHS (shhs_rf_eeg) ; CNN de zéro 100 % (graine 42) ;
           CNN pré-entraîné affiné 10 % et 1 % (graine 42) ; Random Forest PhysioNet (externe).
  apnée  : classe majoritaire ; Random Forest SHHS (shhs_rf_ecg) ; CNN de zéro 100 % (graine 42) ;
           Random Forest PhysioNet (externe) ; par personne : Spearman avec l'index clinique.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import PROCESSED, charger_tableau as charger_signal, normaliser_lot, personnes_du_split  # noqa: E402
from somnia.deep.model import CNN1D  # noqa: E402
from somnia.evaluation import entrainer_rf, metriques_apnee, metriques_stades  # noqa: E402
from somnia.physionet import charger_tableau as charger_features  # noqa: E402
from somnia.splits import charger_decoupage, masques  # noqa: E402

OUT = ROOT / "docs" / "RESULTATS_TEST.md"
OUT_JSON = ROOT / "models" / "test_final.json"
CLES = {"eeg": ["accuracy", "kappa", "f1_macro", "f1_N1"], "ecg": ["auc_roc", "auc_pr", "f1_apnee"]}


def cnn_proba(chemin: Path, tache: str, tab) -> np.ndarray:
    m = CNN1D(5 if tache == "eeg" else 2); m.load_state_dict(torch.load(chemin, map_location="cpu")); m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tab), 1024):
            x = torch.from_numpy(tab.X[i:i + 1024].astype(np.float32)).unsqueeze(1)
            lg = m(normaliser_lot(x)).numpy()
            z = lg - lg.max(axis=1, keepdims=True); out.append(np.exp(z) / np.exp(z).sum(axis=1, keepdims=True))
    return np.concatenate(out)


def mesurer(tache, y, proba):
    pred = proba.argmax(axis=1)
    return metriques_stades(y, pred) if tache == "eeg" else metriques_apnee(y, pred, proba[:, 1])


def main() -> int:
    if OUT.exists():
        sys.exit(f"{OUT} existe : le test a déjà été ouvert. Il ne se rouvre pas.")
    res = {"date": date.today().isoformat(), "regle": "ouvert une seule fois ; liste des modèles déclarée avant exécution"}
    physio_split = charger_decoupage(ROOT / "data" / "splits" / "physionet_v1.json")["taches"]
    L = ["# Résultats sur le TEST SHHS — ouvert une seule fois", "",
         f"*{res['date']}. 40 personnes jamais vues, ni pour entraîner, ni pour choisir quoi que ce soit. "
         "La liste des modèles a été écrite avant d'exécuter ; ce fichier ne sera pas régénéré.*", ""]
    for tache in ("eeg", "ecg"):
        f = charger_features(PROCESSED / f"features_{tache}.npz")
        te = f["ensemble"] == "test"; Xte, yte = f["X"][te], f["y"][te]
        lignes = {}
        maj = int(np.bincount(f["y"][f["ensemble"] == "train"]).argmax())
        pm = np.zeros((len(yte), 5 if tache == "eeg" else 2)); pm[:, maj] = 1
        lignes["Classe majoritaire"] = mesurer(tache, yte, pm)
        rf = joblib.load(ROOT / "models" / f"shhs_rf_{tache}.joblib")
        lignes["Random Forest SHHS, 16 caract."] = mesurer(tache, yte, rf.predict_proba(Xte))
        p = charger_features(ROOT / "data" / "processed" / f"{tache}_features.npz")
        mp = masques(physio_split[tache], p["personne"])
        rf_p = entrainer_rf(p["X"][mp["train"]], p["y"][mp["train"]])
        lignes["Random Forest PhysioNet (externe)"] = mesurer(tache, yte, rf_p.predict_proba(Xte))
        sig = charger_signal(tache, personnes_du_split("test"))
        declares = [("CNN de zéro, 100 %", f"{tache}_f1_s42.pt")]
        if tache == "eeg":
            declares += [("CNN pré-entraîné affiné, 10 %", "eeg_f0.1_s42_pre.pt"), ("CNN pré-entraîné affiné, 1 %", "eeg_f0.01_s42_pre.pt")]
        pp = {}
        for nom, fichier in declares:
            chemin = ROOT / "models" / "cnn" / fichier
            if chemin.exists():
                proba = cnn_proba(chemin, tache, sig)
                lignes[nom] = mesurer(tache, sig.y, proba)
                if tache == "ecg":
                    pp[nom] = (proba.argmax(axis=1), sig.y, sig.personne)
            else:
                lignes[nom] = None
        if tache == "ecg":
            pp["Random Forest SHHS, 16 caract."] = (rf.predict(Xte), yte, f["personne"][te])
            cov = {}
            with open(PROCESSED / "covariables.csv", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    cov[f"shhs-{r['nsrrid']}"] = r
            from scipy.stats import spearmanr
            for nom, (pred, y, pers) in pp.items():
                est, a3 = [], []
                for q in sorted(set(pers)):
                    mm = pers == q; est.append(pred[mm].mean()); a3.append(float(cov[q]["ahi_a0h3a"] or "nan"))
                lignes[nom]["spearman_ahi_a0h3a"] = float(spearmanr(est, a3, nan_policy="omit").correlation)
        res[tache] = {"n_personnes_test": int(len(set(f["personne"][te]))), "n_lignes_test": int(te.sum()), "modeles": lignes}
        cles = CLES[tache] + (["spearman_ahi_a0h3a"] if tache == "ecg" else [])
        L += [f"## {'Stades (EEG)' if tache == 'eeg' else 'Apnée (ECG)'} — {res[tache]['n_personnes_test']} personnes, {res[tache]['n_lignes_test']:,} lignes", "",
              "| Modèle | " + " | ".join(cles) + " |", "|---|" + "---|" * len(cles)]
        for nom, m in lignes.items():
            L.append(f"| {nom} | " + (" | ".join(f"{m.get(k, float('nan')):.3f}" for k in cles) if m else "non disponible") + " |")
        L.append("")
    L += ["## Lecture", "", "À comparer aux chiffres de validation (`docs/RESULTATS_SHHS.md`, `docs/RESULTATS_CNN.md`) : "
          "un écart important entre validation et test signifierait que la validation a été trop regardée.", ""]
    OUT.write_text("\n".join(L), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
