#!/usr/bin/env python3
"""
Réentraînement des deux modèles Somnia (EEG stades, ECG apnée) avec le découpage PAR PERSONNE.

Prérequis :
    python python_scripts/prepare_features.py     # data/processed/{eeg,ecg}_features.npz
    python python_scripts/make_split.py           # data/splits/physionet_v1.json
    python python_scripts/evaluate_cv.py          # models/cv_results.json (chiffres par personne)

Usage :
    python python_scripts/retrain.py
    python python_scripts/retrain.py --task eeg --min-kappa 0.5
    python python_scripts/retrain.py --dry-run

Garde-fou : si la métrique sur la VALIDATION (personnes jamais vues) passe sous le
seuil, le modèle n'est pas écrit. Les seuils par défaut sont fixés d'après la
validation croisée (docs/RESULTATS.md) : un peu sous la moyenne moins un écart-type.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.evaluation import RF_PARAMS, metriques_apnee, metriques_stades, pipeline_rf  # noqa: E402
from somnia.physionet import charger_tableau  # noqa: E402
from somnia.splits import charger_decoupage, masques  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
SPLIT_PATH = ROOT / "data" / "splits" / "physionet_v1.json"
MODELS_DIR = ROOT / "models"
BASELINE_PATH = MODELS_DIR / "baseline_stats.json"
METRICS_PATH = MODELS_DIR / "training_metrics.json"
CV_PATH = MODELS_DIR / "cv_results.json"

MODEL_FILES = {"eeg": "somnia_eeg_pipeline.joblib", "ecg": "somnia_ecg_pipeline.joblib"}
TASK_KEYS = {"eeg": "sleep_stage", "ecg": "apnea"}   # clés attendues par app/baseline.py
DATASETS = {
    "eeg": "Sleep-EDF Expanded (PhysioNet) — 16 personnes, 28 enregistrements",
    "ecg": "Apnea-ECG (PhysioNet) — 31 enregistrements (30 groupes, c05 = c06)",
}


def _metriques(tache, modele, X, y):
    pred = modele.predict(X)
    if tache == "eeg":
        return metriques_stades(y, pred)
    return metriques_apnee(y, pred, modele.predict_proba(X)[:, 1])


def entrainer(tache: str, seuil: float, cle_seuil: str) -> tuple[dict, dict]:
    t = charger_tableau(PROCESSED / f"{tache}_features.npz")
    decoupage = charger_decoupage(SPLIT_PATH)["taches"][tache]
    m = masques(decoupage, t["personne"])
    assert not np.any(m["train"] & m["test"]) and not np.any(m["train"] & m["val"])

    modele = pipeline_rf(RF_PARAMS["random_state"]).fit(t["X"][m["train"]], t["y"][m["train"]])
    val = _metriques(tache, modele, t["X"][m["val"]], t["y"][m["val"]])
    test = _metriques(tache, modele, t["X"][m["test"]], t["y"][m["test"]])

    if val[cle_seuil] < seuil:
        raise RuntimeError(f"[{tache}] {cle_seuil} validation = {val[cle_seuil]:.3f} < seuil {seuil}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(modele, MODELS_DIR / MODEL_FILES[tache])

    resultat = {
        "model_path": f"models/{MODEL_FILES[tache]}",
        "dataset": DATASETS[tache],
        "split": {
            "fichier": str(SPLIT_PATH.relative_to(ROOT)),
            "methode": "par personne",
            "n_personnes": {k: len(v) for k, v in decoupage.items()},
            "n_epoques": {k: int(m[k].sum()) for k in m},
        },
        "garde_fou": {"metrique": cle_seuil, "seuil": seuil, "valeur_val": val[cle_seuil]},
        "val": val,
        "test": test,
    }
    baseline = {
        "n_samples": int(m["train"].sum()),
        "mean": t["X"][m["train"]].mean(axis=0).tolist(),
        "std": t["X"][m["train"]].std(axis=0).tolist(),
        "feature_names": t["feature_names"].tolist(),
    }
    return resultat, baseline


def _resume_cv(tache: str) -> dict | None:
    """Le chiffre à afficher est celui de la validation croisée, pas celui d'un seul test de 2 personnes."""
    if not CV_PATH.exists():
        return None
    cv = json.loads(CV_PATH.read_text(encoding="utf-8")).get(tache)
    if not cv:
        return None
    return {
        "methode": cv["par_personne"]["methode"],
        "moyenne": cv["par_personne"]["moyenne"],
        "ecart_type": cv["par_personne"]["ecart_type"],
        "avant_fuite": cv["aleatoire"]["metriques"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Réentraînement Somnia (découpage par personne)")
    parser.add_argument("--task", choices=["all", "eeg", "ecg"], default="all")
    parser.add_argument("--min-kappa", type=float, default=0.50, help="seuil kappa validation (EEG) : CV 0,63 ± 0,08")
    parser.add_argument("--min-auc", type=float, default=0.65, help="seuil AUC-ROC validation (ECG) : CV 0,77 ± 0,08")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requis = [PROCESSED / "eeg_features.npz", PROCESSED / "ecg_features.npz", SPLIT_PATH]
    if args.dry_run:
        for p in requis:
            print(f"[DRY-RUN] {'OK      ' if p.exists() else 'Manquant'} {p.relative_to(ROOT)}")
        print("[DRY-RUN] Script OK." if all(p.exists() for p in requis)
              else "[DRY-RUN] Lance prepare_features.py puis make_split.py.")
        return 0

    manquants = [p for p in requis if not p.exists()]
    if manquants:
        print("❌ Fichiers manquants :", ", ".join(str(p.relative_to(ROOT)) for p in manquants))
        return 1

    payload = {"training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "split_method": "par personne"}
    baseline = {"updated_at": datetime.now(timezone.utc).isoformat(), "tasks": {}}
    if BASELINE_PATH.exists():  # ne pas perdre la référence de l'autre tâche si on n'en réentraîne qu'une
        baseline["tasks"] = json.loads(BASELINE_PATH.read_text(encoding="utf-8")).get("tasks", {})
    if METRICS_PATH.exists():
        ancien = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        payload.update({k: ancien[k] for k in ("eeg", "ecg") if k in ancien})

    try:
        if args.task in ("all", "eeg"):
            print("=== EEG — stades de sommeil ===")
            res, base = entrainer("eeg", args.min_kappa, "kappa")
            res["cv"] = _resume_cv("eeg")
            payload["eeg"], baseline["tasks"][TASK_KEYS["eeg"]] = res, base
            print(f"val  : kappa {res['val']['kappa']:.3f}  acc {res['val']['accuracy']:.3f}")
            print(f"test : kappa {res['test']['kappa']:.3f}  acc {res['test']['accuracy']:.3f}")
        if args.task in ("all", "ecg"):
            print("=== ECG — apnée ===")
            res, base = entrainer("ecg", args.min_auc, "auc_roc")
            res["cv"] = _resume_cv("ecg")
            payload["ecg"], baseline["tasks"][TASK_KEYS["ecg"]] = res, base
            print(f"val  : AUC {res['val']['auc_roc']:.3f}  AUC-PR {res['val']['auc_pr']:.3f}")
            print(f"test : AUC {res['test']['auc_roc']:.3f}  AUC-PR {res['test']['auc_pr']:.3f}")
    except RuntimeError as e:
        print(f"❌ Qualité insuffisante — modèle non écrit : {e}")
        return 2

    BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBaseline  → {BASELINE_PATH.relative_to(ROOT)}\nMétriques → {METRICS_PATH.relative_to(ROOT)}")
    print("✅ Terminé. Redémarre l'API pour charger les nouveaux modèles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
