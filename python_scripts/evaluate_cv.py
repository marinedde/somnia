#!/usr/bin/env python3
"""
Tâches 0.3, 0.4 et 0.13 — Mesure par personne des deux modèles Random Forest.

Pour chaque tâche :
  1. validation croisée par personne (5 plis) : le chiffre honnête ;
  2. l'ancien découpage aléatoire par époque, refait à l'identique : le chiffre « avant » ;
  3. pour l'ECG, les deux vraies courbes ROC (plus de courbe dessinée à la main).

Sorties :
    models/cv_results.json            tout, lisible par retrain.py et par l'API
    docs/RESULTATS.md                 le tableau de résultats du projet
    data/figures/fuite_roc_ecg.png    courbes ROC avant / après
    data/figures/fuite_par_personne.png   barres avant / après, deux tâches
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.evaluation import decoupage_aleatoire_par_epoque, validation_croisee_par_personne  # noqa: E402
from somnia.physionet import charger_tableau  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "data" / "figures"
RESULTS_JSON = ROOT / "models" / "cv_results.json"
RESULTS_MD = ROOT / "docs" / "RESULTATS.md"

CLES = {
    "eeg": ["accuracy", "f1_macro", "kappa", "f1_Wake", "f1_N1", "f1_N2", "f1_N3", "f1_REM"],
    "ecg": ["auc_roc", "auc_pr", "f1_apnee", "accuracy"],
}
TITRES = {"eeg": "Stades de sommeil (EEG, Sleep-EDF)", "ecg": "Apnée (ECG, Apnea-ECG)"}


def _pm(moy: float, ecart: float) -> str:
    return f"{moy:.3f} ± {ecart:.3f}"


def tableau_markdown(res: dict) -> str:
    lignes = [
        "# Résultats — Somnia",
        "",
        f"*Généré le {res['date']} par `python_scripts/evaluate_cv.py`. Ne pas éditer à la main.*",
        "",
        "Deux chiffres par métrique : **avant** (découpage aléatoire par époque, celui des notebooks "
        "d'origine, fuité) et **après** (validation croisée par personne, 5 plis, moyenne ± écart-type). "
        "Même modèle, mêmes 16 caractéristiques, mêmes données. Seul le découpage change.",
        "",
    ]
    for tache in ("eeg", "ecg"):
        r = res[tache]
        lignes += [
            f"## {TITRES[tache]}",
            "",
            f"{r['n_personnes']} personnes, {r['n_enregistrements']} enregistrements, {r['n_epoques']:,} époques.",
            "",
            "| Métrique | Avant : aléatoire par époque | Après : par personne (5 plis) |",
            "|---|---|---|",
        ]
        for k in CLES[tache]:
            lignes.append(f"| {k} | {r['aleatoire']['metriques'][k]:.3f} | "
                          f"{_pm(r['par_personne']['moyenne'][k], r['par_personne']['ecart_type'][k])} |")
        lignes += ["", "Détail par pli (personnes mises de côté) :", "",
                   "| Pli | n personnes | n époques | " + " | ".join(CLES[tache][:3]) + " |",
                   "|---|---|---|" + "---|" * 3]
        for i, p in enumerate(r["par_personne"]["plis"], 1):
            lignes.append(f"| {i} | {p['n_personnes_test']} | {p['n_epoques_test']:,} | "
                          + " | ".join(f"{p[k]:.3f}" for k in CLES[tache][:3]) + " |")
        lignes.append("")
    lignes += [
        "## Lecture",
        "",
        "- La baisse entre les deux colonnes mesure la **fuite** de l'ancien découpage : deux nuits de la "
        "même personne (Sleep-EDF) et des minutes voisines du même enregistrement (Apnea-ECG) se "
        "retrouvaient des deux côtés.",
        "- Les écarts-types sont grands parce que les plis contiennent peu de personnes (3 à 4 pour l'EEG, "
        "6 pour l'ECG). C'est la vraie incertitude de ce que l'on peut dire avec 16 et 30 personnes.",
        "- Figures : `data/figures/fuite_roc_ecg.png` (vraies courbes ROC) et `data/figures/fuite_par_personne.png`.",
        "",
    ]
    return "\n".join(lignes)


def figures(res: dict, oof_ecg: dict, aleatoire_ecg: dict, y_ecg: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. ROC ECG : les deux courbes sont calculées, pas dessinées
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(aleatoire_ecg["y_test"], aleatoire_ecg["proba"])
    ax.plot(fpr, tpr, color="#048A81", lw=2,
            label=f"Aléatoire par époque (fuite)  AUC = {res['ecg']['aleatoire']['metriques']['auc_roc']:.3f}")
    ok = ~np.isnan(oof_ecg["proba"])
    fpr, tpr, _ = roc_curve(y_ecg[ok], oof_ecg["proba"][ok])
    ax.plot(fpr, tpr, color="#E84855", lw=2, ls="--",
            label=f"Par personne, hors pli  AUC = {res['ecg']['par_personne']['moyenne']['auc_roc']:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Hasard")
    ax.set_xlabel("Taux de faux positifs"); ax.set_ylabel("Sensibilité")
    ax.set_title("Apnée (ECG) — même modèle, deux découpages")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "fuite_roc_ecg.png", dpi=150); plt.close(fig)

    # 2. Barres avant / après pour les deux tâches
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, tache, cles in zip(axes, ("eeg", "ecg"), (["accuracy", "f1_macro", "kappa"], ["auc_roc", "auc_pr", "f1_apnee"])):
        r = res[tache]
        avant = [r["aleatoire"]["metriques"][k] for k in cles]
        apres = [r["par_personne"]["moyenne"][k] for k in cles]
        err = [r["par_personne"]["ecart_type"][k] for k in cles]
        x = np.arange(len(cles)); w = 0.38
        ax.bar(x - w / 2, avant, w, color="#048A81", label="Aléatoire par époque (fuite)")
        ax.bar(x + w / 2, apres, w, yerr=err, capsize=4, color="#E84855", label="Par personne (5 plis)")
        for xi, v in zip(x - w / 2, avant):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        for xi, v in zip(x + w / 2, apres):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(cles); ax.set_ylim(0, 1.05)
        ax.set_title(TITRES[tache]); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Fuite de données : ce que change le découpage par personne", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "fuite_par_personne.png", dpi=150); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plis", type=int, default=5)
    parser.add_argument("--graine", type=int, default=42)
    args = parser.parse_args()

    res = {"date": date.today().isoformat(), "graine": args.graine, "n_plis": args.plis}
    extras = {}
    for tache in ("eeg", "ecg"):
        t = charger_tableau(PROCESSED / f"{tache}_features.npz")
        print(f"\n=== {TITRES[tache]} : {len(t['y']):,} époques, {len(set(t['personne']))} personnes ===")
        cv = validation_croisee_par_personne(t["X"], t["y"], t["personne"], tache, args.plis, args.graine)
        alea = decoupage_aleatoire_par_epoque(t["X"], t["y"], tache, args.graine)
        for k in CLES[tache]:
            print(f"   {k:10s}  avant {alea['metriques'][k]:.3f}   après {_pm(cv['moyenne'][k], cv['ecart_type'][k])}")
        res[tache] = {
            "n_personnes": cv["n_personnes"],
            "n_enregistrements": int(len(set(t["enregistrement"]))),
            "n_epoques": cv["n_epoques"],
            "par_personne": {k: v for k, v in cv.items() if k != "oof"},
            "aleatoire": {"methode": alea["methode"], "metriques": alea["metriques"]},
        }
        extras[tache] = (cv["oof"], alea, t["y"])

    RESULTS_JSON.parent.mkdir(exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    RESULTS_MD.write_text(tableau_markdown(res), encoding="utf-8")
    figures(res, *extras["ecg"])
    print(f"\n→ {RESULTS_JSON.relative_to(ROOT)}\n→ {RESULTS_MD.relative_to(ROOT)}\n→ {FIGURES.relative_to(ROOT)}/fuite_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
