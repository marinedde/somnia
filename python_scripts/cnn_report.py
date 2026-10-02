#!/usr/bin/env python3
"""
Étape 4 — Le tableau de résultats des réseaux, à côté des références (docs/RESULTATS_CNN.md).

Lit models/cnn/*.json (un par entraînement) et models/shhs_baselines.json, écrit le tableau
(validation SHHS seulement : le test reste fermé) et deux figures agrégées :
  data/figures/cnn_courbes.png     pertes et score de validation par époque, par run
  data/figures/cnn_etiquettes.png  score en fonction de la fraction de personnes étiquetées

Usage : python python_scripts/cnn_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CNN = ROOT / "models" / "cnn"
BASE = ROOT / "models" / "shhs_baselines.json"
OUT = ROOT / "docs" / "RESULTATS_CNN.md"
FIG = ROOT / "data" / "figures"
CLES = {"eeg": ["accuracy", "kappa", "f1_macro", "f1_N1"], "ecg": ["auc_roc", "auc_pr", "f1_apnee"]}
TITRES = {"eeg": "Stades de sommeil (EEG, époques de 30 s)", "ecg": "Apnée (ECG, fenêtres de 60 s en sommeil)"}


def main() -> int:
    runs = sorted((json.loads(p.read_text(encoding="utf-8")) for p in CNN.glob("*_s*.json") if "petit_lot" not in p.name),
                  key=lambda r: (r["tache"], -r["fraction"], r["graine"]))
    base = json.loads(BASE.read_text(encoding="utf-8")) if BASE.exists() else {}
    if not runs:
        sys.exit("aucun entraînement dans models/cnn")

    L = ["# Résultats — réseaux convolutifs (étape 4)", "",
         f"*Généré le {date.today().isoformat()} par `python_scripts/cnn_report.py`. Validation SHHS uniquement "
         "(40 personnes) : le test n'a pas été ouvert. Une graine par ligne ; l'écart entre graines reste à mesurer.*", ""]
    for tache in ("eeg", "ecg"):
        rs = [r for r in runs if r["tache"] == tache]
        if not rs:
            continue
        cles = CLES[tache]
        L += [f"## {TITRES[tache]}", "",
              "| Modèle | Personnes étiquetées | Exemples | Meilleure époque | " + " | ".join(cles) + " | ECE avant → après | Durée |",
              "|---|---|---|---|" + "---|" * len(cles) + "---|---|"]
        if base.get(tache):
            b = base[tache]
            L.append(f"| Classe majoritaire | — | — | — | " + " | ".join(f"{b['majoritaire_val'][k]:.3f}" for k in cles) + " | — | — |")
            L.append(f"| Random Forest, 16 caract. | {b['n']['train_personnes']} (100 %) | {b['n']['train_lignes']:,} | — | "
                     + " | ".join(f"{b['rf_shhs_val'][k]:.3f}" for k in cles) + " | — | — |")
        for r in rs:
            c = r["calibration"]
            L.append(f"| CNN 1D, de zéro | {r['n_personnes_train']} ({r['fraction']:.0%}) | {r['n_exemples_train']:,} | {r['meilleure_epoque']} | "
                     + " | ".join(f"{r['validation'][k]:.3f}" for k in cles)
                     + f" | {c['ece_avant']:.3f} → {c['ece_apres']:.3f} | {r['duree_min']} min |")
        L.append("")
        plein = next((r for r in rs if r["fraction"] >= 1.0), None)
        if plein:
            L += ["Courbe précision / couverture (modèle à 100 %) : on ne classe que les époques les plus sûres.", "",
                  "| Part gardée | Seuil de confiance | " + ("accuracy | kappa" if tache == "eeg" else "accuracy | f1_apnee") + " |", "|---|---|---|---|"]
            for c in plein["couverture"]:
                k2 = "kappa" if tache == "eeg" else "f1_apnee"
                L.append(f"| {c['couverture']:.0%} | {c['seuil_confiance']:.2f} | {c['accuracy']:.3f} | {c[k2]:.3f} |")
            L.append("")
            if tache == "ecg" and plein.get("par_personne"):
                pp = plein["par_personne"]
                L += ["Par personne (validation), corrélation de Spearman entre la part de minutes prédites positives et :", "",
                      f"- la part annotée : {pp.get('spearman_part_annotee', float('nan')):.2f}",
                      f"- l'index clinique ahi_a0h3a : {pp.get('spearman_ahi_a0h3a', float('nan')):.2f}",
                      f"- l'index clinique ahi_a0h4 : {pp.get('spearman_ahi_a0h4', float('nan')):.2f}", ""]
                if base.get("ecg", {}).get("par_personne_val"):
                    L.append(f"Référence Random Forest sur la même mesure : {base['ecg']['par_personne_val']['correlation']:.2f} avec la part annotée.")
                    L.append("")
    L += ["## Lecture", "",
          "- Les lignes « 10 % » et « 1 % » mesurent la faim d'étiquettes du réseau : c'est le trou que le pré-entraînement de l'étape 5 doit combler.",
          "- L'ECE est mesuré sur une moitié des personnes de validation, la température étant ajustée sur l'autre, puis l'inverse.",
          "- Figures : `data/figures/cnn_courbes.png`, `data/figures/cnn_etiquettes.png`.", ""]
    OUT.write_text("\n".join(L), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, tache in zip(axes, ("eeg", "ecg")):
        for r in [r for r in runs if r["tache"] == tache]:
            ep = [e["epoque"] for e in r["historique"]]
            ax.plot(ep, [e["perte_val"] for e in r["historique"]], marker="o", ms=3, label=f"val, {r['fraction']:.0%} des personnes")
            ax.plot(ep, [e["perte_train"] for e in r["historique"]], ls="--", alpha=0.6, color=ax.lines[-1].get_color())
        ax.set_title(TITRES[tache]); ax.set_xlabel("époque"); ax.set_ylabel("perte (— train, ● val)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Courbes de perte", fontweight="bold"); fig.tight_layout(); fig.savefig(FIG / "cnn_courbes.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, tache, cle in zip(axes, ("eeg", "ecg"), ("kappa", "auc_pr")):
        rs = sorted([r for r in runs if r["tache"] == tache], key=lambda r: r["fraction"])
        if rs:
            ax.plot([r["fraction"] * 100 for r in rs], [r["validation"][cle] for r in rs], marker="o", label="CNN 1D, de zéro", color="#E84855")
        if base.get(tache):
            ax.axhline(base[tache]["rf_shhs_val"][cle], color="#048A81", ls="--", label="Random Forest, 100 % des personnes")
            ax.axhline(base[tache]["majoritaire_val"][cle], color="grey", ls=":", label="classe majoritaire")
        ax.set_xscale("log"); ax.set_xlabel("% des personnes d'entraînement étiquetées"); ax.set_ylabel(cle + " (validation)")
        ax.set_title(TITRES[tache]); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle("Faim d'étiquettes : score selon la fraction de personnes étiquetées", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "cnn_etiquettes.png", dpi=130); plt.close(fig)
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
