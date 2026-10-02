#!/usr/bin/env python3
"""
EDA de la cohorte SHHS préparée (avant toute modélisation).

Pour chaque nuit retenue (.npz), sans jamais garder les signaux en mémoire :
  - par époque : écart-type et maximum absolu de l'EEG (µV) et de l'ECG (mV), part
    d'échantillons ECG saturés (butée du convertisseur, ±1,25 mV dans SHHS1) ;
  - par nuit : proportions de stades, part d'époques apnée, époques plates ou extrêmes.

Puis, sur l'ensemble :
  - distributions (figures agrégées, sans identifiant ni tracé brut : publiables) ;
  - nuits et époques aberrantes, avec des seuils ÉCRITS ICI, pas choisis après coup ;
  - équilibre des classes par ensemble du découpage (train / val / test).

Sorties :
    docs/EDA_SHHS.md                           chiffres agrégés (versionné)
    data/figures/shhs_eda_*.png                figures agrégées (versionnées)
    ~/data/shhs/processed/eda_nuits.csv        une ligne par nuit, avec identifiants (hors git)

Usage :
    python python_scripts/shhs_eda.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.shhs_prepare import charger_nuit  # noqa: E402

PROCESSED = Path(os.environ.get("SHHS_PROCESSED", Path.home() / "data" / "shhs" / "processed"))
SPLIT = ROOT / "data" / "splits" / "shhs_v1.json"
FIGURES = ROOT / "data" / "figures"
RAPPORT = ROOT / "docs" / "EDA_SHHS.md"
NOMS = {-1: "Inconnu", 0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# Seuils d'aberration, fixés avant de regarder les résultats
EEG_STD_PLAT = 1.0        # µV  : électrode décollée ou canal mort
EEG_STD_EXTREME = 200.0   # µV  : artefact (en pratique inatteignable : plage physique ±125 µV)
EEG_SAT_UV = 120.0        # µV  : butée du convertisseur SHHS1 (±125 µV, léger dépassement dû au rééchantillonnage)
ECG_SAT_MV = 1.24         # mV  : butée du convertisseur SHHS1 (±1,25 mV)
PART_EPOQUES_ABERRANTES_NUIT = 0.20   # une nuit est signalée au-delà de 20 % d'époques aberrantes


def stats_nuit(chemin: Path) -> tuple[dict, dict]:
    n = charger_nuit(chemin)
    eeg, ecg = n["eeg"], n["ecg"]
    eeg_std, ecg_std = eeg.std(axis=1), ecg.std(axis=1)
    eeg_max, ecg_max = np.abs(eeg).max(axis=1), np.abs(ecg).max(axis=1)
    ecg_sat = (np.abs(ecg) >= ECG_SAT_MV).mean(axis=1)          # part d'échantillons saturés
    eeg_sat = (np.abs(eeg) >= EEG_SAT_UV).mean(axis=1)
    stades, apnee = n["stades"], n["apnee"]
    sommeil = np.isin(stades, [1, 2, 3, 4])

    aberr_eeg = (eeg_std < EEG_STD_PLAT) | (eeg_std > EEG_STD_EXTREME) | (eeg_sat > 0.05)
    aberr_ecg = (ecg_std < 1e-6) | (ecg_sat > 0.05)
    par_epoque = {
        "eeg_std": eeg_std, "ecg_std": ecg_std, "eeg_max": eeg_max, "ecg_max": ecg_max,
        "ecg_sat": ecg_sat, "eeg_sat": eeg_sat, "stades": stades, "apnee": apnee,
    }
    c = Counter(stades.tolist())
    ligne = {
        "enregistrement": n["meta"]["enregistrement"],
        "personne": n["meta"]["personne"],
        "n_epoques": len(stades),
        "temps_sommeil_h": round(sommeil.sum() * 30 / 3600, 2),
        **{f"part_{NOMS[k]}": round(c.get(k, 0) / len(stades), 3) for k in (0, 1, 2, 3, 4, -1)},
        "part_apnee_sommeil": round(float(apnee[sommeil].mean()) if sommeil.any() else 0.0, 3),
        "eeg_std_mediane": round(float(np.median(eeg_std)), 1),
        "ecg_std_mediane": round(float(np.median(ecg_std)), 4),
        "part_eeg_plat": round(float((eeg_std < EEG_STD_PLAT).mean()), 3),
        "part_eeg_extreme": round(float((eeg_std > EEG_STD_EXTREME).mean()), 3),
        "part_ecg_sature": round(float((ecg_sat > 0.05).mean()), 3),
        "part_eeg_sature": round(float((eeg_sat > 0.05).mean()), 3),
        "part_epoques_aberrantes": round(float((aberr_eeg | aberr_ecg).mean()), 3),
    }
    return ligne, par_epoque


def main() -> int:
    fichiers = sorted(PROCESSED.glob("shhs1-*.npz"))
    if not fichiers:
        sys.exit(f"aucune nuit préparée sous {PROCESSED}")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))["taches"]["shhs"] if SPLIT.exists() else None
    ensemble_de = {p: k for k, v in (split or {}).items() for p in v}

    lignes, agg = [], {k: [] for k in ("eeg_std", "ecg_std", "eeg_max", "ecg_max", "ecg_sat", "eeg_sat", "stades", "apnee", "ensemble")}
    for i, f in enumerate(fichiers, 1):
        ligne, pe = stats_nuit(f)
        ligne["ensemble"] = ensemble_de.get(ligne["personne"], "?")
        lignes.append(ligne)
        for k in ("eeg_std", "ecg_std", "eeg_max", "ecg_max", "ecg_sat", "eeg_sat", "stades", "apnee"):
            agg[k].append(pe[k])
        agg["ensemble"].append(np.full(len(pe["stades"]), ligne["ensemble"], dtype=object))
        if i % 50 == 0:
            print(f"  {i}/{len(fichiers)} nuits")
    A = {k: np.concatenate(v) for k, v in agg.items()}
    n_ep = len(A["stades"])

    # ── Tableau par nuit (hors git) ──
    with open(PROCESSED / "eda_nuits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader(); w.writerows(lignes)

    # ── Chiffres agrégés ──
    def q(x, p): return float(np.percentile(x, p))
    sommeil = np.isin(A["stades"], [1, 2, 3, 4])
    stades_c = Counter(A["stades"].tolist())
    aberr = ((A["eeg_std"] < EEG_STD_PLAT) | (A["eeg_std"] > EEG_STD_EXTREME) |
             (A["ecg_std"] < 1e-6) | (A["ecg_sat"] > 0.05) | (A["eeg_sat"] > 0.05))
    nuits_signalees = [l for l in lignes if l["part_epoques_aberrantes"] > PART_EPOQUES_ABERRANTES_NUIT]

    L = [f"# EDA — cohorte SHHS préparée", "",
         f"*Généré le {date.today().isoformat()} par `python_scripts/shhs_eda.py` sur {len(lignes)} nuits, "
         f"{n_ep:,} époques. Chiffres agrégés seulement ; le détail par nuit (avec identifiants) reste hors dépôt.*", "",
         "## Stades et étiquettes", "",
         "| Stade | Époques | Part |", "|---|---|---|"]
    for k in (0, 1, 2, 3, 4, -1):
        L.append(f"| {NOMS[k]} | {stades_c.get(k, 0):,} | {stades_c.get(k, 0) / n_ep:.1%} |")
    L += ["", f"Époques de sommeil : {sommeil.sum():,} ({sommeil.mean():.1%}). "
              f"Part d'époques « apnée » pendant le sommeil : **{A['apnee'][sommeil].mean():.1%}** ; pendant l'éveil "
              f"(à exclure de la tâche) : {A['apnee'][~sommeil].mean():.1%}.", ""]
    if split:
        L += ["### Par ensemble du découpage", "", "| Ensemble | Nuits | Époques | Part N1 | Part N3 | Part REM | Part apnée (sommeil) |",
              "|---|---|---|---|---|---|---|"]
        for ens in ("train", "val", "test"):
            m = A["ensemble"] == ens
            ms = m & sommeil
            st = A["stades"][m]
            L.append(f"| {ens} | {sum(l['ensemble'] == ens for l in lignes)} | {m.sum():,} | "
                     f"{(st == 1).mean():.1%} | {(st == 3).mean():.1%} | {(st == 4).mean():.1%} | "
                     f"{A['apnee'][ms].mean():.1%} |")
        L.append("")
    L += ["## Amplitudes des signaux (par époque)", "",
          "| Signal | p1 | p5 | médiane | p95 | p99 | max |", "|---|---|---|---|---|---|---|",
          f"| EEG, écart-type (µV) | {q(A['eeg_std'], 1):.1f} | {q(A['eeg_std'], 5):.1f} | {q(A['eeg_std'], 50):.1f} | "
          f"{q(A['eeg_std'], 95):.1f} | {q(A['eeg_std'], 99):.1f} | {A['eeg_std'].max():.0f} |",
          f"| EEG, max absolu (µV) | {q(A['eeg_max'], 1):.0f} | {q(A['eeg_max'], 5):.0f} | {q(A['eeg_max'], 50):.0f} | "
          f"{q(A['eeg_max'], 95):.0f} | {q(A['eeg_max'], 99):.0f} | {A['eeg_max'].max():.0f} |",
          f"| ECG, écart-type (mV) | {q(A['ecg_std'], 1):.3f} | {q(A['ecg_std'], 5):.3f} | {q(A['ecg_std'], 50):.3f} | "
          f"{q(A['ecg_std'], 95):.3f} | {q(A['ecg_std'], 99):.3f} | {A['ecg_std'].max():.2f} |",
          f"| ECG, max absolu (mV) | {q(A['ecg_max'], 1):.2f} | {q(A['ecg_max'], 5):.2f} | {q(A['ecg_max'], 50):.2f} | "
          f"{q(A['ecg_max'], 95):.2f} | {q(A['ecg_max'], 99):.2f} | {A['ecg_max'].max():.2f} |", "",
          "Écart-type médian de l'EEG par stade (attendu : N3 > N2 > N1 ≈ REM, Wake variable) :", ""]
    for k in (0, 1, 2, 3, 4):
        m = A["stades"] == k
        if m.any():
            L.append(f"- {NOMS[k]} : {np.median(A['eeg_std'][m]):.1f} µV")
    L += ["", "## Valeurs aberrantes", "",
          f"Seuils fixés avant de regarder : EEG plat < {EEG_STD_PLAT} µV, EEG extrême > {EEG_STD_EXTREME} µV, "
          f"EEG saturé (> 5 % des échantillons à ±{EEG_SAT_UV:.0f} µV, plage physique ±125 µV), ECG plat, ECG saturé (> 5 % à ±{ECG_SAT_MV} mV).", "",
          "| Critère | Époques | Part |", "|---|---|---|",
          f"| EEG plat | {(A['eeg_std'] < EEG_STD_PLAT).sum():,} | {(A['eeg_std'] < EEG_STD_PLAT).mean():.2%} |",
          f"| EEG extrême | {(A['eeg_std'] > EEG_STD_EXTREME).sum():,} | {(A['eeg_std'] > EEG_STD_EXTREME).mean():.2%} |",
          f"| EEG saturé | {(A['eeg_sat'] > 0.05).sum():,} | {(A['eeg_sat'] > 0.05).mean():.2%} |",
          f"| ECG plat | {(A['ecg_std'] < 1e-6).sum():,} | {(A['ecg_std'] < 1e-6).mean():.2%} |",
          f"| ECG saturé | {(A['ecg_sat'] > 0.05).sum():,} | {(A['ecg_sat'] > 0.05).mean():.2%} |",
          f"| **Au moins un critère** | {aberr.sum():,} | {aberr.mean():.2%} |", "",
          "Saturation EEG par stade (une saturation surtout en éveil = mouvements et yeux, pas des ondes lentes) :", "",
          *[f"- {NOMS[k]} : {(A['eeg_sat'][A['stades'] == k] > 0.05).mean():.1%} des époques" for k in (0, 1, 2, 3, 4)], "",
          f"Nuits avec plus de {PART_EPOQUES_ABERRANTES_NUIT:.0%} d'époques aberrantes : **{len(nuits_signalees)}** "
          f"(identifiants dans `eda_nuits.csv`, hors dépôt). Par ensemble : "
          + ", ".join(f"{e} {sum(l['ensemble'] == e for l in nuits_signalees)}" for e in ("train", "val", "test")) + ".", "",
          "## Ce qu'on en fait", "",
          "- La saturation EEG touche surtout l'éveil : ce sont des artefacts de mouvement et d'yeux, pas des ondes lentes. "
          "Une poignée de nuits saturées à plus de 50 % (gain mal réglé) concentre une bonne part des époques aberrantes.",
          "- Les époques aberrantes ne sont **pas** retirées des données : un modèle en production les verra. "
          "Elles sont marquées, et un futur module « qualité du signal » devra les refuser plutôt que les classer.",
          "- Pour les références de l'étape 3, on entraîne sur tout et on rapporte aussi le score sans les époques aberrantes, "
          "pour mesurer leur poids.",
          "- Figures : `data/figures/shhs_eda_amplitudes.png`, `data/figures/shhs_eda_stades.png`.", ""]
    RAPPORT.write_text("\n".join(L), encoding="utf-8")

    # ── Figures agrégées ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(np.log10(np.clip(A["eeg_std"], 0.01, None)), bins=80, color="#2E4057")
    for s in (EEG_STD_PLAT, EEG_STD_EXTREME):
        axes[0].axvline(np.log10(s), color="#E84855", ls="--")
    axes[0].set_title("EEG : écart-type par époque"); axes[0].set_xlabel("log10(µV)")
    axes[1].hist(np.log10(np.clip(A["ecg_std"], 1e-4, None)), bins=80, color="#2E4057")
    axes[1].set_title("ECG : écart-type par époque"); axes[1].set_xlabel("log10(mV)")
    axes[2].hist(A["eeg_max"], bins=80, color="#2E4057"); axes[2].axvline(EEG_SAT_UV, color="#E84855", ls="--")
    axes[2].set_title("EEG : maximum absolu par époque (butée ±125 µV)"); axes[2].set_xlabel("µV")
    for ax in axes:
        ax.set_yscale("log"); ax.grid(alpha=0.3)
    fig.suptitle(f"SHHS — {len(lignes)} nuits, {n_ep:,} époques (seuils d'aberration en rouge)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "shhs_eda_amplitudes.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    parts = np.array([[l[f"part_{NOMS[k]}"] for k in (0, 1, 2, 3, 4)] for l in lignes])
    axes[0].boxplot(parts, tick_labels=[NOMS[k] for k in (0, 1, 2, 3, 4)])
    axes[0].set_title("Part de chaque stade, par nuit"); axes[0].set_ylabel("part des époques"); axes[0].grid(alpha=0.3)
    axes[1].hist([l["part_apnee_sommeil"] for l in lignes], bins=40, color="#E84855")
    axes[1].set_title("Part d'époques « apnée » pendant le sommeil, par nuit"); axes[1].set_xlabel("part"); axes[1].grid(alpha=0.3)
    fig.suptitle("SHHS — distribution des étiquettes par nuit", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES / "shhs_eda_stades.png", dpi=130); plt.close(fig)

    print(open(RAPPORT, encoding="utf-8").read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
