#!/usr/bin/env python3
"""
Détection d'événements respiratoires — le tableau de résultats (docs/RESULTATS_EVENEMENTS.md).

Lit models/resp/reference_desaturation.json et models/resp/evenements_s*.json.
Validation SHHS seulement (40 personnes) : le test n'est pas ouvert pour cette tâche.
Usage : python python_scripts/resp_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESP = ROOT / "models" / "resp"
OUT = ROOT / "docs" / "RESULTATS_EVENEMENTS.md"
FIG = ROOT / "data" / "figures"


def main() -> int:
    runs = sorted((json.loads(p.read_text(encoding="utf-8")) for p in RESP.glob("evenements_s*.json")), key=lambda r: r["graine"])
    base = json.loads((RESP / "reference_desaturation.json").read_text(encoding="utf-8")) if (RESP / "reference_desaturation.json").exists() else None
    if not runs:
        sys.exit("aucun entraînement dans models/resp")

    def pm(f):
        v = [f(r) for r in runs]
        return f"{np.mean(v):.3f}" + (f" ± {np.std(v):.3f}" if len(v) > 1 else "")

    def pm1(f):
        v = [f(r) for r in runs]
        return f"{np.mean(v):.1f}" + (f" ± {np.std(v):.1f}" if len(v) > 1 else "")
    r0 = runs[0]; v = r0["validation"]
    L = ["# Résultats — détection d'événements respiratoires", "",
         f"*Généré le {date.today().isoformat()} par `python_scripts/resp_report.py`. Validation SHHS ({r0['n_nuits_val']} personnes) ; "
         f"le test n'est pas ouvert pour cette tâche. {len(runs)} graine(s).*", "",
         "## Ce qui est mesuré", "",
         "Le réseau lit 5 minutes de flux, de ceintures thoracique et abdominale et de saturation (10 Hz), et rend pour chaque "
         "seconde : rien, apnée ou hypopnée. Les secondes positives sont regroupées en événements d'au moins 10 s. La référence "
         "est l'annotation du technicien (apnées obstructives, centrales, mixtes et hypopnées, toutes comptées).", "",
         f"Entraînement : {r0['n_nuits_train']} nuits. Réseau : {r0['parametres']:,} paramètres, meilleure époque {r0['meilleure_epoque']}, "
         f"{r0['duree_min']} min sur {r0['appareil']}.", "",
         "## Par événement", "",
         "| Critère d'appariement | Précision | Rappel | F1 |", "|---|---|---|---|",
         f"| Tout recouvrement | {pm(lambda r: r['validation']['par_evenement']['recouvrement']['precision'])} | "
         f"{pm(lambda r: r['validation']['par_evenement']['recouvrement']['rappel'])} | {pm(lambda r: r['validation']['par_evenement']['recouvrement']['f1'])} |",
         f"| Recouvrement IoU ≥ 0,3 (bornes exigeantes) | {pm(lambda r: r['validation']['par_evenement']['iou_0.3']['precision'])} | "
         f"{pm(lambda r: r['validation']['par_evenement']['iou_0.3']['rappel'])} | {pm(lambda r: r['validation']['par_evenement']['iou_0.3']['f1'])} |", "",
         f"Rappel par type : apnées {pm(lambda r: r['validation']['par_evenement']['rappel_apnees'])} "
         f"({v['par_evenement']['n_apnees_ref']:,} dans la validation), hypopnées {pm(lambda r: r['validation']['par_evenement']['rappel_hypopnees'])} "
         f"({v['par_evenement']['n_hypopnees_ref']:,}).", "",
         f"Par seconde : F1 « événement en cours » {pm(lambda r: r['validation']['par_seconde']['f1_evenement'])} ; "
         f"{v['par_seconde']['part_reference']:.1%} des secondes sont en événement dans la référence, {v['par_seconde']['part_predite']:.1%} dans la prédiction.", "",
         "## Par personne : l'index", "",
         "| Estimateur | Contre | Spearman | Erreur absolue médiane (/h) | Biais (/h) |", "|---|---|---|---|---|",
         f"| Réseau (événements détectés / h de sommeil) | index annoté | {pm(lambda r: r['validation']['par_personne']['spearman'])} | "
         f"{pm1(lambda r: r['validation']['par_personne']['erreur_absolue_mediane'])} | {pm1(lambda r: r['validation']['par_personne']['biais'])} |"]
    for ref in ("ahi_a0h3a", "ahi_a0h4"):
        if ref in v.get("contre_clinique", {}):
            L.append(f"| Réseau | index clinique `{ref}` | {pm(lambda r, ref=ref: r['validation']['contre_clinique'][ref]['spearman'])} | "
                     f"{pm1(lambda r, ref=ref: r['validation']['contre_clinique'][ref]['erreur_absolue_mediane'])} | {pm1(lambda r, ref=ref: r['validation']['contre_clinique'][ref]['biais'])} |")
    if base:
        for k, nom in (("odi_3", "Désaturations ≥ 3 % / h (référence simple)"), ("odi_4", "Désaturations ≥ 4 % / h (référence simple)")):
            b = base[k]
            L.append(f"| {nom} | index annoté | {b['contre_annote']['spearman']:.3f} | {b['contre_annote']['erreur_absolue_mediane']:.1f} | {b['contre_annote']['biais']:+.1f} |")
            for ref in ("ahi_a0h3a", "ahi_a0h4"):
                if ref in b["contre_clinique"]:
                    c = b["contre_clinique"][ref]
                    L.append(f"| {nom} | index clinique `{ref}` | {c['spearman']:.3f} | {c['erreur_absolue_mediane']:.1f} | {c['biais']:+.1f} |")
    lim = v["par_personne"]["limites_accord_95"]
    L += ["", f"Accord réseau / index annoté (Bland-Altman, graine {r0['graine']}) : biais {v['par_personne']['biais']:+.1f} événements par heure, "
              f"limites d'accord à 95 % de {lim[0]:+.1f} à {lim[1]:+.1f}.", "",
          "## Lecture", "",
          "- **Deux index, deux questions.** L'index annoté compte toutes les hypopnées marquées par le technicien ; l'index clinique "
          "SHHS ne garde que celles suivies d'une désaturation. Le réseau apprend le premier ; la référence par désaturation colle "
          "au second par construction.",
          "- **La référence par désaturation est le chiffre à battre pour l'index clinique** : compter les chutes de saturation "
          "suffit presque à retrouver l'index clinique à 4 %. Ce que le réseau apporte en plus, c'est la **position** de chaque "
          "événement : c'est ce qui prend du temps à un lecteur.",
          "- **Le F1 par événement est la mesure du temps gagné** : un événement bien placé est un événement à valider d'un clic "
          "au lieu de le chercher et de le marquer. Le rappel par type dit ce que le lecteur devra encore trouver seul.",
          "- Figure : `data/figures/evenements_index.png`.", ""]
    OUT.write_text("\n".join(L), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(RESP / f"{r0['nom']}_val_detail.npz")
    ref, est = d["index_reference"], d["index_estime"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    m = max(ref.max(), est.max()) * 1.05
    axes[0].scatter(ref, est, color="#2E4057", s=22); axes[0].plot([0, m], [0, m], "k--", lw=1)
    axes[0].set_xlabel("index annoté (événements / h de sommeil)"); axes[0].set_ylabel("index estimé par le réseau")
    axes[0].set_title("Une personne par point (validation)"); axes[0].grid(alpha=0.3)
    moy, diff = (ref + est) / 2, est - ref
    axes[1].scatter(moy, diff, color="#2E4057", s=22)
    for yv, ls in ((diff.mean(), "-"), (diff.mean() - 1.96 * diff.std(), "--"), (diff.mean() + 1.96 * diff.std(), "--")):
        axes[1].axhline(yv, color="#E84855", ls=ls, lw=1)
    axes[1].set_xlabel("moyenne des deux index"); axes[1].set_ylabel("estimé − annoté"); axes[1].set_title("Bland-Altman"); axes[1].grid(alpha=0.3)
    fig.suptitle("Détection d'événements respiratoires : index par personne", fontweight="bold")
    fig.tight_layout(); FIG.mkdir(exist_ok=True); fig.savefig(FIG / "evenements_index.png", dpi=130); plt.close(fig)
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
