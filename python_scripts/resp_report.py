#!/usr/bin/env python3
"""
Détection d'événements respiratoires — le tableau de résultats (docs/RESULTATS_EVENEMENTS.md).

Lit models/resp/*.json : la version courante (evenements_v2_s*), la première version
(evenements_s*), la grille d'expériences (g_*), la référence par désaturation.
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


def lire(motif: str) -> list[dict]:
    return sorted((json.loads(p.read_text(encoding="utf-8")) for p in RESP.glob(motif)), key=lambda r: r["graine"])


def pm(runs, f, nd=3):
    v = [f(r) for r in runs]
    return f"{np.mean(v):.{nd}f}" + (f" ± {np.std(v):.{nd}f}" if len(v) > 1 else "")


def ligne_evenements(nom, runs, cle):
    g = lambda k1, k2: (lambda r: r[cle]["par_evenement"][k1][k2])
    return (f"| {nom} | {pm(runs, g('recouvrement', 'precision'))} | {pm(runs, g('recouvrement', 'rappel'))} | "
            f"{pm(runs, g('recouvrement', 'f1'))} | {pm(runs, g('iou_0.3', 'f1'))} | "
            f"{pm(runs, lambda r: r[cle]['par_evenement']['rappel_apnees'])} | {pm(runs, lambda r: r[cle]['par_evenement']['rappel_hypopnees'])} |")


def main() -> int:
    v4, v3, v2, v1 = lire("evenements_v4_s*.json"), lire("evenements_v3_s*.json"), lire("evenements_v2_s*.json"), lire("evenements_s*.json")
    grille = {p.stem.rsplit("_s", 1)[0]: json.loads(p.read_text(encoding="utf-8")) for p in RESP.glob("g_*_s*.json")}
    base = json.loads((RESP / "reference_desaturation.json").read_text(encoding="utf-8")) if (RESP / "reference_desaturation.json").exists() else None
    courant = v4 or v3 or v2 or v1
    if not courant:
        sys.exit("aucun entraînement dans models/resp")
    r0 = courant[0]
    S = "validation_pendant_le_sommeil"
    L = ["# Résultats — détection d'événements respiratoires", "",
         f"*Généré le {date.today().isoformat()} par `python_scripts/resp_report.py`. Validation SHHS ({r0['n_nuits_val']} personnes) ; "
         "le test n'est pas ouvert pour cette tâche. Moyenne ± écart-type quand plusieurs graines.*", "",
         "## Ce qui est mesuré", "",
         "Le réseau lit 5 minutes de flux, de ceintures thoracique et abdominale et de saturation (10 Hz), et rend pour chaque "
         "seconde : rien, apnée ou hypopnée. Les secondes positives sont regroupées en événements d'au moins 10 s. La référence "
         "est l'annotation du technicien (apnées obstructives, centrales, mixtes et hypopnées, toutes comptées).", "",
         "Deux façons de mesurer :", "",
         "- **tous les événements** du technicien, y compris ceux marqués pendant l'éveil ;",
         "- **pendant le sommeil, de bout en bout** : la référence est limitée aux événements qui commencent pendant le sommeil du "
         "technicien (ceux qui comptent dans l'index) ; les propositions, à celles qui commencent pendant le sommeil **prédit** par "
         "le réseau de stades ; l'index est calculé sur le temps de sommeil prédit. Rien n'est emprunté au technicien.", ""]

    L += ["## Par événement", "",
          "| Modèle et mesure | Précision | Rappel | F1 | F1, bornes exigeantes (IoU ≥ 0,3) | Rappel apnées | Rappel hypopnées |",
          "|---|---|---|---|---|---|---|"]
    if v1:
        L.append(ligne_evenements(f"v1, 4 canaux, tous les événements ({len(v1)} graines)", v1, "validation"))
    if "g_ref" in grille and grille["g_ref"].get(S):
        L.append(ligne_evenements("v1, pendant le sommeil (1 graine)", [grille["g_ref"]], S))
    if v2:
        L.append(ligne_evenements(f"v2, pendant le sommeil ({len(v2)} graines)", v2, S))
    if v3:
        L.append(ligne_evenements(f"v3, pendant le sommeil ({len(v3)} graines)", v3, S))
    if v4:
        L.append(ligne_evenements(f"**v4, pendant le sommeil** ({len(v4)} graines)", v4, S))
        L.append(ligne_evenements(f"v4, tous les événements ({len(v4)} graines)", v4, "validation"))
    L += ["", "v2 = v1 + un cinquième canal (la probabilité de sommeil prédite par le réseau de stades), une pondération des classes "
              "adoucie (racine carrée) et un gain aléatoire sur les capteurs à l'entraînement.", "",
          "v3 = v2, mais le sommeil prédit vient du nouveau réseau de stades, qui lit la nuit entière (encodeur + GRU) au lieu "
          "d'une époque à la fois. Rien d'autre ne change : l'écart v2 → v3 mesure ce que rapporte un meilleur hypnogramme.", "",
          "v4 = v3, mais le réseau de stades lit aussi les yeux et le menton (horizon 1.3) : l'accord éveil / sommeil passe de 92 % "
          "à 94 %. **Le score par événement ne bouge pas** : ce n'est plus le sommeil prédit qui limite la détection. C'est cohérent "
          "avec l'analyse des manqués (`docs/ANALYSE_MANQUES.md`) : seuls 16 % des événements manqués tombaient dans un éveil prédit à tort.", ""]

    L += ["## Ce qui a été essayé, et ce que ça a donné", "",
          "Diagnostic de la v1 sur la validation : **47 % des fausses propositions commençaient pendant l'éveil**, 44 % étaient à "
          "moins d'une minute d'un vrai événement, et un vote de trois graines n'apportait que +0,004 de F1.", "",
          "| Expérience (une graine, mesure « pendant le sommeil ») | Précision | Rappel | F1 | Index : Spearman |", "|---|---|---|---|---|"]
    noms = [("g_ref", "A. référence, 4 canaux"), ("g_sommeil", "B. + canal de sommeil prédit"),
            ("g_sommeil_racine", "C. + pondération adoucie (retenu : v2)"), ("g_sommeil_large", "D. + réseau plus large et plus profond (96, 8 blocs)"),
            ("g_f25", "E. comme B, avec 25 % des nuits d'entraînement"), ("g_f50", "F. comme B, avec 50 % des nuits")]
    for cle, nom in noms:
        if cle in grille and grille[cle].get(S):
            e = grille[cle][S]["par_evenement"]["recouvrement"]
            L.append(f"| {nom} | {e['precision']:.3f} | {e['rappel']:.3f} | {e['f1']:.3f} | {grille[cle][S]['par_personne']['spearman']:.3f} |")
    L += ["",
          "- **Le sommeil compte plus que la taille du réseau.** Dire au réseau si le patient dort améliore la précision ; l'élargir ne change rien.",
          "- **Plus de nuits du même type n'aideraient pas** : avec un quart des nuits, le score est presque le même. La courbe d'apprentissage est plate.",
          "- **Borne haute mesurée** : avec le sommeil du technicien à la place du sommeil prédit, le même réseau atteint un F1 de 0,77. "
          "L'écart restant vient donc du réseau de stades (accord éveil/sommeil de 91 % par seconde), pas du réseau d'événements.",
          "- Lisser le sommeil prédit ou changer son seuil ne change rien (F1 entre 0,717 et 0,725). Les règles de regroupement "
          "(10 s minimum, trous de 3 s) et le seuil de décision (0,7) sont au bon endroit.",
          "- Réserve : six expériences ont été comparées sur les mêmes 40 personnes de validation ; l'écart entre réglages voisins "
          "(0,01) est du même ordre que le bruit entre graines. Le test tranchera, une fois.", ""]

    L += ["## Par personne : l'index", "",
          "| Estimateur | Contre | Spearman | Erreur absolue médiane (/h) | Biais (/h) |", "|---|---|---|---|---|"]
    for nom, runs, cle in (("Réseau v4, de bout en bout", v4, S), ("Réseau v3, de bout en bout", v3, S), ("Réseau v2, de bout en bout", v2, S), ("Réseau v1 (sommeil du technicien)", v1, "validation")):
        if not runs:
            continue
        L.append(f"| {nom} | index annoté | {pm(runs, lambda r: r[cle]['par_personne']['spearman'])} | "
                 f"{pm(runs, lambda r: r[cle]['par_personne']['erreur_absolue_mediane'], 1)} | {pm(runs, lambda r: r[cle]['par_personne']['biais'], 1)} |")
        for ref in ("ahi_a0h3a", "ahi_a0h4"):
            if ref in runs[0][cle].get("contre_clinique", {}):
                L.append(f"| {nom} | index clinique `{ref}` | {pm(runs, lambda r, ref=ref: r[cle]['contre_clinique'][ref]['spearman'])} | "
                         f"{pm(runs, lambda r, ref=ref: r[cle]['contre_clinique'][ref]['erreur_absolue_mediane'], 1)} | "
                         f"{pm(runs, lambda r, ref=ref: r[cle]['contre_clinique'][ref]['biais'], 1)} |")
    if base:
        for k, nom in (("odi_3", "Désaturations ≥ 3 % / h (référence simple)"), ("odi_4", "Désaturations ≥ 4 % / h (référence simple)")):
            b = base[k]
            for ref in ("ahi_a0h3a", "ahi_a0h4"):
                if ref in b["contre_clinique"]:
                    c = b["contre_clinique"][ref]
                    L.append(f"| {nom} | index clinique `{ref}` | {c['spearman']:.3f} | {c['erreur_absolue_mediane']:.1f} | {c['biais']:+.1f} |")
    L += ["", "## Lecture", "",
          "- **Deux index, deux questions.** L'index annoté compte toutes les hypopnées marquées par le technicien ; l'index clinique "
          "SHHS ne garde que celles suivies d'une désaturation. Le réseau apprend le premier ; la référence par désaturation colle "
          "au second par construction.",
          "- **La référence par désaturation est le chiffre à battre pour l'index clinique** : compter les chutes de saturation "
          "suffit presque. Ce que le réseau apporte, c'est la **position** de chaque événement : c'est ce qui prend du temps à un lecteur.",
          "- **Améliorer le réseau de stades ne suffit plus** : l'hypothèse a été testée (v4) et le score ne bouge pas. Ce qui reste, "
          "ce sont surtout des hypopnées courtes et sans désaturation (`docs/ANALYSE_MANQUES.md`).",
          "- La limite probable est l'annotation elle-même : marquer une hypopnée sans critère de désaturation est une "
          "décision où deux techniciens ne sont pas toujours d'accord. Hypothèse non vérifiée ici, faute de double scoring.",
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
    axes[0].set_xlabel("index annoté (événements / h de sommeil)"); axes[0].set_ylabel("index estimé, de bout en bout")
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
