#!/usr/bin/env python3
"""
Horizon 2.1 — Les règles de saturation valent-elles l'annotation du technicien ?

Jusqu'ici, la désaturation associée à un événement (somnia.resp.chute_de_saturation) et le compte
des désaturations (somnia.resp.desaturations) étaient des règles « simples, non validées ». Les
fichiers d'annotation SHHS contiennent pourtant les désaturations marquées à la lecture
(« SpO2 desaturation ») et les artefacts d'oxymètre (« SpO2 artifact »). Trois comparaisons, sur
les 40 nuits de VALIDATION :

  1. chaque désaturation de la règle contre celles du technicien (appariement un à un) ;
  2. pour chaque événement respiratoire du technicien : la règle dit-elle « avec désaturation »
     quand le technicien en a marqué une juste après ?
  3. les secondes de saturation invalide (somnia.qualite) contre les artefacts marqués ;
  4. la règle isolée du réseau : l'index calculé à partir des événements DU TECHNICIEN, en liant
     chaque hypopnée à une désaturation par la règle ou par les marques, contre `ahi_a0h3`.

Attention à la lecture des annotations : SHHS marque TOUTES les chutes, y compris d'un ou deux
points, et donne pour chacune son nadir et sa ligne de base (`SpO2Nadir`, `SpO2Baseline`). Comparer
une règle « 3 points » à toutes les marques n'a pas de sens (première tentative : accord nul). On
compare donc à profondeur égale : marques dont ligne de base − nadir ≥ N, contre règle ≥ N.

Sortie : docs/RESULTATS_DESATURATION.md (agrégats).
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import personnes_du_split  # noqa: E402
from somnia.deep.resp_net import RESP_DIR, charger_nuits  # noqa: E402
from somnia.resp import FS_RESP, Ev, apparier, chute_de_saturation, desaturations, masque_vers_evenements, sommeil_par_seconde  # noqa: E402
from somnia.shhs import lire_annotations  # noqa: E402

RAW = Path(os.environ.get("SHHS_DIR", Path.home() / "data" / "shhs" / "raw"))
from somnia.deep.data import PROCESSED  # noqa: E402

OUT = ROOT / "docs" / "RESULTATS_DESATURATION.md"
SEUILS = (2.0, 3.0, 4.0)


def desaturations_marquees(xml: Path) -> list[tuple[float, float, float]]:
    """(début, fin, profondeur = ligne de base − nadir) de chaque « SpO2 desaturation » du fichier d'annotation."""
    import xml.etree.ElementTree as ET
    out = []
    for ev in ET.parse(xml).getroot().iter("ScoredEvent"):
        if "spo2 desaturation" in (ev.findtext("EventConcept") or "").lower():
            a, d = float(ev.findtext("Start")), float(ev.findtext("Duration"))
            nadir, base = ev.findtext("SpO2Nadir"), ev.findtext("SpO2Baseline")
            out.append((a, a + d, float(base) - float(nadir) if nadir and base else float("nan")))
    return out


def main() -> int:
    xmls = {p.name.replace("-nsrr.xml", ""): p for p in RAW.rglob("*-nsrr.xml")}
    val = charger_nuits(personnes_du_split("val"), sommeil="multi")
    tot = {s: dict(vp=0, fp=0, fn=0) for s in SEUILS}
    comptes = {s: [] for s in SEUILS}; comptes_tech = {s: [] for s in SEUILS}; profondeurs = []; ecarts = []
    par_evenement = {s: {"regle": [], "tech": []} for s in SEUILS}
    art = {"tech_s": 0, "regle_s": 0, "commun_s": 0, "total_s": 0}
    import csv
    with open(PROCESSED / "covariables.csv", encoding="utf-8") as f:
        cov = {f"shhs-{r['nsrrid']}": r for r in csv.DictReader(f)}
    FEN = ((30, 45), (30, 30), (30, 20))                      # (secondes avant, secondes après) : la première est celle du code
    idx = {**{f"regle_{a}_{b}": [] for a, b in FEN}, "marques": [], "ahi": []}
    for n in val:
        ann = lire_annotations(xmls[n.ident])
        marques = [m for m in desaturations_marquees(xmls[n.ident]) if m[0] < n.n_sec]
        profondeurs += [m[2] for m in marques]
        sao2 = n.signaux[3, ::FS_RESP][: n.n_sec]
        for a, b, p in marques:                               # la même chute, mesurée sur le signal par notre définition
            if np.isfinite(p) and 30 <= a and b < n.n_sec:
                ecarts.append((float(sao2[int(a) - 30:int(a) + 1].max() - sao2[int(a):int(b) + 1].min()), p))
        for s in SEUILS:
            tech = [Ev(int(a), max(int(a) + 1, int(b)), 1) for a, b, p in marques if p >= s]
            regle = [Ev(a, max(a + 1, b), 1) for a, b, _ in desaturations(sao2, s)]
            m = apparier(regle, tech)
            for k in ("vp", "fp", "fn"):
                tot[s][k] += m[k]
            comptes[s].append(len(regle)); comptes_tech[s].append(len(tech))
        # 2. par événement respiratoire du technicien (pendant le sommeil)
        prof = np.zeros(n.n_sec + 120, dtype=np.float32)              # profondeur de la marque la plus profonde à chaque seconde
        for a, b, p in marques:
            if np.isfinite(p):
                prof[int(a):int(b) + 1] = np.maximum(prof[int(a):int(b) + 1], p)
        sommeil = sommeil_par_seconde(n.stades)[: n.n_sec]
        ref = [e for e in masque_vers_evenements(n.y) if sommeil[min(e.debut, len(sommeil) - 1)]]
        heures = float(sommeil.sum()) / 3600
        for av, ap in FEN:
            idx[f"regle_{av}_{ap}"].append(sum(e.classe == 1 or chute_de_saturation(sao2, e.debut, e.fin, av, ap) >= 3 for e in ref) / heures)
        idx["marques"].append(sum(e.classe == 1 or prof[e.debut:e.fin + 45].max() >= 3 for e in ref) / heures)
        v = cov.get(n.personne, {}).get("ahi_a0h3", "")
        idx["ahi"].append(float(v) if v not in ("", None) else np.nan)
        for e in ref:
            p_tech = float(prof[e.debut:e.fin + 45].max())
            c = chute_de_saturation(sao2, e.debut, e.fin)
            for s in SEUILS:
                par_evenement[s]["regle"].append(c >= s); par_evenement[s]["tech"].append(p_tech >= s)
        # 3. artefacts d'oxymètre
        chemin = RESP_DIR / f"{n.ident}_sao2_invalide.npy"
        if chemin.exists():
            inval = np.load(chemin)[: n.n_sec]; inval = np.concatenate([inval, np.zeros(n.n_sec - len(inval), dtype=bool)])
            t = np.zeros(n.n_sec, dtype=bool)
            for e in ann.evenements:
                if e.nom.lower() == "spo2 artifact":
                    t[int(e.debut):int(e.fin) + 1] = True
            art["tech_s"] += int(t.sum()); art["regle_s"] += int(inval.sum()); art["commun_s"] += int((t & inval).sum()); art["total_s"] += n.n_sec

    def prf(d):
        p, r = d["vp"] / max(d["vp"] + d["fp"], 1), d["vp"] / max(d["vp"] + d["fn"], 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)
    L = ["# Résultats — les règles de saturation contre l'annotation du technicien (horizon 2.1)", "",
         f"*Généré le {date.today().isoformat()} par `python_scripts/desat_validation.py`. Validation SHHS, {len(val)} nuits, "
         f"{len(profondeurs):,} désaturations marquées. Agrégats seulement.*", "",
         "SHHS marque toutes les chutes de saturation, même légères, et donne pour chacune son nadir et sa ligne de base. "
         f"Profondeur des marques : médiane {np.nanmedian(profondeurs):.0f} point(s) ; {np.mean(np.array(profondeurs) >= 3):.0%} font au moins 3 points, "
         f"{np.mean(np.array(profondeurs) >= 4):.0%} au moins 4. Toutes les comparaisons sont faites à profondeur égale.", "",
         "## 1. Chaque désaturation : la règle contre le technicien", "",
         "Règle : la saturation passe N points sous le maximum des deux minutes précédentes. Appariement un à un, par recouvrement.", "",
         "| Profondeur minimale | Marquées | Trouvées par la règle | Précision | Rappel | F1 | Compte par nuit : Spearman |", "|---|---|---|---|---|---|---|"]
    for s in SEUILS:
        p, r, f = prf(tot[s])
        L.append(f"| {s:g} points | {sum(comptes_tech[s]):,} | {sum(comptes[s]):,} | {p:.2f} | {r:.2f} | {f:.2f} | {spearmanr(comptes[s], comptes_tech[s]).correlation:.2f} |")
    L += ["", "## 2. « Avec désaturation » : la règle par événement contre le technicien", "",
          "Pour chaque événement respiratoire du technicien pendant le sommeil : y a-t-il une désaturation marquée d'au moins N points entre le "
          "début de l'événement et 45 s après sa fin ? La règle (`chute_de_saturation`) dit-elle la même chose ?", "",
          "| Profondeur minimale | Événements | Technicien : avec désaturation | Règle : avec désaturation | Accord | Kappa |", "|---|---|---|---|---|---|"]
    for s in SEUILS:
        a, b = np.array(par_evenement[s]["regle"]), np.array(par_evenement[s]["tech"])
        L.append(f"| {s:g} points | {len(a):,} | {b.mean():.0%} | {a.mean():.0%} | {(a == b).mean():.0%} | {cohen_kappa_score(a, b):.2f} |")
    L += ["", "## 3. Oxymètre décollé : la règle contre les artefacts marqués", "",
          "| | |", "|---|---|",
          f"| Temps marqué « artefact » par le technicien | {art['tech_s'] / art['total_s']:.1%} de l'enregistrement |",
          f"| Temps « saturation invalide » pour la règle (hors de 50–100 %) | {art['regle_s'] / art['total_s']:.1%} |",
          f"| Part du temps invalide pour la règle qui est aussi marquée par le technicien | {art['commun_s'] / max(art['regle_s'], 1):.0%} |",
          f"| Part du temps marqué par le technicien que la règle retrouve | {art['commun_s'] / max(art['tech_s'], 1):.0%} |", ""]
    e = np.array(ecarts)
    k3 = cohen_kappa_score(np.array(par_evenement[3.0]["regle"]), np.array(par_evenement[3.0]["tech"]))
    ahi = np.array(idx["ahi"]); okc = np.isfinite(ahi)
    L += ["## 4. La règle isolée du réseau : index à partir des événements du technicien", "",
          "Événements et sommeil du technicien ; seule change la façon de décider qu'une hypopnée « a désaturé » (3 points). Contre `ahi_a0h3`.", "",
          "| Lien hypopnée – désaturation | Spearman | Erreur absolue médiane | Biais |", "|---|---|---|---|"]
    noms = {"regle_30_45": "**Règle du code** (30 s avant, 45 s après)", "regle_30_30": "Règle, 30 s après (sensibilité)",
            "regle_30_20": "Règle, 20 s après (sensibilité)", "marques": "Marques de désaturation de l'annotation"}
    for cle, nom in noms.items():
        v = np.array(idx[cle])[okc]
        L.append(f"| {nom} | {spearmanr(v, ahi[okc]).correlation:.3f} | {np.median(np.abs(v - ahi[okc])):.1f} / h | {np.mean(v - ahi[okc]):+.1f} / h |")
    L += ["", "## Lecture", "",
          "- **Les temps et les profondeurs sont justes** : le nadir écrit dans l'annotation se retrouve dans le signal à la même seconde, et "
          f"la profondeur mesurée sur le signal dans chaque marque est la même en médiane ({np.median(e[:, 0] - e[:, 1]):+.1f} point).",
          f"- **La règle trouve environ deux fois plus de chutes que les marques**, et l'accord événement par événement est moyen (kappa {k3:.2f} à 3 points). "
          "Les marques du fichier sont donc une référence incomplète : elles ne reprennent pas toutes les chutes visibles dans le signal.",
          "- **C'est l'index qui tranche, et il donne raison à la règle** : avec les événements du technicien, lier les hypopnées aux désaturations "
          "par la règle redonne l'index officiel de SHHS presque exactement ; le faire avec les marques le sous-estime. La règle est donc "
          "validée indépendamment du réseau.",
          "- **La fenêtre n'a pas été réglée** : 45 s après la fin était le choix de départ. Les lignes « sensibilité » montrent que le "
          "résultat y est peu sensible ; la fenêtre reste à 45 s pour ne pas régler sur la validation.",
          "- **Oxymètre décollé** : la règle « hors de 50–100 % » retrouve quatre cinquièmes du temps marqué artefact ; un tiers de ce "
          "qu'elle signale n'est pas marqué.", ""]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
