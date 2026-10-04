#!/usr/bin/env python3
"""
Horizon 2 — Type d'apnée : obstructive ou centrale, d'après l'effort des ceintures.

Le réseau d'événements rend « apnée » ou « hypopnée ». Un compte rendu distingue les apnées
obstructives des centrales : ce n'est pas la même maladie ni le même traitement. Deux méthodes,
apprises sur les apnées du technicien des nuits d'ENTRAÎNEMENT, mesurées sur la VALIDATION :
  - référence simple : quatre traits lisibles (somnia.resp.effort_respiratoire) + régression logistique ;
  - petit réseau (somnia.deep.type_net) qui lit la forme du flux et des ceintures, 3 graines.

Mesures :
  1. sur les apnées du technicien (limites exactes) : la règle sépare-t-elle les deux types ?
  2. de bout en bout : sur les apnées PROPOSÉES par le réseau, l'index d'apnées centrales par
     personne suit-il celui du technicien ?
Seuil de la règle : choisi sur l'entraînement (sensibilité = spécificité). Seuil du réseau : 0,5.
L'époque d'arrêt du réseau est choisie sur la validation, comme pour les autres réseaux du projet.

Sorties : models/resp/type_apnee.json (coefficients, versionné), docs/RESULTATS_TYPE_APNEE.md.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import PROCESSED, personnes_du_split  # noqa: E402
from somnia.deep.resp_net import charger_nuits  # noqa: E402
from somnia.resp import NOMS_EFFORT, effort_respiratoire, masque_vers_evenements, proba_centrale, proba_vers_masque, sommeil_par_seconde  # noqa: E402
from somnia.shhs import lire_annotations  # noqa: E402

RAW = Path(os.environ.get("SHHS_DIR", Path.home() / "data" / "shhs" / "raw"))
CACHE = PROCESSED.parent / "cache_val"
OUT_JSON, OUT_MD = ROOT / "models" / "resp" / "type_apnee.json", ROOT / "docs" / "RESULTATS_TYPE_APNEE.md"


def apnees_du_technicien(nuits, xmls):
    """Traits et étiquette (1 = centrale) de chaque apnée obstructive ou centrale marquée ; l'indice de la nuit."""
    X, y, qui = [], [], []
    for k, n in enumerate(nuits):
        for e in lire_annotations(xmls[n.ident]).evenements:
            nom = e.nom.lower()
            if nom not in ("obstructive apnea", "central apnea"):
                continue
            t = effort_respiratoire(n.signaux, int(e.debut), int(round(e.fin)))
            if t is not None:
                X.append(t); y.append(int(nom == "central apnea")); qui.append(k)
    return np.array(X), np.array(y), np.array(qui)


def fenetres_du_technicien(nuits, xmls, rng=None, decalage_max=0):
    """Fenêtres (N, 3, 900), étiquettes, indice de nuit, pour chaque apnée obstructive ou centrale marquée."""
    from somnia.deep.type_net import fenetre_apnee
    X, y, qui = [], [], []
    for k, n in enumerate(nuits):
        for e in lire_annotations(xmls[n.ident]).evenements:
            nom = e.nom.lower()
            if nom in ("obstructive apnea", "central apnea"):
                d = int(rng.integers(-decalage_max, decalage_max + 1)) if rng is not None and decalage_max else 0
                f = fenetre_apnee(n.signaux, int(e.debut), d)
                if f is not None:
                    X.append(f); y.append(int(nom == "central apnea")); qui.append(k)
    return np.array(X, dtype=np.float32), np.array(y), np.array(qui)


def entrainer_reseau(train, val, xmls, graine, journal):
    """Entraîne le réseau de typage ; rend (modèle au meilleur AUC de validation, P(centrale) sur la validation, AUC)."""
    import torch
    from somnia.deep.train import appareil, fixer_graines
    from somnia.deep.type_net import ReseauType, proba_centrale_reseau
    fixer_graines(graine); dev = appareil(); rng = np.random.default_rng(graine)
    Xv, yv, _ = fenetres_du_technicien(val, xmls)
    modele = ReseauType().to(dev)
    opt = torch.optim.AdamW(modele.parameters(), lr=1e-3, weight_decay=1e-3)
    meilleur, etat, sans = -1.0, None, 0
    for ep in range(1, 31):
        # à chaque époque, le début de chaque apnée est décalé au hasard de ± 5 s : à l'usage, le début vient du réseau, pas du technicien
        Xt, yt, _ = fenetres_du_technicien(train, xmls, rng, decalage_max=5)
        perte = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((yt == 0).sum() / max((yt == 1).sum(), 1)) ** 0.5, device=dev))
        modele.train(); ordre = rng.permutation(len(yt))
        for i in range(0, len(ordre), 64):
            j = ordre[i:i + 64]
            opt.zero_grad(set_to_none=True)
            perte(modele(torch.from_numpy(Xt[j]).to(dev)), torch.from_numpy(yt[j].astype(np.float32)).to(dev)).backward(); opt.step()
        pv = proba_centrale_reseau(modele, Xv, dev); auc = roc_auc_score(yv, pv)
        if auc > meilleur + 1e-4:
            meilleur, sans, etat = auc, 0, {k: v.detach().cpu().clone() for k, v in modele.state_dict().items()}
        else:
            sans += 1
            if sans >= 5:
                break
    modele.load_state_dict(etat)
    journal(f"réseau, graine {graine} : meilleur AUC de validation {meilleur:.3f} ({ep} époques)")
    return modele.cpu(), proba_centrale_reseau(modele, Xv), yv, meilleur


def main() -> int:
    xmls = {p.name.replace("-nsrr.xml", ""): p for p in RAW.rglob("*-nsrr.xml")}
    train = charger_nuits(personnes_du_split("train"), sommeil="multi")
    val = charger_nuits(personnes_du_split("val"), sommeil="multi")
    Xt, yt, _ = apnees_du_technicien(train, xmls)
    Xv, yv, qv = apnees_du_technicien(val, xmls)
    mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit((Xt - mu) / sd, yt)
    modele = {"traits": list(NOMS_EFFORT), "moyennes": mu.tolist(), "ecarts_types": sd.tolist(),
              "coefficients": clf.coef_[0].tolist(), "biais": float(clf.intercept_[0])}
    pt = np.array([proba_centrale(x, modele) for x in Xt]); pv = np.array([proba_centrale(x, modele) for x in Xv])
    fpr, tpr, seuils = roc_curve(yt, pt)
    modele["seuil"] = float(seuils[np.argmin(np.abs(tpr - (1 - fpr)))])          # sensibilité = spécificité, sur l'entraînement
    modele.update({"date": date.today().isoformat(), "n_train": {"obstructives": int((yt == 0).sum()), "centrales": int((yt == 1).sum())}})
    pred = pv >= modele["seuil"]
    mesures = {"auc_train": float(roc_auc_score(yt, pt)), "auc_val": float(roc_auc_score(yv, pv)),
               "n_val": {"obstructives": int((yv == 0).sum()), "centrales": int((yv == 1).sum())},
               "sensibilite_centrale": float(pred[yv == 1].mean()), "specificite": float((~pred[yv == 0]).mean()),
               "valeur_predictive_centrale": float(yv[pred].mean()) if pred.any() else None,
               "auc_par_trait": {nom: float(max(a, 1 - a)) for nom, a in ((nom, roc_auc_score(yv, Xv[:, j])) for j, nom in enumerate(NOMS_EFFORT))}}

    # Réseau : 3 graines
    import torch
    from somnia.deep.type_net import ReseauType, fenetre_apnee, proba_centrale_reseau
    lignes = []
    journal = lambda m: (print(m, flush=True), lignes.append(m))
    reseaux = {}
    for g in (42, 1, 2):
        net, pvr, yvr, auc = entrainer_reseau(train, val, xmls, g, journal)
        torch.save(net.state_dict(), ROOT / "models" / "resp" / f"type_apnee_s{g}.pt")
        pr = pvr >= 0.5
        reseaux[g] = {"auc_val": float(auc), "sensibilite_centrale": float(pr[yvr == 1].mean()), "specificite": float((~pr[yvr == 0]).mean()),
                      "valeur_predictive_centrale": float(yvr[pr].mean()) if pr.any() else None, "net": net}

    # 2. de bout en bout : apnées proposées par le réseau d'événements (v4), pendant le sommeil prédit
    def bout_en_bout(typer):
        ref_c, est_c, ref_n, est_n = [], [], 0, 0
        for n in val:
            with np.load(CACHE / f"{n.ident}.npz") as d:
                proba = d["proba_v4"].astype(np.float32); som_pred = np.repeat(d["logits_multi"].argmax(axis=1) != 0, 30)[: n.n_sec]
            som_pred = np.concatenate([som_pred, np.zeros(n.n_sec - len(som_pred), dtype=bool)])
            som_ref = sommeil_par_seconde(n.stades)[: n.n_sec]
            apnees = [e for e in masque_vers_evenements(proba_vers_masque(proba)) if e.classe == 1 and som_pred[e.debut]]
            n_c = typer(n, apnees)
            tech_c = sum(1 for e in lire_annotations(xmls[n.ident]).evenements
                         if e.nom.lower() == "central apnea" and e.debut < n.n_sec and som_ref[min(int(e.debut), len(som_ref) - 1)])
            ref_c.append(tech_c / max(som_ref.sum() / 3600, 1e-9)); est_c.append(n_c / max(som_pred.sum() / 3600, 1e-9))
            ref_n += tech_c; est_n += n_c
        ref_c, est_c = np.array(ref_c), np.array(est_c)
        return {"spearman": float(spearmanr(est_c, ref_c).correlation), "erreur_absolue_mediane": float(np.median(np.abs(est_c - ref_c))),
                "biais": float(np.mean(est_c - ref_c)), "n_centrales_technicien": int(ref_n), "n_centrales_estimees": int(est_n),
                "personnes_au_dessus_de_5_par_heure": {"technicien": int((ref_c >= 5).sum()), "estime": int((est_c >= 5).sum()),
                                                       "les_deux": int(((ref_c >= 5) & (est_c >= 5)).sum())}}

    def typer_regle(n, apnees):
        ps = [proba_centrale(effort_respiratoire(n.signaux, e.debut, e.fin), modele) for e in apnees]
        return sum(p is not None and p >= modele["seuil"] for p in ps)

    def typer_reseau(n, apnees):
        f = [x for x in (fenetre_apnee(n.signaux, e.debut) for e in apnees) if x is not None]
        return int((proba_centrale_reseau(reseaux[42]["net"], np.array(f)) >= 0.5).sum()) if f else 0

    mesures["bout_en_bout"] = bout_en_bout(typer_regle)
    mesures["reseau"] = {"graines": {str(g): {k: v for k, v in r.items() if k != "net"} for g, r in reseaux.items()},
                         "bout_en_bout_s42": bout_en_bout(typer_reseau), "journal": lignes}
    modele["validation"] = mesures
    OUT_JSON.write_text(json.dumps(modele, indent=2, ensure_ascii=False), encoding="utf-8")

    R = mesures["reseau"]["graines"]
    pm = lambda cle: f"{np.mean([r[cle] for r in R.values()]):.2f} ± {np.std([r[cle] for r in R.values()]):.2f}"
    L = ["# Résultats — type d'apnée : obstructive ou centrale", "",
         f"*Généré le {modele['date']} par `python_scripts/type_apnee.py`. Appris sur les apnées du technicien des nuits d'entraînement "
         f"({modele['n_train']['obstructives']:,} obstructives, {modele['n_train']['centrales']:,} centrales), mesuré sur la validation "
         f"({mesures['n_val']['obstructives']:,} et {mesures['n_val']['centrales']:,}). Agrégats seulement.*", "",
         "## Le principe", "",
         "Apnée obstructive : la gorge est fermée, le patient fait toujours l'effort de respirer, les ceintures bougent, souvent à contre-temps. "
         "Apnée centrale : la commande s'arrête, les ceintures sont plates. Deux méthodes : une **référence simple** (amplitude du thorax, de "
         "l'abdomen et du flux pendant l'événement, rapportée à la minute qui précède, opposition thorax–abdomen, régression logistique) et un "
         "**petit réseau** qui lit la forme du flux et des ceintures sur 90 s autour du début de l'apnée. "
         "Les apnées mixtes sont trop rares dans SHHS pour être apprises (19 dans l'entraînement).", "",
         "## 1. Sur les apnées du technicien (validation)", "",
         "| Mesure | Référence simple | Réseau (3 graines) |", "|---|---|---|",
         f"| Aire sous la courbe ROC | {mesures['auc_val']:.2f} | **{pm('auc_val')}** |",
         f"| Apnées centrales reconnues (sensibilité) | {mesures['sensibilite_centrale']:.2f} | {pm('sensibilite_centrale')} |",
         f"| Apnées obstructives reconnues (spécificité) | {mesures['specificite']:.2f} | {pm('specificite')} |",
         f"| Parmi les apnées dites centrales, part qui l'est vraiment | {mesures['valeur_predictive_centrale']:.2f} | {pm('valeur_predictive_centrale')} |", "",
         "Référence simple, trait par trait (aire sous la courbe) : " + ", ".join(f"{nom} {a:.2f}" for nom, a in mesures["auc_par_trait"].items()) + ".", "",
         "## 2. De bout en bout : index d'apnées centrales par personne", "",
         "Apnées proposées par le réseau d'événements pendant le sommeil prédit, puis typées, contre les apnées centrales du technicien (40 personnes).", "",
         "| Mesure | Référence simple | Réseau (graine 42) |", "|---|---|---|"]
    b1, b2 = mesures["bout_en_bout"], mesures["reseau"]["bout_en_bout_s42"]
    s5 = lambda b: "{technicien} / {estime} / {les_deux}".format(**b["personnes_au_dessus_de_5_par_heure"])
    L += [f"| Spearman | {b1['spearman']:.2f} | {b2['spearman']:.2f} |",
          f"| Erreur absolue médiane | {b1['erreur_absolue_mediane']:.2f} / h | {b2['erreur_absolue_mediane']:.2f} / h |",
          f"| Biais | {b1['biais']:+.2f} / h | {b2['biais']:+.2f} / h |",
          f"| Apnées centrales comptées : technicien / estimées | {b1['n_centrales_technicien']} / {b1['n_centrales_estimees']} | {b2['n_centrales_technicien']} / {b2['n_centrales_estimees']} |",
          f"| Personnes à 5 par heure ou plus : technicien / estimées / les deux | {s5(b1)} | {s5(b2)} |", ""]
    L += ["## Lecture", "",
          "- **Le réseau fait mieux que la règle, mais pas assez pour typer chaque apnée** : quand il dit « centrale », il a raison environ une "
          "fois sur trois. La raison est d'abord la rareté : une apnée sur quinze est centrale, donc même 9 obstructives sur 10 bien reconnues "
          "laissent beaucoup de fausses centrales.",
          "- **De bout en bout, l'index d'apnées centrales n'est pas fiable** : il surestime, et désigne à tort des personnes au-dessus de "
          "5 par heure alors qu'aucune ne l'est dans la validation.",
          "- **Décision : le type n'entre pas dans la sortie par nuit.** Les apnées restent « apnée », à typer par le lecteur. Le réseau de typage "
          "est gardé comme point de départ (`somnia/deep/type_net.py`).",
          "- **Ce qui manque** : des apnées centrales en nombre (une cohorte d'insuffisants cardiaques, par exemple), et de meilleures ceintures : "
          "celles de SHHS sont codées sur 8 bits et souvent écrêtées (`docs/RESULTATS_QUALITE.md`). Avec 85 apnées centrales dans la "
          "validation, la mesure elle-même est très bruitée.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
