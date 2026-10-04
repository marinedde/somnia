"""Tests de somnia/evaluation.py sur des données synthétiques (aucune donnée réelle)."""

import numpy as np
import pytest

from somnia.evaluation import (
    decoupage_aleatoire_par_epoque,
    metriques_apnee,
    metriques_stades,
    validation_croisee_par_personne,
)


def _jeu(n_personnes=10, n_par_personne=60, tache="eeg", graine=0):
    """Un jeu où la classe dépend des caractéristiques ET d'un décalage propre à la personne."""
    rng = np.random.default_rng(graine)
    X, y, pers = [], [], []
    for p in range(n_personnes):
        decalage = rng.normal(0, 2, size=4)
        for _ in range(n_par_personne):
            classe = rng.integers(0, 5 if tache == "eeg" else 2)
            X.append(rng.normal(classe, 1.0, size=4) + decalage)
            y.append(classe); pers.append(f"P{p:02d}")
    return np.array(X), np.array(y), np.array(pers)


def test_metriques_stades_parfaites_et_nulles():
    y = np.array([0, 1, 2, 3, 4] * 4)
    m = metriques_stades(y, y)
    assert m["accuracy"] == 1.0 and m["kappa"] == 1.0 and m["f1_N1"] == 1.0
    m0 = metriques_stades(y, np.full_like(y, 2))
    assert m0["accuracy"] == pytest.approx(0.2) and m0["kappa"] == pytest.approx(0.0)
    assert set(m0) >= {"f1_Wake", "f1_N1", "f1_N2", "f1_N3", "f1_REM", "f1_macro", "f1_weighted"}


def test_metriques_apnee():
    y = np.array([0, 0, 1, 1]); proba = np.array([0.1, 0.4, 0.6, 0.9])
    m = metriques_apnee(y, (proba > 0.5).astype(int), proba)
    assert m["auc_roc"] == 1.0 and m["auc_pr"] == 1.0 and m["f1_apnee"] == 1.0
    m1 = metriques_apnee(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert np.isnan(m1["auc_roc"])          # une seule classe : pas d'AUC, pas de plantage


def test_validation_croisee_ne_melange_jamais_les_personnes():
    X, y, pers = _jeu()
    res = validation_croisee_par_personne(X, y, pers, "eeg", n_plis=5, graine=1)
    assert len(res["plis"]) == 5
    assert sum(p["n_personnes_test"] for p in res["plis"]) == 10      # chaque personne testée une fois
    assert sum(p["n_epoques_test"] for p in res["plis"]) == len(y)
    assert set(res["moyenne"]) == set(res["ecart_type"])
    assert (res["oof"]["pred"] >= 0).all()                             # chaque époque a une prédiction hors pli


def test_validation_croisee_reproductible():
    X, y, pers = _jeu(tache="ecg")
    a = validation_croisee_par_personne(X, y, pers, "ecg", n_plis=4, graine=3)
    b = validation_croisee_par_personne(X, y, pers, "ecg", n_plis=4, graine=3)
    assert a["moyenne"] == b["moyenne"]
    assert np.allclose(a["oof"]["proba"], b["oof"]["proba"], equal_nan=True)


def test_le_decoupage_par_epoque_est_plus_optimiste_que_par_personne():
    """Avec un effet « personne », l'ancien découpage surestime : c'est exactement la fuite mesurée."""
    X, y, pers = _jeu(n_personnes=12, n_par_personne=80, tache="ecg", graine=5)
    par_personne = validation_croisee_par_personne(X, y, pers, "ecg", n_plis=4, graine=0)["moyenne"]["auc_roc"]
    par_epoque = decoupage_aleatoire_par_epoque(X, y, "ecg", graine=0)["metriques"]["auc_roc"]
    assert par_epoque >= par_personne - 0.02


# ── Bootstrap par personne ────────────────────────────────────────────────
def test_bootstrap_encadre_l_estimation_et_se_reproduit():
    import numpy as np
    from somnia.evaluation import bootstrap_par_personne
    v = np.random.default_rng(0).normal(0.7, 0.1, 40)
    r = bootstrap_par_personne(lambda i: v[i].mean(), 40, n_tirages=500)
    assert r["bas"] < r["estimation"] < r["haut"] and abs(r["estimation"] - v.mean()) < 1e-12
    assert 0.03 < r["haut"] - r["bas"] < 0.10                    # ≈ 4 erreurs-types de 0,1 / √40
    assert r == bootstrap_par_personne(lambda i: v[i].mean(), 40, n_tirages=500)


def test_bootstrap_apparie_plus_fin_que_deux_intervalles():
    """Deux modèles mesurés sur les mêmes personnes : la différence appariée est bien plus précise."""
    import numpy as np
    from somnia.evaluation import bootstrap_par_personne
    rng = np.random.default_rng(1)
    difficulte = rng.normal(0.7, 0.1, 40)                        # certaines nuits sont dures pour tout le monde
    a, b = difficulte + rng.normal(0, 0.005, 40), difficulte + 0.02 + rng.normal(0, 0.005, 40)
    d = bootstrap_par_personne(lambda i: b[i].mean() - a[i].mean(), 40, n_tirages=500)
    seul = bootstrap_par_personne(lambda i: a[i].mean(), 40, n_tirages=500)
    assert d["bas"] > 0 and d["part_positive"] == 1.0
    assert d["haut"] - d["bas"] < 0.25 * (seul["haut"] - seul["bas"])
