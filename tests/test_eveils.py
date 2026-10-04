"""Tests de la détection des micro-éveils (CPU, données synthétiques)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch absent : environnement API")

from somnia.deep.eveils import (FENETRE_S, FS, NuitEveils, ReseauEveils, etiquettes_eveils, evenements_eveils, extraire,  # noqa: E402
                                mesurer_eveils, proba_eveils, signaux_continus)
from somnia.shhs import Annotations, Evenement  # noqa: E402


def _nuit(n_sec=600, eveils=((100, 110), (300, 306))):
    y = np.zeros(n_sec, dtype=np.int8)
    for a, b in eveils:
        y[a:b] = 1
    X = np.random.default_rng(0).normal(size=(3, n_sec * FS)).astype(np.float16)
    return NuitEveils("n", "p", X, y, np.full(n_sec // 30, 2, dtype=np.int8), p_sommeil=np.ones(n_sec, dtype=np.float32))


def test_etiquettes_par_seconde():
    ann = Annotations.__new__(Annotations)
    ann.evenements = [Evenement("Arousal", "Arousals", 10.4, 6.0), Evenement("Hypopnea", "Respiratory", 50.0, 20.0),
                      Evenement("ASDA arousal", "Arousals", 100.0, 4.0)]
    y = etiquettes_eveils(ann, 200)
    assert y[10:17].all() and y[100:104].all() and y.sum() == 11        # l'hypopnée n'est pas un micro-éveil


def test_forme_du_reseau_une_decision_par_seconde():
    assert ReseauEveils()(torch.randn(2, 3, FENETRE_S * FS)).shape == (2, 2, FENETRE_S)


def test_fenetre_en_fin_de_nuit_ignoree_par_la_perte():
    x, y = extraire(_nuit(150), 60)
    assert x.shape == (3, FENETRE_S * FS) and (y[90:] == -100).all() and (y[:90] >= 0).all()


def test_proba_sur_la_nuit_entiere():
    n = _nuit(500)
    p = proba_eveils(ReseauEveils(), n, torch.device("cpu"))
    assert p.shape == (500,) and ((p >= 0) & (p <= 1)).all()


def test_regle_des_trois_secondes_et_mesure():
    p = np.zeros(600); p[100:110] = 0.9; p[200:202] = 0.9; p[300:306] = 0.9      # la bouffée de 2 s n'est pas un micro-éveil
    assert [(e.debut, e.fin) for e in evenements_eveils(p)] == [(100, 110), (300, 306)]
    m = mesurer_eveils([_nuit()], [p])
    assert m["precision"] == 1.0 and m["rappel"] == 1.0 and m["n_reference"] == 2
    p[450:460] = 0.9                                                               # une fausse proposition
    assert abs(mesurer_eveils([_nuit()], [p], sommeil_predit=True)["precision"] - 2 / 3) < 1e-9


def test_signaux_continus_reduits_par_nuit():
    s = (np.random.default_rng(1).normal(size=(4, 5, 3000)) * 30).astype(np.float16)
    X = signaux_continus(s)
    assert X.shape == (3, 12000) and 0.9 < X.astype(np.float32).std() < 1.1
