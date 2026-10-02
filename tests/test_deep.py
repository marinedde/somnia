"""Tests du réseau convolutif et de ses briques, sur CPU, sans donnée réelle.

PyTorch n'est installé que dans l'environnement de recherche (requirements-research.txt), pas
dans celui de l'API ni dans la CI : sans lui, ce module est ignoré au lieu de casser la collecte.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch absent : environnement API / CI")

from somnia.deep.data import Tableau, normaliser_lot, poids_de_classes, sous_ensemble_de_personnes
from somnia.deep.model import CNN1D, Encodeur, n_parametres
from somnia.deep.train import courbe_couverture, ece, ajuster_temperature, petit_lot


def test_forme_de_sortie_eeg_et_ecg():
    """Critère de fin 4.2 : un signal (8, 1, 3000) donne une sortie (8, 5) ; (8, 1, 6000) -> (8, 2)."""
    assert CNN1D(5)(torch.randn(8, 1, 3000)).shape == (8, 5)
    assert CNN1D(2)(torch.randn(8, 1, 6000)).shape == (8, 2)


def test_encodeur_vecteur_de_taille_fixe():
    enc = Encodeur(dim=64)
    assert enc(torch.randn(4, 1, 3000)).shape == (4, 64)
    assert enc(torch.randn(4, 1, 6000)).shape == (4, 64)        # même encodeur, deux longueurs


def test_taille_du_reseau():
    n = n_parametres(CNN1D(5))
    assert 30_000 < n < 150_000, n                               # ~70 000 attendus


def test_normalisation_par_exemple():
    x = torch.randn(3, 1, 3000) * torch.tensor([1.0, 50.0, 1e-3]).view(3, 1, 1) + 7
    z = normaliser_lot(x)
    assert torch.allclose(z.mean(dim=-1), torch.zeros(3, 1), atol=1e-3)   # float32 : un décalage de 7 sur un signal de 1e-3 laisse ~2e-4
    assert torch.allclose(z.std(dim=-1), torch.ones(3, 1), atol=1e-2)
    assert torch.isfinite(normaliser_lot(torch.zeros(1, 1, 3000))).all()   # signal plat : 0, pas NaN


def test_poids_de_classes_equilibrent():
    w = poids_de_classes(np.array([0] * 90 + [1] * 10), 2)
    assert w[1] > w[0] and abs(w[0] * 90 + w[1] * 10 - 100) < 1e-4


def test_sous_ensemble_de_personnes_reproductible():
    pers = [f"P{i}" for i in range(192)]
    a = sous_ensemble_de_personnes(pers, 0.10, 42)
    assert len(a) == 19 and a == sous_ensemble_de_personnes(pers, 0.10, 42)
    assert len(sous_ensemble_de_personnes(pers, 0.01, 42)) == 2
    assert sous_ensemble_de_personnes(pers, 1.0) == sorted(pers)


def test_petit_lot_apprend_par_coeur_un_jeu_separable():
    """Critère 4.3 : sur 32 exemples séparables, la perte tombe près de 0."""
    rng = np.random.default_rng(0)
    t = np.arange(3000) / 100
    X = np.stack([np.sin(2 * np.pi * (1 + 3 * k) * t) + 0.1 * rng.normal(size=3000) for k in rng.integers(0, 5, 64)])
    y = np.array([int(round((np.abs(np.fft.rfft(x)).argmax() / 30 - 1) / 3)) for x in X])
    tab = Tableau(X.astype(np.float16), y, np.array(["p"] * 64))
    r = petit_lot("eeg", tab, n=32, pas=150, journal=lambda *_: None)
    assert r["perte_fin"] < r["perte_debut"] * 0.2


def test_ece_et_temperature():
    y = np.array([0, 1] * 500)
    logits_surs = np.where(y[:, None] == np.arange(2), 6.0, -6.0) * np.where(np.arange(1000) % 10 == 0, -1, 1)[:, None]
    proba = np.exp(logits_surs) / np.exp(logits_surs).sum(axis=1, keepdims=True)
    e = ece(proba, y)
    assert 0.05 < e < 0.15                                      # 90 % juste mais 99,9 % sûr : mal calibré
    T = ajuster_temperature(logits_surs, y)
    assert T > 1.5                                               # la température adoucit des logits trop sûrs
    proba_T = np.exp(logits_surs / T) / np.exp(logits_surs / T).sum(axis=1, keepdims=True)
    assert ece(proba_T, y) < e


def test_courbe_couverture_monotone_sur_un_cas_simple():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    conf = rng.uniform(0.5, 1, 400)
    pred = np.where(rng.uniform(size=400) < conf, y, 1 - y)   # plus la confiance est haute, plus c'est juste
    proba = np.zeros((400, 2)); proba[np.arange(400), pred] = conf; proba[np.arange(400), 1 - pred] = 1 - conf
    c = courbe_couverture(proba, y, "ecg")
    assert [r["couverture"] for r in c] == [1.0, 0.9, 0.8, 0.7, 0.5]
    assert c[-1]["accuracy"] >= c[0]["accuracy"] - 0.02
