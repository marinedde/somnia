"""Tests du modèle de stades à plusieurs capteurs (CPU, données synthétiques)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch absent : environnement API / CI")

from somnia.deep.multi import ECRETAGE, NuitMulti, ReseauMulti, predire_nuit_multi, preparer_signaux, reduire_par_nuit  # noqa: E402
from somnia.deep.seq import IGNORER, ReseauSequence, Tranches  # noqa: E402


def test_reduction_par_nuit_garde_le_niveau_relatif():
    """Une époque au tonus effondré reste plus petite que les autres : c'est l'information du REM."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3000)) * 10 + 500          # décalage continu : doit disparaître
    x[5] = rng.normal(size=3000) * 1 + 500              # tonus dix fois plus bas
    r = reduire_par_nuit(x)
    assert abs(r.mean(axis=1)).max() < 1e-3
    assert 0.9 < np.median(r.std(axis=1)) < 1.1 and r[5].std() < 0.15


def test_reduction_capteur_absent_ou_artefact():
    assert not reduire_par_nuit(np.zeros((5, 3000))).any()      # absent : zéros, pas NaN
    x = np.random.default_rng(1).normal(size=(10, 3000)); x[2, 100] = 1e4
    assert abs(reduire_par_nuit(x)).max() <= ECRETAGE


def test_preparer_selectionne_les_capteurs():
    s = (np.random.default_rng(2).normal(size=(6, 5, 3000)) * 30).astype(np.float16)
    X = preparer_signaux(s, (0, 4))
    assert X.shape == (6, 2, 3000) and np.array_equal(X[:, 0], s[:, 0])      # l'EEG reste brut (réduit par époque dans le réseau)
    assert 0.9 < np.median(X[:, 1].astype(np.float32).std(axis=1)) < 1.1


def test_forme_et_tranches():
    m = ReseauMulti(canaux=(0, 2, 3, 4))
    assert m(torch.randn(2, 8, 4, 3000)).shape == (2, 8, 5)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 10).astype(np.int64); y[1] = -1
    n = NuitMulti("n", "p", rng.normal(size=(10, 4, 3000)).astype(np.float16), y)
    x, yy = Tranches([n], longueur=32)[0]
    assert x.shape == (32, 4, 3000) and yy[1] == IGNORER and (yy[10:] == IGNORER).all()
    assert predire_nuit_multi(m, n, torch.device("cpu")).shape == (10, 5)


def test_depart_identique_a_l_encodeur_eeg():
    """Poids des nouveaux capteurs à zéro : au départ, même vecteur par époque que l'encodeur EEG seul."""
    torch.manual_seed(0)
    seq = ReseauSequence().eval()
    m = ReseauMulti(canaux=(0, 2, 3)).eval()
    m.partir_de_l_encodeur_eeg(seq.encodeur.state_dict())
    x = torch.randn(1, 6, 3, 3000)
    with torch.no_grad():
        assert torch.allclose(m.encoder_epoques(x[0]), seq.encoder(x[:, :, 0])[0], atol=1e-5)


def test_les_nouveaux_capteurs_recoivent_un_gradient():
    """Partir de zéro ne bloque pas l'apprentissage : le gradient des nouveaux poids n'est pas nul."""
    m = ReseauMulti(canaux=(0, 4))
    m.partir_de_l_encodeur_eeg(ReseauSequence().encodeur.state_dict())
    m(torch.randn(1, 4, 2, 3000)).sum().backward()
    assert m.encodeur.blocs[0].net[0].weight.grad[:, 1].abs().sum() > 0


def test_un_canal_eog_seul_n_est_pas_reduit_par_epoque():
    m = ReseauMulti(canaux=(0, 4))
    x = torch.randn(3, 2, 3000) * 5
    z = m.normaliser(x)
    assert torch.allclose(z[:, 1], x[:, 1]) and abs(float(z[:, 0].std()) - 1) < 0.05
