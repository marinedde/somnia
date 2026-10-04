"""Tests du modèle de séquence pour les stades (CPU, données synthétiques)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch absent : environnement API / CI")

from somnia.deep.seq import IGNORER, NuitEEG, ReseauSequence, Tranches, accord_eveil_sommeil, kappa_par_nuit, predire_nuit  # noqa: E402


def _nuit(n=100, ident="n", inconnus=()):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, n).astype(np.int64)
    for i in inconnus:
        y[i] = -1
    return NuitEEG(ident, f"p-{ident}", (rng.normal(size=(n, 3000)) * 20).astype(np.float16), y)


def test_forme_de_sortie():
    """(2 tranches, 8 époques, 3000 points) -> (2, 8, 5) : un stade par époque."""
    assert ReseauSequence()(torch.randn(2, 8, 3000)).shape == (2, 8, 5)


def test_tranches_et_epoques_non_scorees():
    ds = Tranches([_nuit(100, inconnus=(3, 4))], longueur=32)
    x, y = ds[0]
    assert x.shape == (32, 3000) and y.shape == (32,)
    assert y[3] == IGNORER and y[4] == IGNORER and (y[:3] >= 0).all()
    assert len(ds) == 3                                           # 100 époques -> tranches à 0, 32, 64


def test_nuit_plus_courte_que_la_tranche():
    x, y = Tranches([_nuit(10)], longueur=32)[0]
    assert x.shape == (32, 3000) and (y[10:] == IGNORER).all()


def test_la_perte_ignore_les_epoques_non_scorees():
    perte = torch.nn.CrossEntropyLoss(ignore_index=IGNORER)
    logits = torch.zeros(4, 5); logits[:, 2] = 5.0
    y = torch.tensor([2, 2, IGNORER, IGNORER])
    assert perte(logits, y) < 0.05                                # les deux époques ignorées ne pèsent pas


def test_predire_nuit_un_stade_par_epoque():
    n = _nuit(70)
    lg = predire_nuit(ReseauSequence(), n, torch.device("cpu"))
    assert lg.shape == (70, 5) and np.isfinite(lg).all()


def test_le_contexte_change_la_prediction():
    """La même époque, entourée différemment, ne reçoit pas les mêmes logits : c'est le but."""
    torch.manual_seed(0)
    m = ReseauSequence().eval()
    e = torch.randn(1, 1, 3000)
    a = torch.cat([torch.randn(1, 3, 3000), e, torch.randn(1, 3, 3000)], dim=1)
    b = torch.cat([torch.randn(1, 3, 3000) * 5, e, torch.randn(1, 3, 3000) * 5], dim=1)
    with torch.no_grad():
        assert not torch.allclose(m(a)[0, 3], m(b)[0, 3], atol=1e-4)


def test_accord_eveil_sommeil_et_kappa_par_nuit():
    y = np.array([0, 0, 2, 2, 3, 4, 0, 1]); p = np.array([0, 2, 2, 2, 3, 4, 0, 0])
    a = accord_eveil_sommeil(y, p)
    assert a["accord"] == 0.75 and abs(a["sensibilite_eveil"] - 2 / 3) < 1e-9 and a["specificite_eveil"] == 0.8
    n = _nuit(50)
    assert kappa_par_nuit([n], [n.y.copy()])[0] == 1.0
