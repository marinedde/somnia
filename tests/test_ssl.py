"""Tests du pré-entraînement contrastif : transformations, perte InfoNCE, lots par personne."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch absent : environnement API / CI")

from somnia.deep.data import Tableau  # noqa: E402
from somnia.deep.model import Encodeur  # noqa: E402
from somnia.deep.ssl import (  # noqa: E402
    LotsParPersonne,
    ProjectionContrastive,
    amplitude,
    bruit,
    decalage,
    info_nce,
    inversion_signe,
    masquage,
    petit_lot_contrastif,
    vue,
)


def test_transformations_gardent_la_forme_et_restent_reconnaissables():
    torch.manual_seed(0)
    t = torch.arange(3000) / 100
    x = torch.sin(2 * np.pi * 1.0 * t).view(1, 1, -1).repeat(4, 1, 1)
    for f in (bruit, amplitude, decalage, masquage, inversion_signe):
        y = f(x.clone())
        assert y.shape == x.shape
    # décalage : même contenu, décalé (corrélation max après réalignement proche de 1)
    y = decalage(x.clone())
    assert torch.allclose(y.abs().max(), x.abs().max(), atol=1e-5)
    # masquage : au plus 3 s mises à zéro
    y = masquage(x.clone())
    assert ((y == 0).float().sum(dim=-1) <= 301).all()
    # amplitude : facteur entre 0,8 et 1,2
    y = amplitude(x.clone())
    assert ((y.abs().max(dim=-1).values >= 0.8 - 1e-4) & (y.abs().max(dim=-1).values <= 1.2 + 1e-4)).all()
    # la vue complète est centrée-réduite
    v = vue(x.clone(), "eeg")
    assert torch.allclose(v.mean(dim=-1), torch.zeros(4, 1), atol=1e-3)


def test_inversion_de_signe_seulement_pour_l_ecg():
    """Pour l'EEG, aucune transformation ne retourne le signal."""
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 3000).view(1, 1, -1).repeat(64, 1, 1)
    v = vue(x.clone(), "eeg")
    # la pente reste positive pour tous les exemples (pas d'inversion), malgré le décalage circulaire
    pentes = v[:, 0, 1500] - v[:, 0, 1000]
    assert (pentes > 0).float().mean() > 0.9


def test_info_nce_basse_pour_des_lots_identiques_haute_sans_rapport():
    """Critère 5.3 : deux lots identiques -> perte faible ; deux lots sans rapport -> perte élevée."""
    torch.manual_seed(0)
    z = torch.randn(64, 32)
    perte_identiques = info_nce(z, z.clone(), temperature=0.1)
    perte_sans_rapport = info_nce(z, torch.randn(64, 32), temperature=0.1)
    hasard = float(np.log(2 * 64 - 1))
    assert perte_identiques < 0.1
    assert perte_sans_rapport > hasard - 0.5


def test_lots_une_epoque_par_personne():
    pers = np.repeat([f"P{i}" for i in range(20)], 50)
    tab = Tableau(np.zeros((1000, 3000), dtype=np.float16), np.zeros(1000, dtype=np.int64), pers)
    lots = LotsParPersonne(tab, taille_lot=16, graine=1)
    x, qui = lots.lot()
    assert x.shape == (16, 1, 3000)
    assert len(set(qui.tolist())) == 16                 # 16 personnes différentes


def test_projection_contrastive_forme():
    m = ProjectionContrastive(Encodeur(dim=64), dim_proj=32)
    assert m(torch.randn(8, 1, 3000)).shape == (8, 32)


def test_petit_lot_contrastif_apprend():
    """Critère 5.4 : la perte contrastive descend nettement sur 32 époques fixes."""
    rng = np.random.default_rng(0)
    t = np.arange(3000) / 100
    X = np.stack([np.sin(2 * np.pi * rng.uniform(0.5, 8) * t + rng.uniform(0, 6)) * rng.uniform(0.5, 2) for _ in range(64)])
    tab = Tableau(X.astype(np.float16), np.zeros(64, dtype=np.int64), np.array(["p"] * 64))
    r = petit_lot_contrastif("eeg", tab, n=32, pas=120, journal=lambda *_: None)
    assert r["perte_fin"] < 0.5 * r["perte_debut"]
