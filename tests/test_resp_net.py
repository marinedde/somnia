"""Tests du réseau de détection d'événements : fenêtres, forme, masque, mesures (CPU, données synthétiques)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch absent : environnement API / CI")

from somnia.deep.resp_net import (  # noqa: E402
    FenetresResp,
    NuitResp,
    ReseauEvenements,
    debuts_de_fenetres,
    extraire_fenetre,
    mesurer_nuits,
    proba_nuit,
    proba_vers_masque,
)


def _nuit(n_sec=1200, evenements=((100, 130, 2), (400, 425, 1), (800, 840, 2)), ident="n"):
    rng = np.random.default_rng(0)
    S = rng.normal(size=(4, n_sec * 10)).astype(np.float32); S[3] = 95.0
    y = np.zeros(n_sec, dtype=np.int8)
    for a, b, c in evenements:
        y[a:b] = c
    return NuitResp(ident, f"p-{ident}", S, y, np.full(n_sec // 30, 2, dtype=np.int8))


def test_debuts_de_fenetres_couvrent_la_nuit():
    d = debuts_de_fenetres(1000)
    assert d[0] == 0 and d[-1] == 700 and all(b - a <= 150 for a, b in zip(d[:-1], d[1:]))
    assert debuts_de_fenetres(200) == [0]
    couvert = np.zeros(1000, bool)
    for a in d:
        couvert[a:a + 300] = True
    assert couvert.all()


def test_extraire_fenetre_formes():
    x, y = extraire_fenetre(_nuit(), 150)
    assert x.shape == (4, 3000) and y.shape == (300,) and x.dtype == np.float32
    x2, y2 = extraire_fenetre(_nuit(n_sec=200, evenements=()), 0)       # nuit plus courte qu'une fenêtre
    assert x2.shape == (4, 3000) and y2.shape == (300,)


def test_forme_de_sortie_du_reseau():
    """Un lot (8, 4, 3000) donne (8, 3, 300) : une réponse par seconde."""
    assert ReseauEvenements()(torch.randn(8, 4, 3000)).shape == (8, 3, 300)


def test_dataset():
    ds = FenetresResp([_nuit(), _nuit(ident="m")], entrainement=True)
    x, y = ds[0]
    assert x.shape == (4, 3000) and y.shape == (300,) and y.dtype == torch.int64
    assert len(ds) == 2 * len(debuts_de_fenetres(1200))


def test_proba_vers_masque():
    p = np.array([[0.9, 0.05, 0.05], [0.3, 0.6, 0.1], [0.4, 0.1, 0.5], [0.6, 0.3, 0.1]])
    assert proba_vers_masque(p, seuil=0.5).tolist() == [0, 1, 2, 0]
    assert proba_vers_masque(p).tolist() == [0, 1, 0, 0]          # seuil par défaut 0,7, choisi sur la validation


def test_proba_nuit_forme_et_normalisation():
    n = _nuit(n_sec=700)
    p = proba_nuit(ReseauEvenements(), n, torch.device("cpu"))
    assert p.shape == (700, 3) and np.allclose(p.sum(axis=1), 1, atol=1e-5)


def test_mesures_parfaites_et_degradees():
    n = _nuit()
    parfait = mesurer_nuits([n], [n.y.copy()])
    assert parfait["par_evenement"]["recouvrement"]["f1"] == 1.0 and parfait["par_seconde"]["f1_evenement"] == 1.0
    assert parfait["par_evenement"]["rappel_apnees"] == 1.0 and parfait["par_evenement"]["n_hypopnees_ref"] == 2
    rien = mesurer_nuits([n], [np.zeros_like(n.y)])
    assert rien["par_evenement"]["recouvrement"]["f1"] == 0.0 and rien["par_evenement"]["recouvrement"]["fn"] == 3
    decale = n.y.copy(); decale[:] = 0; decale[110:150] = 2; decale[400:425] = 1          # 2 trouvés sur 3, bornes approximatives
    m = mesurer_nuits([n], [decale])
    assert m["par_evenement"]["recouvrement"]["vp"] == 2 and m["par_evenement"]["recouvrement"]["fn"] == 1


# ── Type d'apnée (horizon 2) ──────────────────────────────────────────────
def test_fenetre_apnee_forme_et_bords():
    from somnia.deep.type_net import N_POINTS, ReseauType, fenetre_apnee, proba_centrale_reseau
    s = np.random.default_rng(0).normal(size=(4, 6000)).astype(np.float32)          # 10 minutes à 10 Hz
    x = fenetre_apnee(s, 100)
    assert x.shape == (3, N_POINTS) and abs(x.mean()) < 1e-4 and abs(x.std() - 1) < 1e-2
    assert fenetre_apnee(s, 10) is None and fenetre_apnee(s, 590) is None           # sort de la nuit
    s[1, 700:1600] = 0.0
    assert fenetre_apnee(s, 100) is None                                            # ceinture plate : pas de typage
    p = proba_centrale_reseau(ReseauType(), np.stack([x, x]), torch.device("cpu"))
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()


def test_effort_respiratoire_distingue_ceintures_plates():
    from somnia.resp import effort_respiratoire
    t = np.arange(3000) / 10
    s = np.stack([np.sin(2 * np.pi * 0.25 * t)] * 3).astype(np.float32)             # respiration régulière sur 3 canaux
    s[:, 1000:1200] *= 0.05                                                         # 20 s où tout s'aplatit : centrale
    traits = effort_respiratoire(s, 100, 120)
    assert traits[0] < -2 and traits[1] < -2                                        # thorax et abdomen : amplitude / 20
    assert effort_respiratoire(s, 10, 30) is None                                   # pas assez de signal avant
