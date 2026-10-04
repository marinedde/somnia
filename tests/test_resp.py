"""Tests de la détection d'événements respiratoires : étiquettes, masque <-> événements, appariement, index, désaturation."""

import numpy as np

from somnia.resp import (
    Ev,
    apparier,
    etiquettes_par_seconde,
    evenements_de_reference,
    index_de_desaturation,
    index_par_heure,
    masque_vers_evenements,
    nettoyer_sao2,
    normaliser_fenetre,
    sommeil_par_seconde,
    trouver,
    vers_10hz,
)
from somnia.shhs import Annotations, Evenement


def _ann(stades, evs):
    return Annotations(30.0, np.asarray(stades, dtype=np.int64), list(evs))


def test_etiquettes_par_seconde():
    ann = _ann([2, 2, 2, 2], [Evenement("Obstructive apnea", "Respiratory", 10.4, 12.0),      # 10 -> 23
                               Evenement("Hypopnea", "Respiratory", 60.0, 20.0),               # 60 -> 80
                               Evenement("SpO2 desaturation", "Respiratory", 30.0, 15.0),      # ignorée
                               Evenement("Arousal", "Arousals", 90.0, 5.0)])                   # ignoré
    y = etiquettes_par_seconde(ann)
    assert len(y) == 120
    assert (y[10:23] == 1).all() and y[9] == 0 and y[23] == 0
    assert (y[60:80] == 2).all() and y[30:45].sum() == 0 and y[90:95].sum() == 0


def test_apnee_l_emporte_sur_hypopnee_en_cas_de_recouvrement():
    ann = _ann([2, 2], [Evenement("Hypopnea", "Respiratory", 10, 30), Evenement("Central apnea", "Respiratory", 20, 10)])
    y = etiquettes_par_seconde(ann)
    assert (y[10:20] == 2).all() and (y[20:30] == 1).all() and (y[30:40] == 2).all()


def test_masque_vers_evenements_duree_minimale_et_trous():
    y = np.zeros(200, dtype=int)
    y[10:25] = 2            # 15 s : gardé
    y[40:46] = 1            # 6 s : trop court, retiré
    y[100:108] = 1; y[110:120] = 1    # deux morceaux séparés par 2 s : fusionnés en 20 s
    evs = masque_vers_evenements(y)
    assert [(e.debut, e.fin, e.classe) for e in evs] == [(10, 25, 2), (100, 120, 1)]


def test_aller_retour_reference():
    ann = _ann([2] * 10, [Evenement("Hypopnea", "Respiratory", 30, 18), Evenement("Obstructive apnea", "Respiratory", 100, 25)])
    ref = evenements_de_reference(ann)
    relu = masque_vers_evenements(etiquettes_par_seconde(ann))
    assert [(e.debut, e.fin, e.classe) for e in ref] == [(e.debut, e.fin, e.classe) for e in relu]


def test_appariement():
    ref = [Ev(10, 30, 1), Ev(100, 120, 2), Ev(200, 215, 2)]
    pred = [Ev(12, 28, 1), Ev(105, 150, 2), Ev(300, 320, 2)]          # 2 trouvés, 1 inventé, 1 manqué
    m = apparier(pred, ref)
    assert (m["vp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert abs(m["f1"] - 2 / 3) < 1e-9
    strict = apparier(pred, ref, iou_min=0.5)                         # (105-150 vs 100-120) : IoU 0,3 -> refusé
    assert strict["vp"] == 1
    assert apparier([], ref)["f1"] == 0.0 and apparier(pred, [])["fp"] == 3


def test_un_evenement_de_reference_n_est_apparie_qu_une_fois():
    m = apparier([Ev(10, 20, 1), Ev(20, 30, 1)], [Ev(10, 30, 1)])
    assert (m["vp"], m["fp"], m["fn"]) == (1, 1, 0)


def test_index_par_heure_ne_compte_que_le_sommeil():
    sommeil = sommeil_par_seconde(np.array([0] * 120 + [2] * 240))     # 1 h d'éveil puis 2 h de sommeil
    evs = [Ev(100, 120, 1)] + [Ev(3600 + 600 * i, 3600 + 600 * i + 20, 2) for i in range(10)]
    assert index_par_heure(evs, sommeil) == 5.0                        # 10 événements en 2 h de sommeil


def test_nettoyer_sao2():
    x = np.array([0, 0, 96, 97, 0, 0, 95, 120, 94], dtype=float)
    assert nettoyer_sao2(x).tolist() == [96, 96, 96, 97, 97, 97, 95, 95, 94]


def test_index_de_desaturation():
    sao2 = np.full(3600, 96.0)                       # une heure de sommeil
    for t0 in (300, 900, 1500, 2100, 2700):          # cinq chutes de 5 points, 30 s chacune
        sao2[t0:t0 + 30] = 91.0
    sao2[3300:3330] = 94.5                           # une chute de 1,5 point : pas comptée
    sommeil = np.ones(3600, dtype=bool)
    assert index_de_desaturation(sao2, sommeil, chute=3.0) == 5.0
    assert index_de_desaturation(sao2, np.zeros(3600, dtype=bool)) != index_de_desaturation(sao2, sommeil)  # nan sans sommeil


def test_vers_10hz_et_noms_de_canaux():
    assert vers_10hz(np.array([95.0, 96.0]), 1).tolist() == [95.0] * 10 + [96.0] * 10      # maintien, pas d'ondulation
    assert len(vers_10hz(np.zeros(500), 50)) == 100
    noms = ["SaO2", "THOR RES", "ABDO RES", "AIRFLOW", "ECG"]
    assert trouver(noms, ["NEW AIR", "AIRFLOW"]) == "AIRFLOW" and trouver(noms, ["Thor Res"]) == "THOR RES"


def test_normaliser_fenetre():
    x = np.stack([np.random.default_rng(0).normal(5, 3, 3000) for _ in range(3)] + [np.full(3000, 90.0)])
    z = normaliser_fenetre(x)
    assert abs(z[0].mean()) < 1e-3 and abs(z[0].std() - 1) < 1e-2
    assert np.allclose(z[3], -1.0)                   # 90 % -> −1
    assert np.isfinite(normaliser_fenetre(np.zeros((4, 3000)))).all()
