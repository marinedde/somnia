"""Tests de la qualité du signal et du refus (données synthétiques, sans PyTorch)."""

import numpy as np

from somnia.qualite import defauts_par_epoque, en_epoques, par_seconde, qualite_respiration, qualite_tete, rapport_qualite


def _nuit(n=40, graine=0):
    return np.random.default_rng(graine).normal(0, 20, size=(n, 3000))


def test_signal_propre_rien_a_signaler():
    d = defauts_par_epoque(_nuit())
    assert not d["inexploitable"].any()


def test_electrode_decollee_epoque_plate():
    x = _nuit(); x[5] = 0.0; x[6] = np.random.default_rng(1).normal(0, 0.2, 3000)       # plat, et presque plat
    d = defauts_par_epoque(x)
    assert d["plat"][5] and d["plat"][6] and d["plat"].sum() == 2


def test_amplificateur_sature_epoque_ecretee():
    x = _nuit(); x[8] = np.clip(x[8] * 20, -125, 125); x[9, 100] = 125                  # un seul point au rail : pas écrêté
    d = defauts_par_epoque(x)
    assert d["ecrete"][8] and not d["ecrete"][9]


def test_un_pic_isole_n_ecrete_pas_la_nuit():
    x = _nuit(); x[3, 50] = 900.0                           # le maximum de la nuit est un pic unique
    assert not defauts_par_epoque(x)["ecrete"].any()


def test_mouvement_signale_mais_exploitable():
    """Un mouvement est de l'éveil, que le réseau reconnaît : signalé, mais pas « inexploitable »."""
    x = _nuit(); x[12] *= 10
    d = defauts_par_epoque(x)
    assert d["artefact"][12] and not d["inexploitable"][12]
    assert defauts_par_epoque(x, causes=("plat", "ecrete", "artefact"))["inexploitable"][12]


def test_flux_ecrete_reste_exploitable_ceintures_non():
    rng = np.random.default_rng(3)
    s = rng.normal(0, 1, size=(4, 20 * 300)).astype(np.float32)
    for c in (0, 1, 2):
        s[c, 0:300] = np.clip(s[c, 0:300] * 50, -10, 10)                                # époque 0 : flux et ceintures écrêtés
    s[0, 300:600] = np.clip(s[0, 300:600] * 50, -10, 10)                                # époque 1 : flux seul écrêté
    q = qualite_respiration(s)
    assert q["flux"]["ecrete"][:2].all() and not q["flux"]["inexploitable"][:2].any()
    assert q["inexploitable"][0] and not q["inexploitable"][1]


def test_capteur_mort_toute_la_nuit():
    d = defauts_par_epoque(np.zeros((10, 3000)))
    assert d["inexploitable"].all()


def test_respiration_regle_de_combinaison():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, size=(4, 20 * 300)).astype(np.float32)                         # 20 époques à 10 Hz
    s[1, 0:300] = 0                                         # époque 0 : thorax seul perdu -> exploitable
    s[1, 300:600] = 0; s[2, 300:600] = 0                    # époque 1 : les deux ceintures -> inexploitable
    s[0, 600:900] = 0                                       # époque 2 : flux perdu -> inexploitable
    invalide = np.zeros(600, dtype=bool); invalide[90:120] = True; invalide[120:130] = True   # époque 3 entière, 10 s de l'époque 4
    q = qualite_respiration(s, sao2_invalide=invalide)
    assert q["inexploitable"].tolist()[:6] == [False, True, True, True, False, False]
    assert q["thorax"]["plat"][0] and q["sao2"]["invalide"][3] and not q["sao2"]["invalide"][4]


def test_par_seconde_et_decoupage():
    assert en_epoques(np.arange(700), 300).shape == (2, 300)
    m = par_seconde(np.array([True, False]), 75)
    assert m[:30].all() and not m[30:].any() and len(m) == 75


def test_refus_de_la_nuit():
    x = np.stack([_nuit(40, 0), _nuit(40, 1)], axis=1)       # EEG, EMG
    x[:12, 0] = 0                                           # EEG perdu sur 30 % de la nuit
    x[:15, 1] = 0                                           # menton perdu sur 37 %
    r = rapport_qualite(tete=qualite_tete(x, ("EEG", "EMG")))
    assert r["refus"]["stades"] and not r["refus"]["index"]
    assert r["capteurs"]["EEG"]["plat"] == 30.0 and len(r["motifs"]) == 2 and r["modele_stades"] == "eeg"
    bon = rapport_qualite(tete=qualite_tete(np.stack([_nuit(40, 0)], axis=1), ("EEG",)))
    assert not bon["refus"]["stades"] and bon["motifs"] == [] and bon["modele_stades"] == "eeg_eog_emg"
