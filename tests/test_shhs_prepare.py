"""Tests des briques de préparation SHHS, sans fichier SHHS (signaux synthétiques)."""

import numpy as np
import pytest

from somnia.shhs import Annotations, Evenement
from somnia.shhs_prepare import (
    N_POINTS,
    construire_nuit,
    decouper_en_epoques,
    fenetres_60s,
    reechantillonner,
    trouver_canal,
)


def test_reechantillonnage_125_vers_100():
    fs = 125
    t = np.arange(0, 30, 1 / fs)                       # une époque de 30 s : 3750 points
    x = np.sin(2 * np.pi * 2 * t)                      # 2 Hz, bien sous Nyquist
    y = reechantillonner(x, fs, 100)
    assert len(y) == 3000
    assert abs(np.abs(y).max() - 1.0) < 0.05           # amplitude préservée
    f = np.fft.rfftfreq(len(y), 1 / 100)[np.argmax(np.abs(np.fft.rfft(y)))]
    assert abs(f - 2.0) < 0.1                           # fréquence préservée


def test_reechantillonnage_sans_changement():
    x = np.random.default_rng(0).normal(size=1000)
    assert reechantillonner(x, 100, 100) is x


def test_decoupage_en_epoques():
    x = np.arange(3 * N_POINTS, dtype=float)
    e, inc = decouper_en_epoques(x, 3)
    assert e.shape == (3, N_POINTS) and not inc.any()
    assert e[1, 0] == N_POINTS                          # la 2e époque commence au point 3000


def test_epoque_incomplete_est_marquee():
    """Signal plus court que les annotations : l'époque manquante est marquée, pas apprise."""
    x = np.ones(2 * N_POINTS + 10)
    e, inc = decouper_en_epoques(x, 3)
    assert e.shape == (3, N_POINTS)
    assert inc.tolist() == [False, False, True]


def test_trouver_canal_tolerant_aux_variantes():
    noms = ["SaO2", "EEG(sec)", "ECG", "EEG", "NEW AIR"]
    assert trouver_canal(noms, ["EEG"]) == "EEG"
    assert trouver_canal(noms, ["EEG (sec)"]) == "EEG(sec)"
    assert trouver_canal(noms, ["New Air", "Airflow"]) == "NEW AIR"
    assert trouver_canal(noms, ["Airflow"]) is None


# ── construire_nuit : unités, 125 -> 100 Hz, alignement, époques incomplètes ──
def _annotations(stades, evenements=()):
    return Annotations(30.0, np.asarray(stades, dtype=np.int64), list(evenements))


def test_construire_nuit_rend_les_amplitudes_d_origine_en_uv_et_mv():
    """Le seul test qui attraperait une régression d'unité : volts en entrée, µV / mV en sortie."""
    fs = 125
    t = np.arange(0, 4 * 30, 1 / fs)
    eeg_v = 50e-6 * np.sin(2 * np.pi * 1.0 * t)        # EEG : 50 µV d'amplitude, en volts
    ecg_v = 1.2e-3 * np.sin(2 * np.pi * 1.2 * t)       # ECG : 1,2 mV d'amplitude, en volts
    nuit = construire_nuit(eeg_v, fs, ecg_v, fs, _annotations([0, 2, 3, 4]))
    assert nuit["eeg"].shape == (4, 3000) and nuit["ecg"].shape == (4, 3000)
    assert abs(np.abs(nuit["eeg"]).max() - 50.0) / 50.0 < 0.01     # µV, à 1 % près
    assert abs(np.abs(nuit["ecg"]).max() - 1.2) / 1.2 < 0.01       # mV, à 1 % près
    assert nuit["stades"].tolist() == [0, 2, 3, 4] and not nuit["incomplet"].any()


def test_construire_nuit_aligne_le_signal_sur_les_epoques():
    """Un signal qui vaut k pendant la k-ième époque (à 125 Hz) doit se retrouver dans l'époque k à 100 Hz."""
    fs = 125
    eeg_v = np.repeat(np.arange(1, 4, dtype=float) * 1e-6, 30 * fs)
    nuit = construire_nuit(eeg_v, fs, eeg_v, fs, _annotations([2, 2, 2]))
    for k in range(3):
        milieu = nuit["eeg"][k, 500:2500]               # on évite les bords (filtre anti-repliement)
        assert np.allclose(milieu, k + 1, atol=0.05)


def test_construire_nuit_marque_les_epoques_sans_signal():
    fs = 125
    court = np.ones(int(2.5 * 30 * fs)) * 1e-6          # 2,5 époques de signal pour 4 annotées
    ann = _annotations([0, 2, 2, 2], [Evenement("Hypopnea", "Respiratory", 95.0, 20.0)])
    nuit = construire_nuit(court, fs, court, fs, ann)
    assert nuit["incomplet"].tolist() == [False, False, True, True]
    assert nuit["stades"].tolist() == [0, 2, -1, -1]    # jamais apprises
    assert nuit["apnee"].tolist() == [0, 0, 0, 0]       # l'hypopnée tombait dans une époque incomplète


def test_construire_nuit_etiquette_apnee_alignee():
    fs = 125
    x = np.ones(4 * 30 * fs) * 1e-6
    ann = _annotations([2, 2, 2, 2], [Evenement("Obstructive apnea", "Respiratory", 70.5, 22.0)])
    nuit = construire_nuit(x, fs, x, fs, ann)
    assert nuit["apnee"].tolist() == [0, 0, 1, 0]       # 19,5 s dans l'époque 2, 2,5 s dans la 3


# ── fenetres_60s : paires (2k, 2k+1), étiquette max, éveil exclu ──
def test_fenetres_60s():
    stades = np.array([0, 2, 2, 2, 2, 0, 3, 4, 4])     # 9 époques -> 4 paires, la dernière ignorée
    apnee = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1])
    aberr = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)
    garde, y, ab = fenetres_60s(stades, apnee, aberr)
    assert garde.tolist() == [False, True, False, True]  # (0,2) éveil ; (2,2) ok ; (2,0) éveil ; (3,4) ok
    assert y.tolist() == [1, 1, 0, 0]                    # max des deux époques
    assert ab.tolist() == [False, False, True, False]
    assert len(garde) == 4
