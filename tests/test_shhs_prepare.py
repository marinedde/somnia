"""Tests des briques de préparation SHHS (sans fichier SHHS)."""

import numpy as np
import pytest

from somnia.shhs_prepare import N_POINTS, decouper_en_epoques, reechantillonner, trouver_canal


def test_reechantillonnage_125_vers_100():
    fs = 125
    t = np.arange(0, 30, 1 / fs)                       # une époque de 30 s : 3750 points
    x = np.sin(2 * np.pi * 2 * t)                      # 2 Hz, bien sous Nyquist
    y = reechantillonner(x, fs, 100)
    assert len(y) == 3000
    # le contenu est préservé : même amplitude, même fréquence
    assert abs(np.abs(y).max() - 1.0) < 0.05
    f = np.fft.rfftfreq(len(y), 1 / 100)[np.argmax(np.abs(np.fft.rfft(y)))]
    assert abs(f - 2.0) < 0.1


def test_reechantillonnage_sans_changement():
    x = np.random.default_rng(0).normal(size=1000)
    assert reechantillonner(x, 100, 100) is x


def test_decoupage_en_epoques():
    x = np.arange(3 * N_POINTS, dtype=float)
    e = decouper_en_epoques(x, 3)
    assert e.shape == (3, N_POINTS)
    assert e[1, 0] == N_POINTS                          # la 2e époque commence au point 3000


def test_decoupage_signal_trop_court_complete_par_zeros():
    x = np.ones(2 * N_POINTS + 10)
    e = decouper_en_epoques(x, 3)
    assert e.shape == (3, N_POINTS)
    assert e[2, 10:].sum() == 0 and e[2, :10].sum() == 10


def test_trouver_canal_tolerant_aux_variantes():
    noms = ["SaO2", "EEG(sec)", "ECG", "EEG", "NEW AIR"]
    assert trouver_canal(noms, ["EEG"]) == "EEG"
    assert trouver_canal(noms, ["EEG (sec)"]) == "EEG(sec)"
    assert trouver_canal(noms, ["New Air", "Airflow"]) == "NEW AIR"
    assert trouver_canal(noms, ["Airflow"]) is None
