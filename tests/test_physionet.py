"""Tests de somnia/physionet.py sans fichier EDF : élagage de l'éveil, appariement, assemblage."""

import numpy as np

from somnia.physionet import STAGE_MAPPING, _assembler, elaguer_eveil, enregistrements_apnea_ecg, paires_sleep_edf


def test_elaguer_eveil_garde_les_micro_eveils_internes():
    labels = np.array([0, 0, 0, 2, 2, 0, 0, 3, 4, 0, 0])   # éveil avant, micro-éveil au milieu, éveil après
    epochs = np.arange(len(labels))[:, None] * np.ones((1, 4))
    e, l = elaguer_eveil(epochs, labels)
    assert l.tolist() == [2, 2, 0, 0, 3, 4]
    assert e[0, 0] == 3 and e[-1, 0] == 8                    # les époques restent alignées sur les étiquettes


def test_elaguer_eveil_sans_sommeil_ne_change_rien():
    labels = np.zeros(5, dtype=int); epochs = np.zeros((5, 3))
    e, l = elaguer_eveil(epochs, labels)
    assert len(l) == 5 and e.shape == (5, 3)


def test_mapping_des_stades_fusionne_3_et_4_et_ignore_inconnu():
    assert STAGE_MAPPING["Sleep stage 3"] == STAGE_MAPPING["Sleep stage 4"] == 3
    assert STAGE_MAPPING["Sleep stage R"] == 4 and STAGE_MAPPING["Movement time"] == 0
    assert "Sleep stage ?" not in STAGE_MAPPING


def test_paires_sleep_edf_ignore_les_psg_sans_hypnogramme(tmp_path):
    for nom in ("SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf", "SC4011E0-PSG.edf",
                "SC4012E0-PSG.edf", "SC4012EC-Hypnogram.edf"):
        (tmp_path / nom).write_bytes(b"")
    paires = paires_sleep_edf(tmp_path)
    assert [p[0] for p in paires] == ["SC4001E", "SC4012E"]      # SC4011 n'a pas d'hypnogramme


def test_enregistrements_apnea_ecg_exige_signal_et_annotations(tmp_path):
    for nom in ("a01.dat", "a01.apn", "a01.hea", "a02.apn", "a03.dat"):
        (tmp_path / nom).write_bytes(b"")
    assert enregistrements_apnea_ecg(tmp_path) == ["a01"]


def test_assembler_aligne_et_nettoie():
    X = [np.array([[1.0, np.nan], [np.inf, 2.0]], dtype=np.float32)]
    t = _assembler(X, [np.array([0, 1])], ["SC00", "SC00"], ["SC4001E", "SC4001E"], ["a", "b"])
    assert t["X"].shape == (2, 2) and np.isfinite(t["X"]).all()
    assert t["y"].tolist() == [0, 1] and t["personne"].tolist() == ["SC00", "SC00"]
    assert t["feature_names"].tolist() == ["a", "b"]
