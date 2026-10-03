"""Tests de la sortie par nuit : indices et file de relecture, sur des données synthétiques."""

import numpy as np

from somnia.nuit import analyser_nuit, file_de_relecture, indices_de_nuit


def test_indices_de_nuit():
    # 4 éveil, 10 N2, 2 éveil (intra-sommeil), 4 N3, 4 éveil final : 20 époques de sommeil sur 24
    stades = np.array([0] * 4 + [2] * 10 + [0] * 2 + [3] * 4 + [0] * 4)
    i = indices_de_nuit(stades)
    assert i["n_epoques"] == 24
    assert i["temps_enregistrement_min"] == 12.0
    assert i["temps_sommeil_total_min"] == 7.0
    assert i["latence_endormissement_min"] == 2.0
    assert i["eveil_intra_sommeil_min"] == 1.0
    assert abs(sum(i["parts_pct"].values()) - 100) < 0.2


def test_file_de_relecture_segments_tries():
    conf = np.array([0.9, 0.9, 0.3, 0.4, 0.9, 0.9, 0.5, 0.9, 0.2, 0.2, 0.2, 0.9])
    stades = np.array([0, 0, 1, 2, 2, 2, 4, 4, 2, 2, 3, 3])
    f = file_de_relecture(conf, stades, seuil=0.6)
    assert f["n_epoques_a_relire"] == 6 and f["temps_relecture_estime_min"] == 3.0
    assert [s["debut_epoque"] for s in f["segments"]] == [8, 2, 6]      # le moins sûr d'abord
    assert f["segments"][0]["duree_min"] == 1.5 and f["segments"][0]["debut_hhmm"] == "00:04"


def test_analyser_nuit_bout_en_bout():
    rng = np.random.default_rng(0)
    epoques = rng.normal(size=(50, 3000)) * 2e-5
    cibles = rng.integers(0, 5, 50)

    def predire(x):
        logits = np.full((len(x), 5), -2.0); logits[np.arange(len(x)), cibles] = 2.0
        logits[:10] = 0.0                                              # dix époques sans avis
        return logits
    r = analyser_nuit(epoques, predire, temperature=1.0, seuil=0.6)
    assert len(r["hypnogramme"]) == 50 and set(r["hypnogramme"]) <= {"W", "N1", "N2", "N3", "R"}
    assert r["relecture"]["n_epoques_a_relire"] == 10
    assert "NON VALIDÉ" in r["statut"]
