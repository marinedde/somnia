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


# ── Événements respiratoires dans la sortie par nuit ──────────────────────
from somnia.nuit import analyser_evenements, analyser_nuit_complete, file_commune  # noqa: E402


def _proba(n_sec, blocs):
    """blocs : (debut, fin, classe, p) -> probabilités à la seconde."""
    p = np.tile([0.97, 0.015, 0.015], (n_sec, 1))
    for a, b, c, pe in blocs:
        p[a:b] = [1 - pe, pe if c == 1 else 0.0, pe if c == 2 else 0.0]
    return p


def test_analyser_evenements_trois_niveaux():
    proba = _proba(3600, [(100, 130, 1, 0.95),      # apnée sûre
                          (500, 520, 2, 0.75),      # hypopnée proposée mais à relire
                          (900, 925, 2, 0.50),      # sous le seuil de décision : « possible »
                          (1500, 1504, 1, 0.95)])   # 4 s : trop court, ignoré
    r = analyser_evenements(proba, np.ones(3600, dtype=bool))
    assert [(e["debut_s"], e["duree_s"], e["type"], e["a_relire"]) for e in r["evenements"]] == \
        [(100, 30, "apnée", False), (500, 20, "hypopnée", True)]
    assert [(e["debut_s"], e["duree_s"]) for e in r["possibles"]] == [(900, 25)]
    res = r["resume"]
    assert (res["n_evenements"], res["n_surs"], res["n_a_relire"], res["n_possibles"]) == (2, 1, 1, 1)
    assert res["index_par_heure"] == 2.0 and r["evenements"][0]["debut"] == "00:01:40"


def test_index_ne_compte_que_le_sommeil_predit():
    proba = _proba(7200, [(100, 130, 1, 0.95), (4000, 4030, 2, 0.95)])
    sommeil = np.concatenate([np.zeros(3600, bool), np.ones(3600, bool)])     # première heure en éveil
    r = analyser_evenements(proba, sommeil)
    assert r["resume"]["n_evenements"] == 2 and r["resume"]["index_par_heure"] == 1.0
    assert [e["pendant_le_sommeil"] for e in r["evenements"]] == [False, True]


def test_file_commune_union_et_ordre():
    relecture = {"segments": [{"debut_epoque": 10, "fin_epoque": 11, "confiance_min": 0.4, "stades_proposes": "N1N2"}]}  # 300-360 s
    respiration = {"evenements": [{"debut_s": 310, "debut": "00:05:10", "duree_s": 20, "type": "hypopnée", "confiance": 0.75, "a_relire": True},
                                  {"debut_s": 50, "debut": "00:00:50", "duree_s": 20, "type": "apnée", "confiance": 0.95, "a_relire": False}],
                   "possibles": [{"debut_s": 1000, "debut": "00:16:40", "duree_s": 15, "confiance": 0.5}]}
    f = file_commune(relecture, respiration, n_sec=3600)
    assert [i["quoi"] for i in f["items"]] == ["stade", "événement", "événement possible"]        # ordre de la nuit, le sûr n'y est pas
    # union : stade 300-360 ; événement 280-360 (30 s de contexte) -> 280-360 = 80 s ; possible 970-1045 = 75 s
    assert f["signal_a_relire_min"] == round((80 + 75) / 60, 1) and f["signal_total_min"] == 60.0


def test_analyser_nuit_complete():
    rng = np.random.default_rng(0)
    epoques = rng.normal(size=(120, 3000)) * 2e-5                      # une heure

    def predire(x):
        lg = np.full((len(x), 5), -3.0); lg[:, 2] = 3.0; lg[:5] = 0.0   # tout en N2, 5 époques sans avis
        return lg
    proba = _proba(3600, [(1000, 1030, 1, 0.95), (2000, 2020, 2, 0.75)])
    r = analyser_nuit_complete(epoques, predire, proba)
    # les 5 époques sans avis sont classées éveil : 57,5 min de sommeil prédit -> 2 événements / 0,958 h = 2,1
    assert r["respiration"]["resume"]["n_evenements"] == 2 and r["respiration"]["resume"]["index_par_heure"] == 2.1
    assert r["file_commune"]["n_items"] == 2                            # un segment de stades douteux + un événement à relire
    assert 0 < r["file_commune"]["part_a_relire_pct"] < 20 and "NON VALIDÉ" in r["statut"]
