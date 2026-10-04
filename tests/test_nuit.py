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


# ── Désaturation associée et index cliniques (horizon 1.4) ────────────────
def test_chute_de_saturation_regarde_apres_l_evenement():
    from somnia.resp import chute_de_saturation
    sao2 = np.full(600, 96.0)
    sao2[130:150] = 91.0                                   # la chute arrive après la fin de l'événement (100-120)
    assert chute_de_saturation(sao2, 100, 120) == 5.0
    assert chute_de_saturation(sao2, 300, 320) == 0.0      # rien autour
    assert chute_de_saturation(sao2, 598, 610) == 0.0      # fenêtre hors de la nuit : 0, pas d'erreur


def test_index_cliniques_apnees_toujours_hypopnees_si_desaturation():
    proba = np.zeros((3600, 3)); proba[:, 0] = 1.0
    for a, c in ((100, 1), (500, 2), (900, 2)):            # une apnée, deux hypopnées de 20 s
        proba[a:a + 20] = 0.0; proba[a:a + 20, c] = 1.0
    sao2 = np.full(3600, 96.0)
    sao2[525:540] = 92.5                                   # première hypopnée : chute de 3,5 points
    r = analyser_evenements(proba, np.ones(3600, dtype=bool), sao2_1hz=sao2)
    assert [e["desaturation"] for e in r["evenements"]] == [0.0, 3.5, 0.0]
    res = r["resume"]
    assert res["index_par_heure"] == 3.0                   # les trois événements
    assert res["index_clinique_3"] == 2.0                  # apnée + hypopnée avec chute ≥ 3
    assert res["index_clinique_4"] == 1.0                  # apnée seule : 3,5 < 4
    assert "index_clinique_3" not in analyser_evenements(proba, np.ones(3600, dtype=bool))["resume"]


def test_zones_possibles_reglables():
    proba = np.zeros((3600, 3)); proba[:, 0] = 1.0
    proba[200:206, 0], proba[200:206, 2] = 0.5, 0.5         # 6 s à P = 0,5 : trop bref pour la règle des 10 s
    sommeil = np.ones(3600, dtype=bool)
    assert analyser_evenements(proba, sommeil)["possibles"] == []
    assert len(analyser_evenements(proba, sommeil, duree_possible=5)["possibles"]) == 1


# ── Qualité du signal et refus (horizon 1.5) ──────────────────────────────
def test_epoque_inexploitable_part_en_relecture_malgre_la_confiance():
    from somnia.nuit import analyser_nuit
    logits = np.zeros((10, 5)); logits[:, 2] = 20.0                    # réseau très sûr partout
    mauvais = np.zeros(10, dtype=bool); mauvais[4] = True
    r = analyser_nuit(None, lambda _: logits, inexploitable=mauvais)
    assert r["relecture"]["n_epoques_a_relire"] == 1 and r["confiance"][4] == 0.0
    assert analyser_nuit(None, lambda _: logits)["relecture"]["n_epoques_a_relire"] == 0


def test_evenement_ecarte_et_index_sur_le_temps_analysable():
    proba = np.zeros((7200, 3)); proba[:, 0] = 1.0
    for a in (100, 4000):
        proba[a:a + 20] = 0.0; proba[a:a + 20, 1] = 1.0
    sommeil = np.ones(7200, dtype=bool)
    mauvais = np.zeros(7200, dtype=bool); mauvais[3600:] = True       # la seconde heure : capteur perdu
    r = analyser_evenements(proba, sommeil, inexploitable_sec=mauvais)["resume"]
    assert (r["n_evenements"], r["n_ecartes_signal_inexploitable"]) == (1, 1)
    assert r["heures_de_sommeil"] == 1.0 and r["index_par_heure"] == 1.0   # 1 événement sur 1 h analysable, pas sur 2 h


def test_nuit_refusee_pas_d_index_et_motif_dans_le_statut():
    logits = np.zeros((20, 5)); logits[:, 2] = 20.0
    proba = np.zeros((600, 3)); proba[:, 0] = 1.0
    q = {"eeg_inexploitable": np.zeros(20, dtype=bool), "resp_inexploitable": np.r_[np.ones(300, dtype=bool), np.zeros(300, dtype=bool)],
         "rapport": {"refus": {"stades": False, "index": True}, "motifs": ["respiration inexploitable sur 50 % de la nuit : pas d'index"]}}
    r = analyser_nuit_complete(None, lambda _: logits, proba, sao2_1hz=np.full(600, 96.0), qualite=q)
    assert r["respiration"]["resume"]["index_par_heure"] is None and r["respiration"]["resume"]["index_clinique_3"] is None
    assert "QUALITÉ DU SIGNAL" in r["statut"] and r["indices"] is not None
    item = [i for i in r["file_commune"]["items"] if i["quoi"] == "signal respiratoire inexploitable"]
    assert len(item) == 1 and item[0]["duree_s"] == 300


# ── Position et désaturations dans la sortie (horizon 2) ──────────────────
def test_index_par_position_et_apnee_positionnelle():
    proba = np.zeros((7200, 3)); proba[:, 0] = 1.0
    for a in (100, 400, 700, 1000, 1300, 1600, 4000):      # 6 apnées la première heure (sur le dos), 1 la seconde
        proba[a:a + 20] = 0.0; proba[a:a + 20, 1] = 1.0
    dos = np.zeros(7200, dtype=bool); dos[:3600] = True
    r = analyser_evenements(proba, np.ones(7200, dtype=bool), sao2_1hz=np.full(7200, 96.0), dorsal_sec=dos)
    res = r["resume"]
    assert (res["index_dorsal"], res["index_non_dorsal"], res["sommeil_dorsal_pct"]) == (6.0, 1.0, 50.0)
    assert res["positionnel"] is True and r["evenements"][0]["position"] == "dos" and r["evenements"][-1]["position"] == "autre"


def test_pas_d_index_par_position_sans_assez_de_sommeil():
    proba = np.zeros((7200, 3)); proba[:, 0] = 1.0
    dos = np.zeros(7200, dtype=bool); dos[:600] = True     # 10 minutes sur le dos : trop peu pour un taux
    res = analyser_evenements(proba, np.ones(7200, dtype=bool), dorsal_sec=dos)["resume"]
    assert res["index_dorsal"] is None and res["index_non_dorsal"] == 0.0 and res["positionnel"] is None


def test_desaturations_par_heure_dans_le_resume():
    proba = np.zeros((3600, 3)); proba[:, 0] = 1.0
    sao2 = np.full(3600, 96.0); sao2[500:520] = 92.5; sao2[2000:2030] = 90.0
    res = analyser_evenements(proba, np.ones(3600, dtype=bool), sao2_1hz=sao2)["resume"]
    assert res["desaturations_par_heure_3"] == 2.0 and res["desaturations_par_heure_4"] == 1.0
