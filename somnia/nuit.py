"""
Analyse d'une nuit entière : hypnogramme, indices, et file de relecture triée par incertitude.

C'est l'idée de départ de Somnia : un médecin met quatre heures à lire une nuit. Le modèle ne
remplace pas cette lecture ; il la trie. Pour chaque époque il donne un stade, une probabilité,
et une confiance calibrée ; les époques les moins sûres forment la file de relecture.

Entrée  : les époques EEG d'une nuit (n, 3000) à 100 Hz, en volts (comme un EDF lu par MNE).
Sortie  : un dictionnaire sérialisable (voir `analyser_nuit`).

Les étiquettes « non validé » sont systématiques : rien ici n'est un diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STADES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
DUREE_EPOQUE_S = 30


@dataclass
class ResultatNuit:
    stades: np.ndarray          # (n,) codes 0-4 prédits
    proba: np.ndarray           # (n, 5) probabilités calibrées
    confiance: np.ndarray       # (n,) max des probabilités

    @property
    def n(self) -> int:
        return len(self.stades)


def indices_de_nuit(stades: np.ndarray, duree_epoque_s: int = DUREE_EPOQUE_S) -> dict:
    """Les chiffres qu'un compte rendu de sommeil donne toujours (sur les stades prédits)."""
    n = len(stades)
    sommeil = np.isin(stades, [1, 2, 3, 4])
    idx = np.flatnonzero(sommeil)
    tst_min = sommeil.sum() * duree_epoque_s / 60
    periode = (idx[-1] - idx[0] + 1) if len(idx) else 0
    return {
        "temps_enregistrement_min": round(n * duree_epoque_s / 60, 1),
        "temps_sommeil_total_min": round(float(tst_min), 1),
        "efficacite_sommeil_pct": round(float(sommeil.mean() * 100), 1) if n else 0.0,
        "latence_endormissement_min": round(float(idx[0] * duree_epoque_s / 60), 1) if len(idx) else None,
        "eveil_intra_sommeil_min": round(float((periode - len(idx)) * duree_epoque_s / 60), 1) if len(idx) else None,
        "parts_pct": {STADES[k]: round(float((stades == k).mean() * 100), 1) for k in STADES},
        "n_epoques": int(n),
    }


def file_de_relecture(confiance: np.ndarray, stades: np.ndarray, seuil: float = 0.6,
                      duree_epoque_s: int = DUREE_EPOQUE_S) -> dict:
    """Les époques sous le seuil de confiance, groupées en segments contigus, triées de la moins sûre à la plus sûre."""
    douteuses = confiance < seuil
    segments, debut = [], None
    for i, d in enumerate(np.append(douteuses, False)):
        if d and debut is None:
            debut = i
        elif not d and debut is not None:
            seg = slice(debut, i)
            segments.append({
                "debut_epoque": int(debut), "fin_epoque": int(i - 1),
                "debut_hhmm": _hhmm(debut * duree_epoque_s), "duree_min": round((i - debut) * duree_epoque_s / 60, 1),
                "confiance_min": round(float(confiance[seg].min()), 3),
                "stades_proposes": "".join(STADES[int(s)] for s in stades[seg][:12]) + ("…" if i - debut > 12 else ""),
            })
            debut = None
    segments.sort(key=lambda s: s["confiance_min"])
    n_douteuses = int(douteuses.sum())
    return {
        "seuil_confiance": seuil,
        "n_epoques_a_relire": n_douteuses,
        "part_a_relire_pct": round(float(douteuses.mean() * 100), 1) if len(confiance) else 0.0,
        "temps_relecture_estime_min": round(n_douteuses * duree_epoque_s / 60, 1),
        "segments": segments,
    }


def _hhmm(secondes: float) -> str:
    h, m = divmod(int(secondes // 60), 60)
    return f"{h:02d}:{m:02d}"


def analyser_nuit(epoques_volts: np.ndarray, predire_proba, temperature: float = 1.0, seuil: float = 0.6,
                  inexploitable: np.ndarray | None = None) -> dict:
    """Tout le compte rendu d'une nuit à partir d'une fonction `predire_proba(epoques) -> logits (n, 5)`.

    `temperature` : celle ajustée sur la validation (calibration). `seuil` : sous cette confiance,
    l'époque part en relecture. `inexploitable` (booléen par époque, voir somnia.qualite) : la
    confiance de ces époques est mise à zéro, elles partent donc en relecture quoi que dise le réseau.
    """
    logits = np.asarray(predire_proba(epoques_volts), dtype=np.float64) / temperature
    z = logits - logits.max(axis=1, keepdims=True)
    proba = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    stades = proba.argmax(axis=1)
    confiance = proba.max(axis=1)
    if inexploitable is not None:
        confiance = np.where(np.asarray(inexploitable, dtype=bool)[: len(confiance)], 0.0, confiance)
    return {
        "statut": "NON VALIDÉ — proposition automatique, à relire par un professionnel formé",
        "hypnogramme": [STADES[int(s)] for s in stades],
        "confiance": [round(float(c), 3) for c in confiance],
        "indices": indices_de_nuit(stades),
        "relecture": file_de_relecture(confiance, stades, seuil),
    }


# ── Événements respiratoires dans la sortie par nuit ──────────────────────
CONFIANCE_SURE = 0.85        # au-dessus : événement proposé comme sûr ; en dessous : à relire
P_POSSIBLE = 0.40            # entre ce niveau et le seuil de décision : « événement possible », à regarder
CONTEXTE_S = 30              # ce qu'un lecteur regarde autour d'un événement à relire
DUREE_POSSIBLE_S = 10        # durée minimale d'une zone « événement possible »
TROU_POSSIBLE_S = 0          # trous comblés à l'intérieur d'une zone « possible »
DESATURATIONS = (3.0, 4.0)   # points de saturation : critères des index cliniques (hypopnées à 3 % et à 4 %)

_CODES = {v: k for k, v in STADES.items()}


def _hhmmss(secondes: float) -> str:
    s = int(secondes)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def analyser_evenements(proba_sec: np.ndarray, sommeil_sec: np.ndarray, seuil: float | None = None,
                        confiance_sure: float = CONFIANCE_SURE, sommeil_seulement: bool = False,
                        sao2_1hz: np.ndarray | None = None, p_possible: float | None = None,
                        duree_possible: int | None = None, trou_possible: int | None = None,
                        inexploitable_sec: np.ndarray | None = None) -> dict:
    """Probabilités à la seconde (n_sec, 3 : rien, apnée, hypopnée) -> événements proposés.

    Chaque événement reçoit une confiance (moyenne de P(événement) sur sa durée). Trois niveaux :
      - sûr          : confiance ≥ `confiance_sure`, à valider d'un coup d'œil ;
      - à relire     : événement proposé mais confiance plus basse ;
      - possible     : pas proposé, mais P(événement) entre 0,4 et le seuil pendant au moins 10 s :
                       c'est là que se cachent les événements manqués.

    Avec `sao2_1hz` (saturation nettoyée, une valeur par seconde), chaque événement reçoit sa
    désaturation associée, et le résumé donne les index CLINIQUES : toutes les apnées, plus les
    hypopnées suivies d'une chute d'au moins 3 (ou 4) points. C'est la définition des index SHHS.

    Avec `inexploitable_sec` (booléen par seconde, voir somnia.qualite) : rien n'est proposé là où la
    respiration est inexploitable, et ce temps sort du dénominateur de l'index. Un index calculé sur
    six heures dont deux sans capteur serait sous-estimé d'un tiers.
    """
    from somnia.resp import CLASSES, SEUIL_EVENEMENT, chute_de_saturation, masque_vers_evenements, proba_vers_masque

    p_possible = P_POSSIBLE if p_possible is None else p_possible
    duree_possible = DUREE_POSSIBLE_S if duree_possible is None else duree_possible
    trou_possible = TROU_POSSIBLE_S if trou_possible is None else trou_possible

    seuil = SEUIL_EVENEMENT if seuil is None else seuil
    proba_sec = np.asarray(proba_sec, dtype=np.float64)
    n = len(proba_sec)
    sommeil_sec = np.asarray(sommeil_sec, dtype=bool)[:n]
    if len(sommeil_sec) < n:
        sommeil_sec = np.concatenate([sommeil_sec, np.zeros(n - len(sommeil_sec), dtype=bool)])
    p_ev = 1.0 - proba_sec[:, 0]
    masque = proba_vers_masque(proba_sec, seuil)
    evenements = []
    for e in masque_vers_evenements(masque):
        conf = float(p_ev[e.debut:e.fin].mean())
        evenements.append({
            "debut_s": e.debut, "debut": _hhmmss(e.debut), "duree_s": e.duree, "type": CLASSES[e.classe],
            "confiance": round(conf, 3), "a_relire": bool(conf < confiance_sure),
            "pendant_le_sommeil": bool(sommeil_sec[e.debut]),
        })
        if sao2_1hz is not None:
            evenements[-1]["desaturation"] = round(chute_de_saturation(sao2_1hz, e.debut, e.fin), 1)
    gris = ((p_ev >= p_possible) & (masque == 0)).astype(np.int8)
    possibles = [{"debut_s": e.debut, "debut": _hhmmss(e.debut), "duree_s": e.duree,
                  "confiance": round(float(p_ev[e.debut:e.fin].mean()), 3)}
                 for e in masque_vers_evenements(gris, duree_min=duree_possible, trou_max=trou_possible)]
    n_ecartes = 0
    if inexploitable_sec is not None:
        mauvais = np.asarray(inexploitable_sec, dtype=bool)[:n]
        mauvais = np.concatenate([mauvais, np.zeros(n - len(mauvais), dtype=bool)])
        # compte ce que l'outil aurait proposé : pendant le sommeil prédit seulement, si c'est la règle
        n_ecartes = sum(bool(mauvais[e["debut_s"]]) and (e["pendant_le_sommeil"] or not sommeil_seulement) for e in evenements)
        evenements = [e for e in evenements if not mauvais[e["debut_s"]]]
        possibles = [e for e in possibles if not mauvais[e["debut_s"]]]
        sommeil_sec = sommeil_sec & ~mauvais
    if sommeil_seulement:
        # 47 % des fausses propositions commençaient pendant l'éveil : un événement respiratoire se
        # marque pendant le sommeil. On ne garde que ce qui commence pendant le sommeil prédit.
        evenements = [e for e in evenements if e["pendant_le_sommeil"]]
        possibles = [e for e in possibles if sommeil_sec[e["debut_s"]]]
    heures = float(sommeil_sec.sum()) / 3600
    en_sommeil = [e for e in evenements if e["pendant_le_sommeil"]]
    cliniques = {}
    if sao2_1hz is not None:
        for k in DESATURATIONS:
            n_k = sum(e["type"] == "apnée" or e["desaturation"] >= k for e in en_sommeil)
            cliniques[f"index_clinique_{k:g}"] = round(n_k / heures, 1) if heures > 0 else None
    return {
        "evenements": evenements,
        "possibles": possibles,
        "resume": {
            "n_evenements": len(evenements),
            "n_apnees": sum(e["type"] == "apnée" for e in evenements),
            "n_hypopnees": sum(e["type"] == "hypopnée" for e in evenements),
            "n_surs": sum(not e["a_relire"] for e in evenements),
            "n_a_relire": sum(e["a_relire"] for e in evenements),
            "n_possibles": len(possibles),
            "heures_de_sommeil": round(heures, 2),
            "index_par_heure": round(len(en_sommeil) / heures, 1) if heures > 0 else None,
            "seuil_decision": seuil, "confiance_sure": confiance_sure, **cliniques,
            "n_ecartes_signal_inexploitable": n_ecartes,
        },
    }


def file_commune(relecture_stades: dict, respiration: dict, n_sec: int, duree_epoque_s: int = DUREE_EPOQUE_S,
                 resp_inexploitable: np.ndarray | None = None, sommeil_sec: np.ndarray | None = None) -> dict:
    """Une seule liste, dans l'ordre de la nuit, de tout ce qu'il faut regarder, et le temps de signal que ça représente.

    Le temps de signal à relire est l'UNION des passages concernés (une époque douteuse et un événement
    au même endroit ne comptent qu'une fois), avec 30 s de contexte autour des événements.
    """
    a_voir = np.zeros(n_sec, dtype=bool)
    items = []
    for s in relecture_stades["segments"]:
        a, b = s["debut_epoque"] * duree_epoque_s, (s["fin_epoque"] + 1) * duree_epoque_s
        a_voir[a:min(b, n_sec)] = True
        items.append({"quoi": "stade", "debut_s": a, "debut": _hhmmss(a), "duree_s": b - a,
                      "confiance": s["confiance_min"], "proposition": s["stades_proposes"]})
    for e in respiration["evenements"]:
        if e["a_relire"]:
            a, b = max(0, e["debut_s"] - CONTEXTE_S), min(n_sec, e["debut_s"] + e["duree_s"] + CONTEXTE_S)
            a_voir[a:b] = True
            items.append({"quoi": "événement", "debut_s": e["debut_s"], "debut": e["debut"], "duree_s": e["duree_s"],
                          "confiance": e["confiance"], "proposition": e["type"]})
    for e in respiration["possibles"]:
        a, b = max(0, e["debut_s"] - CONTEXTE_S), min(n_sec, e["debut_s"] + e["duree_s"] + CONTEXTE_S)
        a_voir[a:b] = True
        items.append({"quoi": "événement possible", "debut_s": e["debut_s"], "debut": e["debut"], "duree_s": e["duree_s"],
                      "confiance": e["confiance"], "proposition": "non proposé"})
    if resp_inexploitable is not None:
        # Là où la respiration est inexploitable pendant le sommeil prédit, l'outil n'a rien proposé : au lecteur de regarder.
        mauvais = np.asarray(resp_inexploitable, dtype=bool)[:n_sec]
        if sommeil_sec is not None:
            mauvais = mauvais & np.asarray(sommeil_sec, dtype=bool)[: len(mauvais)]
        bords = np.diff(np.concatenate([[0], mauvais.astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(bords == 1), np.flatnonzero(bords == -1)):
            a_voir[a:b] = True
            items.append({"quoi": "signal respiratoire inexploitable", "debut_s": int(a), "debut": _hhmmss(a), "duree_s": int(b - a),
                          "confiance": 0.0, "proposition": "non analysé"})
    items.sort(key=lambda x: x["debut_s"])
    return {
        "items": items,
        "n_items": len(items),
        "signal_a_relire_min": round(float(a_voir.sum()) / 60, 1),
        "signal_total_min": round(n_sec / 60, 1),
        "part_a_relire_pct": round(float(a_voir.mean()) * 100, 1) if n_sec else 0.0,
    }


def analyser_nuit_complete(epoques_eeg: np.ndarray, predire_stades, proba_evenements_sec: np.ndarray,
                           temperature: float = 1.0, seuil_stade: float = 0.6, sao2_1hz: np.ndarray | None = None,
                           qualite: dict | None = None) -> dict:
    """Hypnogramme + événements respiratoires + une file de relecture commune.

    Le sommeil utilisé pour l'index est le sommeil PRÉDIT (pas celui du technicien) : c'est ce
    dont disposerait un outil qui reçoit une nuit non scorée.

    `qualite` (facultatif) : {"eeg_inexploitable": booléen par époque, "resp_inexploitable": booléen
    par seconde, "rapport": somnia.qualite.rapport_qualite(...)}. Les passages inexploitables partent
    en relecture ; si la nuit est refusée, les indices ou l'index sont retirés et le motif est donné.
    """
    q = qualite or {}
    rapport = analyser_nuit(epoques_eeg, predire_stades, temperature, seuil_stade, inexploitable=q.get("eeg_inexploitable"))
    stades = np.array([_CODES[s] for s in rapport["hypnogramme"]])
    n_sec = len(proba_evenements_sec)
    sommeil_sec = np.repeat(np.isin(stades, [1, 2, 3, 4]), DUREE_EPOQUE_S)
    rapport["respiration"] = analyser_evenements(proba_evenements_sec, sommeil_sec, sommeil_seulement=True, sao2_1hz=sao2_1hz,
                                                 inexploitable_sec=q.get("resp_inexploitable"))
    rapport["file_commune"] = file_commune(rapport["relecture"], rapport["respiration"], n_sec,
                                           resp_inexploitable=q.get("resp_inexploitable"), sommeil_sec=sommeil_sec)
    if "rapport" in q:
        rapport["qualite"] = q["rapport"]
        if q["rapport"]["refus"]["stades"]:
            rapport["indices"] = None
        if q["rapport"]["refus"]["index"]:
            for cle in [c for c in rapport["respiration"]["resume"] if c.startswith("index_")]:
                rapport["respiration"]["resume"][cle] = None
        if q["rapport"]["motifs"]:
            rapport["statut"] += " | QUALITÉ DU SIGNAL : " + " ; ".join(q["rapport"]["motifs"])
    return rapport
