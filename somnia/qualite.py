"""
Qualité du signal et refus (horizon 1.5).

Un outil d'aide à la lecture doit savoir dire « ici, je ne peux pas » : une électrode décollée,
une ceinture débranchée, un oxymètre tombé du doigt. Sans cela, le réseau rend quand même un
stade ou un événement, avec parfois une confiance élevée, sur un signal qui ne contient rien.

Trois défauts, repérés par époque de 30 s et par capteur, avec des règles simples et lisibles :
  - plat      : l'écart-type de l'époque est inférieur à 5 % de l'écart-type médian de la nuit
                (capteur débranché, électrode décollée) ;
  - écrêté    : plus de 5 % des points de l'époque sont collés à l'extrême de la nuit
                (amplificateur saturé : le haut des ondes est coupé) ;
  - artefact  : l'écart-type de l'époque dépasse 6 fois la médiane de la nuit (mouvement, câble
                tiré). Règle appliquée aux capteurs de la tête seulement : une grande respiration
                après une apnée est un signal, pas un artefact.
La saturation a sa règle propre : valeurs hors de 50–100 % (capteur décollé).

Tout est relatif à la nuit elle-même : aucun seuil en microvolts, donc rien à régler par appareil.
Les seuils ci-dessous ont été écrits AVANT de regarder les nuits de validation.

Ce qui rend un passage « inexploitable » a été RÉVISÉ après la première mesure sur la validation
(docs/RESULTATS_QUALITE.md, models/qualite_v1.json) ; les seuils, eux, n'ont pas bougé :
  - tête : plat ou écrêté. L'artefact n'est plus qu'une information (« mouvement ») : 848 des 850
    époques d'EEG ainsi signalées étaient de l'éveil, que le réseau reconnaît très bien ;
  - flux : plat seulement. Un flux écrêté reste lisible (propositions justes à 75 %, contre 79 %) ;
  - ceintures : plat ou écrêté, et il faut que les DEUX le soient (propositions justes à 51 %) ;
  - saturation : invalide (propositions justes à 39 %).
C'est un choix fait sur les 40 personnes de validation : à confirmer sur d'autres nuits.

Le refus, à deux niveaux :
  - par passage : une époque dont l'EEG est inexploitable part en relecture quelle que soit la
    confiance du réseau ; un événement respiratoire n'est pas proposé là où la respiration est
    inexploitable, et ce temps est retiré du dénominateur de l'index ;
  - par nuit : au-delà d'un quart de la nuit inexploitable, l'outil ne rend pas d'hypnogramme
    résumé (ou pas d'index) et le dit.
"""

from __future__ import annotations

import numpy as np

PLAT_RELATIF = 0.05            # écart-type < 5 % de la médiane de la nuit
ECRETE_PART = 0.05             # > 5 % des points à l'extrême de la nuit
ECRETE_MARGE = 0.99            # « à l'extrême » : au-delà de 99 % du maximum absolu de la nuit
ARTEFACT_RELATIF = 6.0         # écart-type > 6 fois la médiane de la nuit
SAO2_INVALIDE_PART = 0.5       # plus de la moitié des secondes de l'époque hors de 50–100 %
REFUS_NUIT_PART = 0.25         # au-delà : pas d'indices de sommeil / pas d'index respiratoire
EPOQUE_S = 30
CAUSES = ("plat", "ecrete", "artefact")


def defauts_par_epoque(x: np.ndarray, causes: tuple[str, ...] = ("plat", "ecrete")) -> dict[str, np.ndarray]:
    """(n_epoques, n_points) d'un capteur -> un masque booléen par défaut (plat, écrêté, artefact),
    plus `inexploitable` : l'union des défauts nommés dans `causes`."""
    x = np.asarray(x, dtype=np.float32)
    et = x.std(axis=1)
    mediane = float(np.median(et)) if len(et) else 0.0
    if mediane <= 1e-9:                                   # capteur mort toute la nuit
        plat = np.ones(len(x), dtype=bool)
        vide = np.zeros(len(x), dtype=bool)
        return {"plat": plat, "ecrete": vide, "artefact": vide.copy(), "inexploitable": plat.copy()}
    plat = et < PLAT_RELATIF * mediane
    extreme = float(np.abs(x).max())
    ecrete = (np.abs(x) >= ECRETE_MARGE * extreme).mean(axis=1) > ECRETE_PART
    d = {"plat": plat, "ecrete": ecrete, "artefact": et > ARTEFACT_RELATIF * mediane}
    d["inexploitable"] = np.logical_or.reduce([d[c] for c in causes]) if causes else np.zeros(len(x), dtype=bool)
    return d


def qualite_tete(signaux: np.ndarray, noms: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    """(n_epoques, C, 3000) -> {capteur: défauts par époque} pour l'EEG, les yeux, le menton."""
    return {nom: defauts_par_epoque(signaux[:, c]) for c, nom in enumerate(noms)}


def en_epoques(x: np.ndarray, points_par_epoque: int) -> np.ndarray:
    """Signal continu -> (n_epoques, points) ; la fin incomplète est ignorée."""
    n = len(x) // points_par_epoque
    return np.asarray(x[: n * points_par_epoque]).reshape(n, points_par_epoque)


def qualite_respiration(signaux: np.ndarray, fs: int = 10, sao2_invalide: np.ndarray | None = None) -> dict:
    """(4, n_sec * fs) flux, thorax, abdomen, saturation -> défauts par époque et par capteur.

    `sao2_invalide` : booléen par seconde, relevé AVANT le nettoyage de la saturation (le nettoyage
    remplace les valeurs absurdes par la dernière valeur valide : après lui, on ne voit plus rien).
    `inexploitable` : pas de flux, ou les deux ceintures perdues, ou pas de saturation.
    """
    out = {nom: defauts_par_epoque(en_epoques(signaux[c], EPOQUE_S * fs), causes=("plat",) if nom == "flux" else ("plat", "ecrete"))
           for c, nom in enumerate(("flux", "thorax", "abdomen"))}
    n = len(out["flux"]["plat"])
    if sao2_invalide is None:
        sao2 = np.zeros(n, dtype=bool)
    else:
        e = en_epoques(np.asarray(sao2_invalide, dtype=np.float32), EPOQUE_S)[:n]
        sao2 = np.concatenate([e.mean(axis=1) > SAO2_INVALIDE_PART, np.zeros(n - len(e), dtype=bool)])
    out["sao2"] = {"invalide": sao2, "inexploitable": sao2}
    out["inexploitable"] = (out["flux"]["inexploitable"] | (out["thorax"]["inexploitable"] & out["abdomen"]["inexploitable"]) | sao2)
    return out


def par_seconde(masque_epoques: np.ndarray, n_sec: int) -> np.ndarray:
    """Masque par époque -> masque par seconde, complété par « exploitable » si la nuit est plus longue."""
    m = np.repeat(np.asarray(masque_epoques, dtype=bool), EPOQUE_S)[:n_sec]
    return np.concatenate([m, np.zeros(n_sec - len(m), dtype=bool)])


def rapport_qualite(tete: dict | None = None, respiration: dict | None = None, capteur_stades: str = "EEG") -> dict:
    """Résumé lisible d'une nuit, et la décision de refus.

    Les stades dépendent du capteur `capteur_stades` (l'EEG principal) : les yeux et le menton
    aident, mais leur perte ne rend pas la nuit illisible ; elle est signalée.
    """
    r: dict = {"capteurs": {}, "refus": {"stades": False, "index": False}, "motifs": [], "modele_stades": "eeg_eog_emg"}
    for nom, d in (tete or {}).items():
        r["capteurs"][nom] = {"inexploitable_pct": round(float(d["inexploitable"].mean()) * 100, 1),
                              **{c: round(float(d[c].mean()) * 100, 1) for c in CAUSES}}
    if tete and capteur_stades in tete:
        part = float(tete[capteur_stades]["inexploitable"].mean())
        if part > REFUS_NUIT_PART:
            r["refus"]["stades"] = True
            r["motifs"].append(f"{capteur_stades} inexploitable sur {part:.0%} de la nuit : pas d'indices de sommeil")
        for nom, d in tete.items():
            if nom != capteur_stades and d["inexploitable"].mean() > REFUS_NUIT_PART:
                # Un capteur débranché fait plus de mal que son absence : mieux vaut le modèle qui ne le lit pas.
                r["modele_stades"] = "eeg"
                r["motifs"].append(f"{nom} inexploitable sur {d['inexploitable'].mean():.0%} de la nuit : stades calculés avec l'EEG seul")
    if respiration:
        for nom in ("flux", "thorax", "abdomen", "sao2"):
            r["capteurs"][nom] = {"inexploitable_pct": round(float(respiration[nom]["inexploitable"].mean()) * 100, 1)}
        part = float(respiration["inexploitable"].mean())
        r["respiration_inexploitable_pct"] = round(part * 100, 1)
        if part > REFUS_NUIT_PART:
            r["refus"]["index"] = True
            r["motifs"].append(f"respiration inexploitable sur {part:.0%} de la nuit : pas d'index")
    return r


def qualite_nuit(signaux_tete: np.ndarray, noms_tete: tuple[str, ...], signaux_resp: np.ndarray, n_sec: int,
                 sao2_invalide: np.ndarray | None = None, fs_resp: int = 10) -> dict:
    """Qualité complète d'une nuit, au format attendu par `somnia.nuit.analyser_nuit_complete(qualite=...)`."""
    tete = qualite_tete(signaux_tete, noms_tete)
    resp = qualite_respiration(signaux_resp, fs_resp, sao2_invalide)
    return {"tete": tete, "resp": resp, "eeg_inexploitable": tete[noms_tete[0]]["inexploitable"],
            "resp_inexploitable": par_seconde(resp["inexploitable"], n_sec),
            "rapport": rapport_qualite(tete, resp, capteur_stades=noms_tete[0])}
