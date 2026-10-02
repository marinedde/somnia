"""
Découpage train / validation / test PAR PERSONNE, figé dans un fichier (tâches 0.5 et 0.6).

Règles :
- on découpe des identifiants de personnes, jamais des époques ;
- le tirage est fait une fois, avec une graine, et écrit dans un fichier versionné ;
- un fichier de découpage existant ne se modifie pas : on en crée un nouveau (v2).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

ENSEMBLES = ("train", "val", "test")


def decouper_par_personne(identifiants, graine: int = 42,
                          part_val: float = 0.15, part_test: float = 0.15) -> dict[str, list[str]]:
    """Répartit des identifiants de PERSONNES en trois listes disjointes.

    `sorted(set(...))` : le résultat ne dépend ni des doublons ni de l'ordre des fichiers.
    `random.Random(graine)` : un générateur à part, donc reproductible.
    """
    uniques = sorted(set(map(str, identifiants)))
    if len(uniques) < 3:
        raise ValueError(f"Au moins 3 personnes sont nécessaires, reçu {len(uniques)}")
    random.Random(graine).shuffle(uniques)
    n = len(uniques)
    n_test = max(1, round(n * part_test))
    n_val = max(1, round(n * part_val))
    decoupage = {
        "test": sorted(uniques[:n_test]),
        "val": sorted(uniques[n_test:n_test + n_val]),
        "train": sorted(uniques[n_test + n_val:]),
    }
    verifier_decoupage(decoupage)
    return decoupage


def verifier_decoupage(decoupage: dict) -> None:
    """Lève ValueError si une personne est dans deux ensembles, ou si un ensemble est vide."""
    for nom in ENSEMBLES:
        if nom not in decoupage:
            raise ValueError(f"Ensemble manquant : {nom}")
        if len(decoupage[nom]) == 0:
            raise ValueError(f"Ensemble vide : {nom}")
    vus: set[str] = set()
    for nom in ENSEMBLES:
        courant = set(decoupage[nom])
        commun = vus & courant
        if commun:
            raise ValueError(f"Personnes présentes dans deux ensembles : {sorted(commun)}")
        vus |= courant


def masques(decoupage: dict, personnes: np.ndarray) -> dict[str, np.ndarray]:
    """Pour un tableau `personnes` aligné sur X, renvoie un masque booléen par ensemble."""
    personnes = np.asarray(personnes).astype(str)
    m = {nom: np.isin(personnes, list(decoupage[nom])) for nom in ENSEMBLES}
    inconnues = sorted(set(personnes) - set().union(*(set(decoupage[n]) for n in ENSEMBLES)))
    if inconnues:
        raise ValueError(f"Personnes absentes du fichier de découpage : {inconnues}")
    return m


def sauvegarder_decoupage(chemin: Path, contenu: dict) -> None:
    chemin = Path(chemin)
    if chemin.exists():
        raise FileExistsError(f"{chemin} existe déjà : un découpage ne se modifie pas, crée une v2.")
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def charger_decoupage(chemin: Path) -> dict:
    """Lit le fichier et vérifie chaque tâche ('eeg', 'ecg', …) qu'il contient."""
    contenu = json.loads(Path(chemin).read_text(encoding="utf-8"))
    for tache, dec in contenu.get("taches", {}).items():
        try:
            verifier_decoupage(dec)
        except ValueError as e:
            raise ValueError(f"[{tache}] {e}") from e
    return contenu
