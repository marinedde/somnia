"""
Identifiant de la PERSONNE derrière un nom de fichier (tâche 0.1).

Pourquoi : le découpage train / validation / test se fait par personne, jamais
par enregistrement. Or chaque jeu de données cache des personnes qui ont
plusieurs enregistrements :

- Sleep-EDF   : deux nuits par personne. ``SC4011E0`` et ``SC4012E0`` sont la
                personne 01, nuits 1 et 2. Le numéro de personne occupe les
                caractères 3 et 4 (``SC4ssNX``).
- Apnea-ECG   : PhysioNet ne publie pas la correspondance enregistrement →
                personne. Seul cas documenté : ``c06`` est le même enregistrement
                que ``c05`` décalé de 80 s. On découpe donc par enregistrement,
                en fusionnant ce doublon.
- SHHS        : la visite 2 (``shhs2-200001``) porte sur la même personne que la
                visite 1 (``shhs1-200001``). L'identifiant est le numéro ``nsrrid``.

Usage :
    >>> identifiant_personne("SC4012E0-PSG.edf")
    'SC01'
    >>> identifiant_personne("a05")
    'a05'
    >>> identifiant_personne("shhs2-200001.edf")
    'shhs-200001'
"""

from __future__ import annotations

import re
from pathlib import Path

# Apnea-ECG : enregistrements connus pour être la même personne.
# Source : page PhysioNet apnea-ecg 1.0.0, note d'avril 2013 (c05 / c06).
APNEA_ECG_DOUBLONS = {"c06": "c05"}

_SLEEP_EDF = re.compile(r"^(SC|ST)(\d)(\d{2})(\d)([A-Z0-9])")
_APNEA_ECG = re.compile(r"^([abcx]\d{2})$")
_SHHS = re.compile(r"^shhs([12])-(\d{6})")


def _base(nom_fichier: str | Path) -> str:
    """'data/raw/SC4012E0-PSG.edf' -> 'SC4012E0-PSG' ; 'a05.dat' -> 'a05'."""
    nom = Path(nom_fichier).name
    # Path.stem ne retire qu'un suffixe : 'x.edf.gz' -> 'x.edf'. On coupe au premier point.
    return nom.split(".")[0]


def identifiant_personne(nom_fichier: str | Path) -> str:
    """Renvoie un identifiant stable de la personne pour un fichier de signal.

    Lève ``ValueError`` si le nom ne correspond à aucun jeu de données connu :
    mieux vaut planter que de laisser passer un enregistrement sans groupe.
    """
    base = _base(nom_fichier)

    m = _SLEEP_EDF.match(base)
    if m:
        cohorte, _etude, personne, _nuit, _ = m.groups()
        return f"{cohorte}{personne}"

    m = _APNEA_ECG.match(base)
    if m:
        enregistrement = m.group(1)
        return APNEA_ECG_DOUBLONS.get(enregistrement, enregistrement)

    m = _SHHS.match(base)
    if m:
        _visite, nsrrid = m.groups()
        return f"shhs-{nsrrid}"

    raise ValueError(f"Nom de fichier inconnu, impossible d'en déduire la personne : {nom_fichier!r}")


def groupes_disjoints(*ensembles: set[str] | list[str]) -> bool:
    """Vrai si aucune personne n'apparaît dans deux ensembles (train / val / test)."""
    vus: set[str] = set()
    for ens in ensembles:
        ens = set(ens)
        if not vus.isdisjoint(ens):
            return False
        vus |= ens
    return True
