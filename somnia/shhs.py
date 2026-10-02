"""
Lecture des annotations SHHS (fichiers XML « annotations-events-nsrr »), tâche 1.8.

Structure attendue d'un fichier (à CONFIRMER sur un vrai fichier, tâche 1.7) :

    <PSGAnnotation>
      <EpochLength>30</EpochLength>
      <ScoredEvents>
        <ScoredEvent>
          <EventType>Stages|Stages</EventType>
          <EventConcept>Stage 2 sleep|2</EventConcept>
          <Start>600</Start>
          <Duration>30</Duration>
        </ScoredEvent>
        <ScoredEvent>
          <EventType>Respiratory|Respiratory</EventType>
          <EventConcept>Obstructive apnea|Obstructive Apnea</EventConcept>
          <Start>1234.5</Start>
          <Duration>18.2</Duration>
          <SignalLocation>ABDO RES</SignalLocation>
        </ScoredEvent>
        …

Trois décisions d'étiquetage, prises le 2 octobre 2026 et modifiables par paramètre :
  1. Événements comptés comme « apnée » : apnées obstructives, centrales, mixtes ET hypopnées
     (comme l'index d'apnées-hypopnées clinique).
  2. Une époque est positive si un événement couvre AU MOINS 10 s de ses 30 s.
  3. Les époques d'éveil sont exclues de la tâche apnée (une apnée se définit pendant le sommeil) :
     c'est fait en aval, avec `stades`, pas ici.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Stades SHHS (règles R&K, codés 0-5) -> 5 classes AASM, mêmes codes que Sleep-EDF dans Somnia
#   0 éveil, 1-2 sommeil léger, 3-4 sommeil profond (fusionnés en N3), 5 REM
VERS_SOMNIA = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}
STADE_INCONNU = -1   # non scoré, mouvement, code inattendu : à exclure

# Événements respiratoires qui comptent pour l'étiquette « apnée », par défaut
EVENEMENTS_APNEE = ("obstructive apnea", "central apnea", "mixed apnea", "hypopnea")


@dataclass(frozen=True)
class Evenement:
    nom: str        # 'Obstructive apnea', 'Hypopnea', 'SpO2 desaturation', 'Arousal' …
    type: str       # 'Respiratory', 'Arousals', 'Limb Movement' …
    debut: float    # secondes depuis le début de l'enregistrement
    duree: float    # secondes

    @property
    def fin(self) -> float:
        return self.debut + self.duree


@dataclass
class Annotations:
    duree_epoque: float
    stades: np.ndarray          # (n_epoques,) codes Somnia 0-4, ou -1
    evenements: list[Evenement]

    @property
    def n_epoques(self) -> int:
        return len(self.stades)

    def respiratoires(self) -> list[Evenement]:
        return [e for e in self.evenements if e.type.lower().startswith("respiratory")]


def _premier(texte: str | None) -> str:
    """'Stage 2 sleep|2' -> 'Stage 2 sleep' ; 'Respiratory|Respiratory' -> 'Respiratory'."""
    return (texte or "").split("|")[0].strip()


def _code_stade(concept: str) -> int:
    """'Stage 2 sleep|2' -> 2 ; 'Wake|0' -> 0 ; 'REM sleep|5' -> 5 ; sinon -1."""
    parts = (concept or "").split("|")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            return STADE_INCONNU
    return STADE_INCONNU


def lire_annotations(chemin_xml: str | Path) -> Annotations:
    """Lit un XML NSRR : un stade par époque + la liste des événements horodatés."""
    racine = ET.parse(str(chemin_xml)).getroot()
    duree_epoque = float(racine.findtext("EpochLength") or 30)

    stades: list[int] = []
    evenements: list[Evenement] = []
    fin_stades = 0.0
    for ev in racine.iter("ScoredEvent"):
        type_ev = ev.findtext("EventType") or ""
        concept = ev.findtext("EventConcept") or ""
        texte_debut, texte_duree = ev.findtext("Start"), ev.findtext("Duration")
        if type_ev.lower().startswith("stages"):
            if texte_debut is None or texte_duree is None:
                raise ValueError(f"{chemin_xml}: bloc de stade sans Start ou Duration ({concept!r})")
            debut, duree = float(texte_debut), float(texte_duree)
            code = _code_stade(concept)
            stade = VERS_SOMNIA.get(code, STADE_INCONNU)
            n = int(round(duree / duree_epoque))
            # Les stades sont donnés en blocs contigus. Un trou est comblé par « inconnu » ;
            # un chevauchement décalerait tous les stades suivants par rapport aux événements
            # (indexés en temps absolu) : on refuse le fichier plutôt que de désaligner.
            attendu = len(stades) * duree_epoque
            if debut < attendu - duree_epoque / 2:
                raise ValueError(f"{chemin_xml}: blocs de stades qui se chevauchent "
                                 f"(début {debut} s, attendu {attendu} s)")
            if debut > attendu + duree_epoque / 2:
                stades.extend([STADE_INCONNU] * int(round((debut - attendu) / duree_epoque)))
            stades.extend([stade] * n)
            fin_stades = debut + duree
        else:
            debut = float(texte_debut) if texte_debut is not None else 0.0
            duree = float(texte_duree) if texte_duree is not None else 0.0
            evenements.append(Evenement(_premier(concept), _premier(type_ev), debut, duree))
    if stades and abs(len(stades) * duree_epoque - fin_stades) > duree_epoque / 2:
        raise ValueError(f"{chemin_xml}: {len(stades)} époques reconstruites mais les stades finissent à {fin_stades} s")
    return Annotations(duree_epoque, np.asarray(stades, dtype=np.int64), evenements)


def etiquettes_apnee(annotations: Annotations,
                     evenements_comptes: tuple[str, ...] = EVENEMENTS_APNEE,
                     recouvrement_min_s: float = 10.0) -> np.ndarray:
    """1 si l'époque est couverte au moins `recouvrement_min_s` secondes par un événement compté, sinon 0.

    Le recouvrement est cumulé sur l'époque : deux hypopnées de 6 s dans la même époque comptent 12 s.
    """
    d = annotations.duree_epoque
    n = annotations.n_epoques
    couverture = np.zeros(n, dtype=np.float64)
    comptes = {e.lower() for e in evenements_comptes}
    for ev in annotations.evenements:
        if ev.nom.lower() not in comptes:
            continue
        premiere = int(ev.debut // d)
        derniere = int(np.ceil(ev.fin / d)) - 1
        for i in range(max(premiere, 0), min(derniere, n - 1) + 1):
            debut_epoque, fin_epoque = i * d, (i + 1) * d
            couverture[i] += max(0.0, min(ev.fin, fin_epoque) - max(ev.debut, debut_epoque))
    return (couverture >= recouvrement_min_s).astype(np.int64)


def resume(annotations: Annotations) -> dict:
    """Comptages pour le tableau du pilote (tâche 1.10)."""
    from collections import Counter

    noms = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM", -1: "Inconnu"}
    stades = Counter(annotations.stades.tolist())
    resp = Counter(e.nom for e in annotations.respiratoires())
    d = annotations.duree_epoque
    n_sommeil = int(np.isin(annotations.stades, [1, 2, 3, 4]).sum())
    temps_sommeil_h = n_sommeil * d / 3600
    n_ah = sum(v for k, v in resp.items() if k.lower() in EVENEMENTS_APNEE)
    return {
        "duree_h": round(annotations.n_epoques * d / 3600, 2),
        "temps_sommeil_h": round(temps_sommeil_h, 2),
        **{f"n_{noms[k]}": stades.get(k, 0) for k in (0, 1, 2, 3, 4, -1)},
        **{f"n_{k}": v for k, v in sorted(resp.items())},
        "n_apnees_hypopnees": n_ah,
        "iah_estime": round(n_ah / temps_sommeil_h, 1) if temps_sommeil_h > 0 else None,
    }
