"""
Export de la sortie par nuit vers des formats que d'autres logiciels savent ouvrir (horizon 2.8).

L'idée de départ de Somnia est de se brancher sur les logiciels de lecture existants, par fichiers.
Trois formats, du plus universel au plus simple :
  - EDF+ « annotations seules » : le standard européen ; un fichier .edf sans signal, qui ne porte
    que les annotations horodatées. EDFbrowser et les logiciels qui lisent l'EDF+ l'ouvrent en
    superposition de l'enregistrement d'origine. Les stades utilisent les libellés du standard
    (« Sleep stage W », « Sleep stage N2 »...).
  - XML NSRR : le format des annotations de SHHS, MESA, MrOS... relu par `somnia.shhs.lire_annotations`.
  - CSV : début, durée, type, confiance ; pour un tableur ou un import texte.

Chaque export porte la mention « non validé » : ce sont des propositions, pas un scorage.
Les apnées sont exportées sans type (« Apnea ») : Somnia ne distingue pas obstructive et centrale
(docs/RESULTATS_TYPE_APNEE.md).
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

MENTION = "Somnia - propositions automatiques NON VALIDEES"
DUREE_EPOQUE_S = 30
STADES_EDF = {"W": "Sleep stage W", "N1": "Sleep stage N1", "N2": "Sleep stage N2", "N3": "Sleep stage N3", "R": "Sleep stage R"}
STADES_NSRR = {"W": "Wake|0", "N1": "Stage 1 sleep|1", "N2": "Stage 2 sleep|2", "N3": "Stage 3 sleep|3", "R": "REM sleep|5"}
TYPES = {"apnée": "Apnea", "hypopnée": "Hypopnea"}


def lignes_d_annotation(rapport: dict, duree_epoque_s: int = DUREE_EPOQUE_S) -> list[dict]:
    """La sortie de `analyser_nuit_complete` mise à plat : une ligne par stade d'époque et par événement, triées par début.

    Chaque ligne : debut_s, duree_s, famille ('stade', 'respiration', 'micro-éveil'), libelle (code de stade ou type), confiance.
    """
    L = [{"debut_s": float(i * duree_epoque_s), "duree_s": float(duree_epoque_s), "famille": "stade", "libelle": s, "confiance": c}
         for i, (s, c) in enumerate(zip(rapport["hypnogramme"], rapport["confiance"]))]
    for e in rapport.get("respiration", {}).get("evenements", []):
        L.append({"debut_s": float(e["debut_s"]), "duree_s": float(e["duree_s"]), "famille": "respiration", "libelle": e["type"], "confiance": e["confiance"]})
    for e in rapport.get("micro_eveils", {}).get("evenements", []):
        L.append({"debut_s": float(e["debut_s"]), "duree_s": float(e["duree_s"]), "famille": "micro-éveil", "libelle": "micro-éveil", "confiance": e["confiance"]})
    return sorted(L, key=lambda x: (x["debut_s"], x["famille"] != "stade"))


def _texte_edf(l: dict) -> str:
    if l["famille"] == "stade":
        return STADES_EDF[l["libelle"]]
    return TYPES[l["libelle"]] if l["famille"] == "respiration" else "Arousal"


def vers_csv(rapport: dict, chemin: str | Path) -> Path:
    chemin = Path(chemin)
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# {MENTION}"])
        w.writerow(["debut_s", "duree_s", "famille", "libelle", "confiance"])
        for l in lignes_d_annotation(rapport):
            w.writerow([f"{l['debut_s']:g}", f"{l['duree_s']:g}", l["famille"], l["libelle"], l["confiance"]])
    return chemin


def vers_xml_nsrr(rapport: dict, chemin: str | Path, duree_epoque_s: int = DUREE_EPOQUE_S) -> Path:
    """Même structure que les annotations NSRR : stades en blocs contigus, puis événements horodatés."""
    racine = ET.Element("PSGAnnotation")
    ET.SubElement(racine, "SoftwareVersion").text = MENTION
    ET.SubElement(racine, "EpochLength").text = str(duree_epoque_s)
    evs = ET.SubElement(racine, "ScoredEvents")

    def ajouter(type_ev, concept, debut, duree, confiance=None):
        e = ET.SubElement(evs, "ScoredEvent")
        ET.SubElement(e, "EventType").text = type_ev
        ET.SubElement(e, "EventConcept").text = concept
        ET.SubElement(e, "Start").text = f"{debut:g}"
        ET.SubElement(e, "Duration").text = f"{duree:g}"
        if confiance is not None:
            ET.SubElement(e, "SomniaConfidence").text = f"{confiance:g}"

    hyp = rapport["hypnogramme"]
    i = 0
    while i < len(hyp):                                    # blocs contigus de même stade, comme dans les fichiers NSRR
        j = i
        while j < len(hyp) and hyp[j] == hyp[i]:
            j += 1
        ajouter("Stages|Stages", STADES_NSRR[hyp[i]], i * duree_epoque_s, (j - i) * duree_epoque_s)
        i = j
    for l in lignes_d_annotation(rapport, duree_epoque_s):
        if l["famille"] == "respiration":
            nom = TYPES[l["libelle"]]
            ajouter("Respiratory|Respiratory", f"{nom}|{nom}", l["debut_s"], l["duree_s"], l["confiance"])
        elif l["famille"] == "micro-éveil":
            ajouter("Arousals|Arousals", "Arousal|Arousal ()", l["debut_s"], l["duree_s"], l["confiance"])
    chemin = Path(chemin)
    ET.indent(racine)
    ET.ElementTree(racine).write(chemin, encoding="utf-8", xml_declaration=True)
    return chemin


def vers_edf_plus(rapport: dict, chemin: str | Path, debut: datetime | None = None, duree_epoque_s: int = DUREE_EPOQUE_S) -> Path:
    """Fichier EDF+ « annotations seules » : un en-tête, un seul signal « EDF Annotations », aucune donnée de capteur.

    `debut` : date et heure de début de l'enregistrement d'origine, pour que les annotations se
    superposent au bon endroit. Sans elle, une date neutre (01.01.85) : le fichier reste valide,
    les temps sont relatifs au début. Aucune identité de patient n'est écrite.
    """
    debut = debut or datetime(1985, 1, 1)
    lignes = lignes_d_annotation(rapport, duree_epoque_s)
    duree_totale = max(int(len(rapport["hypnogramme"]) * duree_epoque_s), 1)
    # Listes d'annotations horodatées (TAL) : « +début [0x15 durée] 0x14 texte 0x14 0x00 ». La première garde le temps de l'enregistrement.
    tal = b"+0\x14\x14\x00"
    for l in lignes:
        tal += f"+{l['debut_s']:g}\x15{l['duree_s']:g}\x14{_texte_edf(l)}\x14".encode("utf-8") + b"\x00"
    n_echantillons = (len(tal) + 1) // 2 + 1               # échantillons de 2 octets ; au moins un octet nul de fin
    donnees = tal.ljust(n_echantillons * 2, b"\x00")

    def champ(texte: str, taille: int) -> bytes:
        b = texte.encode("ascii", errors="replace")
        if len(b) > taille:
            raise ValueError(f"champ EDF trop long : {texte!r}")
        return b.ljust(taille)

    mois = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")[debut.month - 1]
    entete = b"".join([
        champ("0", 8), champ("X X X X", 80),
        champ(f"Startdate {debut.day:02d}-{mois}-{debut.year} X X Somnia_non_valide", 80),
        champ(debut.strftime("%d.%m.%y"), 8), champ(debut.strftime("%H.%M.%S"), 8),
        champ("512", 8), champ("EDF+C", 44), champ("1", 8), champ(str(duree_totale), 8), champ("1", 4),
        champ("EDF Annotations", 16), champ("", 80), champ("", 8), champ("-1", 8), champ("1", 8),
        champ("-32768", 8), champ("32767", 8), champ("", 80), champ(str(n_echantillons), 8), champ("", 32),
    ])
    assert len(entete) == 512
    chemin = Path(chemin)
    chemin.write_bytes(entete + donnees)
    return chemin
