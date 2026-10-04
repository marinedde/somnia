"""Tests des exports de la sortie par nuit (sans PyTorch)."""

from datetime import datetime

import numpy as np
import pytest

from somnia.export import MENTION, lignes_d_annotation, vers_csv, vers_edf_plus, vers_xml_nsrr
from somnia.nuit import analyser_nuit_complete
from somnia.shhs import lire_annotations


def _rapport():
    codes = [0] * 4 + [2] * 30 + [3] * 10 + [4] * 16                       # 60 époques : W, N2, N3, REM
    logits = np.zeros((60, 5)); logits[np.arange(60), codes] = 20.0
    proba = np.zeros((1800, 3)); proba[:, 0] = 1.0
    proba[300:320] = 0.0; proba[300:320, 1] = 1.0                           # une apnée de 20 s
    proba[900:915] = 0.0; proba[900:915, 2] = 1.0                           # une hypopnée de 15 s
    eveils = np.zeros(1800); eveils[320:326] = 0.9
    return analyser_nuit_complete(None, lambda _: logits, proba, proba_eveils_sec=eveils), codes


def test_lignes_triees_et_completes():
    r, _ = _rapport()
    L = lignes_d_annotation(r)
    assert [l["famille"] for l in L].count("stade") == 60 and [l["libelle"] for l in L if l["famille"] != "stade"] == ["apnée", "micro-éveil", "hypopnée"]
    assert all(a["debut_s"] <= b["debut_s"] for a, b in zip(L, L[1:]))


def test_xml_nsrr_relu_par_notre_lecteur(tmp_path):
    r, codes = _rapport()
    ann = lire_annotations(vers_xml_nsrr(r, tmp_path / "nuit-nsrr.xml"))
    assert ann.stades.tolist() == codes                                      # mêmes stades après un aller-retour
    evs = [(e.nom, e.type, e.debut, e.duree) for e in ann.evenements]
    assert evs == [("Apnea", "Respiratory", 300.0, 20.0), ("Arousal", "Arousals", 320.0, 6.0), ("Hypopnea", "Respiratory", 900.0, 15.0)]
    assert MENTION in (tmp_path / "nuit-nsrr.xml").read_text(encoding="utf-8")


def test_csv(tmp_path):
    r, _ = _rapport()
    lignes = vers_csv(r, tmp_path / "nuit.csv").read_text(encoding="utf-8").splitlines()
    assert lignes[0].startswith("# Somnia") and lignes[1] == "debut_s,duree_s,famille,libelle,confiance" and len(lignes) == 2 + 63


def test_edf_plus_entete_et_annotations(tmp_path):
    r, _ = _rapport()
    octets = vers_edf_plus(r, tmp_path / "nuit.edf", debut=datetime(1995, 3, 14, 22, 30, 0)).read_bytes()
    assert octets[:8] == b"0       " and octets[168:176] == b"14.03.95" and octets[176:184] == b"22.30.00"
    assert octets[192:197] == b"EDF+C" and octets[256:271] == b"EDF Annotations"
    n = int(octets[256 + 216:256 + 224])
    assert len(octets) == 512 + 2 * n                                        # un seul enregistrement, de la taille annoncée
    tal = octets[512:]
    assert tal.startswith(b"+0\x14\x14\x00") and b"+300\x1520\x14Apnea\x14\x00" in tal and b"+0\x1530\x14Sleep stage W\x14\x00" in tal
    assert b"shhs" not in octets.lower()                                     # aucune identité dans le fichier


def test_edf_plus_relu_par_mne(tmp_path):
    mne = pytest.importorskip("mne", reason="MNE absent")
    r, _ = _rapport()
    ann = mne.read_annotations(str(vers_edf_plus(r, tmp_path / "nuit.edf")))
    desc = list(ann.description)
    assert desc.count("Sleep stage N2") == 30 and desc.count("Apnea") == 1 and desc.count("Arousal") == 1
    i = desc.index("Hypopnea")
    assert abs(ann.onset[i] - 900) < 1e-6 and abs(ann.duration[i] - 15) < 1e-6
