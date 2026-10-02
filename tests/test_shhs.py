"""
Tests du lecteur d'annotations SHHS sur un petit XML fabriqué à la main.

Le XML imite la structure NSRR connue ; la validation sur un vrai fichier est la
tâche 1.7 du plan et reste à faire à la main avant de s'y fier.
"""

import numpy as np
import pytest

from somnia.shhs import etiquettes_apnee, lire_annotations, resume

XML = """<?xml version="1.0" encoding="UTF-8"?>
<PSGAnnotation>
  <SoftwareVersion>Compumedics</SoftwareVersion>
  <EpochLength>30</EpochLength>
  <ScoredEvents>
    <ScoredEvent><EventType/><EventConcept>Recording Start Time</EventConcept>
      <Start>0</Start><Duration>180</Duration><ClockTime>00.00.00 21.30.00</ClockTime></ScoredEvent>
    <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Wake|0</EventConcept>
      <Start>0</Start><Duration>60</Duration></ScoredEvent>
    <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 2 sleep|2</EventConcept>
      <Start>60</Start><Duration>30</Duration></ScoredEvent>
    <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 4 sleep|4</EventConcept>
      <Start>90</Start><Duration>30</Duration></ScoredEvent>
    <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>REM sleep|5</EventConcept>
      <Start>120</Start><Duration>30</Duration></ScoredEvent>
    <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Unscored|9</EventConcept>
      <Start>150</Start><Duration>30</Duration></ScoredEvent>
    <ScoredEvent><EventType>Respiratory|Respiratory</EventType>
      <EventConcept>Obstructive apnea|Obstructive Apnea</EventConcept>
      <Start>70.5</Start><Duration>22</Duration><SignalLocation>ABDO RES</SignalLocation></ScoredEvent>
    <ScoredEvent><EventType>Respiratory|Respiratory</EventType>
      <EventConcept>Hypopnea|Hypopnea</EventConcept>
      <Start>125</Start><Duration>6</Duration></ScoredEvent>
    <ScoredEvent><EventType>Respiratory|Respiratory</EventType>
      <EventConcept>SpO2 desaturation|SpO2 desaturation</EventConcept>
      <Start>80</Start><Duration>15</Duration></ScoredEvent>
    <ScoredEvent><EventType>Arousals|Arousals</EventType>
      <EventConcept>Arousal|Arousal ()</EventConcept>
      <Start>92</Start><Duration>5</Duration></ScoredEvent>
  </ScoredEvents>
</PSGAnnotation>
"""


@pytest.fixture
def xml(tmp_path):
    p = tmp_path / "shhs1-200001-nsrr.xml"
    p.write_text(XML, encoding="utf-8")
    return p


def test_stades_un_par_epoque(xml):
    a = lire_annotations(xml)
    # 60 s d'éveil = 2 époques, N2, stade 4 -> N3, REM, non scoré -> -1
    assert a.stades.tolist() == [0, 0, 2, 3, 4, -1]
    assert a.duree_epoque == 30


def test_evenements_lus_sans_les_stades(xml):
    a = lire_annotations(xml)
    noms = sorted(e.nom for e in a.evenements)
    assert noms == ["Arousal", "Hypopnea", "Obstructive apnea", "Recording Start Time", "SpO2 desaturation"]
    resp = a.respiratoires()
    assert {e.nom for e in resp} == {"Obstructive apnea", "Hypopnea", "SpO2 desaturation"}
    apnee = next(e for e in resp if e.nom == "Obstructive apnea")
    assert (apnee.debut, apnee.duree, apnee.fin) == (70.5, 22.0, 92.5)


def test_etiquette_apnee_regle_des_10_secondes(xml):
    a = lire_annotations(xml)
    # Apnée 70,5 -> 92,5 s : 19,5 s dans l'époque 2 (60-90), 2,5 s dans l'époque 3 (90-120)
    # Hypopnée 125 -> 131 s : 6 s dans l'époque 4 -> sous le seuil de 10 s
    assert etiquettes_apnee(a).tolist() == [0, 0, 1, 0, 0, 0]
    # Avec la règle « dès qu'elle touche », les époques 3 et 4 deviennent positives
    assert etiquettes_apnee(a, recouvrement_min_s=0.001).tolist() == [0, 0, 1, 1, 1, 0]


def test_desaturation_ne_compte_pas_comme_apnee(xml):
    a = lire_annotations(xml)
    # La désaturation couvre 80-95 s (10 s dans l'époque 2) : seule l'apnée rend l'époque 2 positive
    seulement_hypopnees = etiquettes_apnee(a, evenements_comptes=("hypopnea",))
    assert seulement_hypopnees.tolist() == [0, 0, 0, 0, 0, 0]


def test_resume_pour_le_tableau_du_pilote(xml):
    r = resume(lire_annotations(xml))
    assert r["n_Wake"] == 2 and r["n_N2"] == 1 and r["n_N3"] == 1 and r["n_REM"] == 1 and r["n_Inconnu"] == 1
    assert r["n_apnees_hypopnees"] == 2
    assert r["duree_h"] == 0.05
    assert r["temps_sommeil_h"] == 0.03   # 3 époques de sommeil
    assert r["iah_estime"] == round(2 / 0.025, 1)


def test_trou_dans_les_stades_comble_par_inconnu(tmp_path):
    p = tmp_path / "trou.xml"
    p.write_text("""<PSGAnnotation><EpochLength>30</EpochLength><ScoredEvents>
      <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Wake|0</EventConcept><Start>0</Start><Duration>30</Duration></ScoredEvent>
      <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 1 sleep|1</EventConcept><Start>90</Start><Duration>30</Duration></ScoredEvent>
    </ScoredEvents></PSGAnnotation>""", encoding="utf-8")
    assert lire_annotations(p).stades.tolist() == [0, -1, -1, 1]


def test_etiquettes_vides_sans_evenement(tmp_path):
    p = tmp_path / "vide.xml"
    p.write_text("""<PSGAnnotation><EpochLength>30</EpochLength><ScoredEvents>
      <ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 2 sleep|2</EventConcept><Start>0</Start><Duration>90</Duration></ScoredEvent>
    </ScoredEvents></PSGAnnotation>""", encoding="utf-8")
    a = lire_annotations(p)
    assert np.array_equal(etiquettes_apnee(a), np.zeros(3, dtype=np.int64))
