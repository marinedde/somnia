"""Tests de la tâche 0.1 : identifiant de la personne à partir d'un nom de fichier."""

from pathlib import Path

import pytest

from somnia.subjects import groupes_disjoints, identifiant_personne


# ── Sleep-EDF ─────────────────────────────────────────────────────────────
def test_sleep_edf_exemple_du_plan():
    assert identifiant_personne("SC4012E0") == "SC01"


def test_sleep_edf_deux_nuits_meme_personne():
    assert identifiant_personne("SC4011E0-PSG.edf") == identifiant_personne("SC4012E0-PSG.edf")


def test_sleep_edf_psg_et_hypnogramme_meme_personne():
    assert identifiant_personne("SC4001E0-PSG.edf") == identifiant_personne("SC4001EC-Hypnogram.edf")


def test_sleep_edf_personnes_differentes():
    assert identifiant_personne("SC4001E0") != identifiant_personne("SC4011E0")


def test_sleep_edf_accepte_un_chemin_complet():
    assert identifiant_personne(Path("data/raw/SC4152E0-PSG.edf")) == "SC15"


def test_sleep_edf_telemetrie_distincte_de_cassette():
    # ST = étude télémétrie, autre groupe de personnes que SC
    assert identifiant_personne("ST7011J0-PSG.edf") == "ST01"
    assert identifiant_personne("ST7011J0-PSG.edf") != identifiant_personne("SC4011E0-PSG.edf")


# ── Apnea-ECG ─────────────────────────────────────────────────────────────
def test_apnea_ecg_exemple_du_plan():
    assert identifiant_personne("a05") == "a05"


def test_apnea_ecg_avec_extension_et_chemin():
    assert identifiant_personne("data/raw_apnea/b02.dat") == "b02"
    assert identifiant_personne("b02.apn") == "b02"


def test_apnea_ecg_doublon_documente_c05_c06():
    assert identifiant_personne("c06") == identifiant_personne("c05")


def test_apnea_ecg_autres_enregistrements_distincts():
    assert identifiant_personne("c04") != identifiant_personne("c05")


# ── SHHS ──────────────────────────────────────────────────────────────────
def test_shhs_deux_visites_meme_personne():
    assert identifiant_personne("shhs1-200001.edf") == identifiant_personne("shhs2-200001.edf")


def test_shhs_edf_et_xml_meme_personne():
    assert identifiant_personne("shhs1-200001.edf") == identifiant_personne("shhs1-200001-nsrr.xml")


def test_shhs_personnes_differentes():
    assert identifiant_personne("shhs1-200001.edf") != identifiant_personne("shhs1-200002.edf")


# ── Garde-fous ────────────────────────────────────────────────────────────
def test_nom_inconnu_leve_une_erreur():
    with pytest.raises(ValueError):
        identifiant_personne("signal_mystere.edf")


def test_groupes_disjoints():
    assert groupes_disjoints({"SC00", "SC01"}, {"SC02"}, {"SC03"})
    assert not groupes_disjoints({"SC00", "SC01"}, {"SC01"})
    assert not groupes_disjoints(["a01"], ["b01"], ["a01"])
