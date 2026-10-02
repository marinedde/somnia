"""
Tâche 0.6 — Aucune personne ne doit être à la fois dans l'entraînement et dans le test.

Ces tests tournent en CI sans données : ils vérifient le module de découpage et
le fichier de découpage versionné. Le dernier test montre que l'ancien
découpage (par époque) est bien refusé.
"""

from pathlib import Path

import numpy as np
import pytest

from somnia.splits import (
    charger_decoupage,
    decouper_par_personne,
    masques,
    verifier_decoupage,
)

ROOT = Path(__file__).resolve().parent.parent
SPLIT_V1 = ROOT / "data" / "splits" / "physionet_v1.json"


def test_decoupage_disjoint_et_complet():
    personnes = [f"SC{i:02d}" for i in range(16)]
    dec = decouper_par_personne(personnes, graine=42)
    tous = dec["train"] + dec["val"] + dec["test"]
    assert sorted(tous) == personnes                      # personne oubliée, personne en double
    assert set(dec["train"]).isdisjoint(dec["test"])
    assert set(dec["train"]).isdisjoint(dec["val"])
    assert set(dec["val"]).isdisjoint(dec["test"])


def test_decoupage_reproductible_et_independant_de_l_ordre():
    personnes = [f"a{i:02d}" for i in range(30)]
    d1 = decouper_par_personne(personnes, graine=42)
    d2 = decouper_par_personne(list(reversed(personnes)) + personnes, graine=42)  # ordre et doublons
    assert d1 == d2
    assert decouper_par_personne(personnes, graine=7) != d1


def test_verifier_refuse_une_personne_dans_deux_ensembles():
    with pytest.raises(ValueError, match="deux ensembles"):
        verifier_decoupage({"train": ["SC00", "SC01"], "val": ["SC02"], "test": ["SC01"]})


def test_verifier_refuse_un_ensemble_vide():
    with pytest.raises(ValueError, match="vide"):
        verifier_decoupage({"train": ["SC00"], "val": [], "test": ["SC01"]})


def test_masques_alignes_sur_les_epoques():
    dec = {"train": ["SC00"], "val": ["SC01"], "test": ["SC02"]}
    personnes = np.array(["SC00", "SC00", "SC01", "SC02", "SC02", "SC02"])
    m = masques(dec, personnes)
    assert m["train"].sum() == 2 and m["val"].sum() == 1 and m["test"].sum() == 3
    assert not np.any(m["train"] & m["test"])


def test_masques_refusent_une_personne_inconnue():
    with pytest.raises(ValueError, match="absentes"):
        masques({"train": ["SC00"], "val": ["SC01"], "test": ["SC02"]}, np.array(["SC00", "SC99"]))


def test_l_ancien_decoupage_par_epoque_est_refuse():
    """Simule train_test_split sur les époques : la même personne finit des deux côtés."""
    rng = np.random.default_rng(0)
    personnes = np.repeat([f"SC{i:02d}" for i in range(16)], 100)   # 100 époques par personne
    idx = rng.permutation(len(personnes))
    n_test = int(0.15 * len(idx))
    ancien = {
        "test": sorted(set(personnes[idx[:n_test]])),
        "val": sorted(set(personnes[idx[n_test:2 * n_test]])),
        "train": sorted(set(personnes[idx[2 * n_test:]])),
    }
    with pytest.raises(ValueError, match="deux ensembles"):
        verifier_decoupage(ancien)


@pytest.mark.skipif(not SPLIT_V1.exists(), reason="fichier de découpage absent")
def test_fichier_physionet_v1_est_valide():
    contenu = charger_decoupage(SPLIT_V1)        # lève ValueError si une tâche fuit
    assert set(contenu["taches"]) >= {"eeg", "ecg"}
    for tache, dec in contenu["taches"].items():
        assert set(dec["train"]).isdisjoint(dec["test"]), tache
        assert set(dec["train"]).isdisjoint(dec["val"]), tache
    # Sleep-EDF : 16 personnes connues, Apnea-ECG : 30 groupes (c05 = c06)
    assert sum(len(v) for v in contenu["taches"]["eeg"].values()) == 16
    assert sum(len(v) for v in contenu["taches"]["ecg"].values()) == 30
