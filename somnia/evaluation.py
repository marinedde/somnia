"""
Mesure honnête des modèles : validation croisée PAR PERSONNE (tâches 0.3 et 0.4).

Chaque pli met des personnes entières de côté. Les métriques sont données en
moyenne ± écart-type sur les plis. Pour la comparaison « avant / après », la
fonction `decoupage_aleatoire_par_epoque` reproduit l'ancien découpage fuité.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from somnia.physionet import STAGE_NAMES

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def pipeline_rf(graine: int = 42) -> Pipeline:
    params = {**RF_PARAMS, "random_state": graine}
    return Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(**params))])


def entrainer_rf(X, y, graine: int = 42) -> Pipeline:
    """Entraîne en parallèle, puis passe l'inférence en séquentiel.

    Avec n_jobs=-1 à la prédiction, les probabilités des arbres sont additionnées dans un ordre
    qui dépend des threads : écart de ~4e-16 d'un appel à l'autre, et une égalité parfaite peut
    basculer. Les arbres, eux, sont identiques. Inférence séquentielle = reproductible au bit près.
    """
    modele = pipeline_rf(graine).fit(X, y)
    modele.named_steps["clf"].set_params(n_jobs=1)
    return modele


# ── Métriques ─────────────────────────────────────────────────────────────
def metriques_stades(y_vrai, y_pred) -> dict[str, float]:
    m = {
        "accuracy": accuracy_score(y_vrai, y_pred),
        "f1_macro": f1_score(y_vrai, y_pred, average="macro", labels=sorted(STAGE_NAMES), zero_division=0),
        "f1_weighted": f1_score(y_vrai, y_pred, average="weighted", labels=sorted(STAGE_NAMES), zero_division=0),
        "kappa": cohen_kappa_score(y_vrai, y_pred),
    }
    f1_par_stade = f1_score(y_vrai, y_pred, average=None, labels=sorted(STAGE_NAMES), zero_division=0)
    for code, f1 in zip(sorted(STAGE_NAMES), f1_par_stade):
        m[f"f1_{STAGE_NAMES[code]}"] = f1
    return {k: float(v) for k, v in m.items()}


def metriques_apnee(y_vrai, y_pred, proba_apnee) -> dict[str, float]:
    m = {
        "accuracy": accuracy_score(y_vrai, y_pred),
        "f1_apnee": f1_score(y_vrai, y_pred, pos_label=1, zero_division=0),
        "auc_roc": roc_auc_score(y_vrai, proba_apnee) if len(set(y_vrai)) == 2 else float("nan"),
        "auc_pr": average_precision_score(y_vrai, proba_apnee) if len(set(y_vrai)) == 2 else float("nan"),
    }
    return {k: float(v) for k, v in m.items()}


def _metriques(tache, y_vrai, y_pred, proba):
    if tache == "eeg":
        return metriques_stades(y_vrai, y_pred)
    return metriques_apnee(y_vrai, y_pred, proba)


def _resume(plis: list[dict]) -> tuple[dict, dict]:
    cles = plis[0].keys()
    moy = {k: float(np.nanmean([p[k] for p in plis])) for k in cles}
    ecart = {k: float(np.nanstd([p[k] for p in plis])) for k in cles}
    return moy, ecart


# ── Validation croisée par personne ───────────────────────────────────────
def validation_croisee_par_personne(X, y, personnes, tache: str,
                                    n_plis: int = 5, graine: int = 42) -> dict:
    """Renvoie {'plis': [...], 'moyenne': {...}, 'ecart_type': {...}, 'oof': {...}}.

    `oof` (out-of-fold) : la prédiction de chaque époque faite par le modèle qui
    n'a PAS vu sa personne. Sert aux courbes ROC honnêtes.
    """
    X, y, personnes = np.asarray(X), np.asarray(y), np.asarray(personnes).astype(str)
    cv = StratifiedGroupKFold(n_splits=n_plis, shuffle=True, random_state=graine)
    plis, oof_pred = [], np.full(len(y), -1)
    oof_proba = np.full(len(y), np.nan)

    for i, (itr, ite) in enumerate(cv.split(X, y, groups=personnes)):
        assert set(personnes[itr]).isdisjoint(personnes[ite]), "fuite : personne dans train ET test"
        modele = entrainer_rf(X[itr], y[itr], graine)
        pred = modele.predict(X[ite])
        proba = modele.predict_proba(X[ite])[:, 1] if tache == "ecg" else None
        oof_pred[ite] = pred
        if proba is not None:
            oof_proba[ite] = proba
        m = _metriques(tache, y[ite], pred, proba)
        m["n_personnes_test"] = len(set(personnes[ite]))
        m["n_epoques_test"] = int(len(ite))
        plis.append(m)

    moyenne, ecart = _resume(plis)
    return {
        "methode": f"StratifiedGroupKFold, {n_plis} plis, graine {graine}, groupes = personnes",
        "n_personnes": int(len(set(personnes))),
        "n_epoques": int(len(y)),
        "plis": plis,
        "moyenne": moyenne,
        "ecart_type": ecart,
        "oof": {"pred": oof_pred, "proba": oof_proba},
    }


def decoupage_aleatoire_par_epoque(X, y, tache: str, graine: int = 42, part_test: float = 0.3) -> dict:
    """L'ANCIEN découpage, reproduit pour la comparaison : train_test_split sur les époques.

    Les deux nuits d'une personne, et des époques voisines de la même nuit, se
    retrouvent des deux côtés : les scores sont gonflés. À ne plus utiliser.
    """
    X, y = np.asarray(X), np.asarray(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=part_test, random_state=graine, stratify=y)
    modele = entrainer_rf(Xtr, ytr, graine)
    pred = modele.predict(Xte)
    proba = modele.predict_proba(Xte)[:, 1] if tache == "ecg" else None
    return {"methode": "train_test_split sur les époques (fuite)", "metriques": _metriques(tache, yte, pred, proba),
            "y_test": yte, "proba": proba}
