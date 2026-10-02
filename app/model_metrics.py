"""
Métriques affichées par l'API : lues dans models/training_metrics.json, écrit par
python_scripts/retrain.py. Plus aucun chiffre codé en dur dans le code (tâche 0.7).

Le chiffre mis en avant est celui de la validation croisée PAR PERSONNE quand il
existe (clé "cv"), sinon celui du jeu de test par personne.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "models" / "training_metrics.json"


def charger_metriques(tache: str) -> dict:
    """Renvoie un dictionnaire plat pour la tâche 'eeg' ou 'ecg' (vide si le fichier manque)."""
    if not METRICS_PATH.exists():
        return {}
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    bloc = data.get(tache) or {}
    if not bloc:
        return {}

    cv = bloc.get("cv") or {}
    principal = cv.get("moyenne") or bloc.get("test") or {}
    source = cv.get("methode") or "jeu de test par personne"

    return {
        "training_date": data.get("training_date", "inconnue"),
        "split_method": data.get("split_method", bloc.get("split", {}).get("methode", "inconnu")),
        "dataset": bloc.get("dataset", "inconnu"),
        "n_personnes": bloc.get("split", {}).get("n_personnes"),
        "metrics_source": source,
        "metrics": {k: round(float(v), 4) for k, v in principal.items()
                    if isinstance(v, (int, float)) and not str(k).startswith("n_")},
        "ecart_type": {k: round(float(v), 4) for k, v in (cv.get("ecart_type") or {}).items()
                       if isinstance(v, (int, float)) and not str(k).startswith("n_")},
        "avant_fuite": cv.get("avant_fuite"),
    }
