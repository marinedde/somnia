#!/usr/bin/env python3
"""
Somnia — Déploiement des deux Spaces HuggingFace, par liste blanche.

Pourquoi ce script : `upload_folder` n'efface jamais rien côté Space. Au fil des
déploiements, les Spaces avaient accumulé caches de tests, présentations, anciens
modèles et notes internes. Ici, chaque déploiement est un commit unique qui
contient EXACTEMENT les fichiers listés ci-dessous et supprime tout le reste
(sauf `.gitattributes`, géré par HuggingFace).

Usage :
    python python_scripts/deploy_hf.py api        --dry-run
    python python_scripts/deploy_hf.py dashboard  --dry-run
    python python_scripts/deploy_hf.py api                    # nécessite HF_TOKEN ou `hf auth login`

Le README de chaque Space est versionné dans le dépôt (`.hf_readme_api.md`,
`.hf_readme_dashboard.md`) : c'est lui qui porte le titre et la configuration du Space.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Pour chaque Space : identifiant, et correspondance {chemin dans le Space: chemin local}.
# Les motifs (avec *) sont développés depuis la racine du dépôt.
SPACES = {
    "api": {
        "repo_id": "marinedde/somnia-api",
        "files": {
            "README.md": ".hf_readme_api.md",
            "Dockerfile": "Dockerfile",
            ".dockerignore": ".dockerignore",
            "requirements.txt": "requirements.txt",
            "app/*.py": "app/*.py",
            "models/somnia_eeg_pipeline.joblib": "models/somnia_eeg_pipeline.joblib",
            "models/somnia_ecg_pipeline.joblib": "models/somnia_ecg_pipeline.joblib",
            "models/baseline_stats.json": "models/baseline_stats.json",
            "models/training_metrics.json": "models/training_metrics.json",
        },
        # Un pointeur Git LFS fait ~130 octets : un modèle plus petit que ça n'en est pas un.
        "min_size": {"models/somnia_eeg_pipeline.joblib": 1_000_000,
                     "models/somnia_ecg_pipeline.joblib": 1_000_000},
    },
    "dashboard": {
        "repo_id": "marinedde/somnia-dashboard",
        "files": {
            "README.md": ".hf_readme_dashboard.md",
            "requirements.txt": "requirements-dashboard.txt",
            "streamlit_app.py": "streamlit_app.py",
            "data/demo/*.npy": "data/demo/*.npy",
        },
        "min_size": {},
    },
}

NEVER_DELETE = {".gitattributes"}


def _resolve(files: dict[str, str]) -> dict[str, Path]:
    """Développe les motifs : {chemin_space: chemin_local}."""
    out: dict[str, Path] = {}
    for remote, local in files.items():
        if "*" in local:
            matches = sorted(ROOT.glob(local))
            if not matches:
                sys.exit(f"Aucun fichier local pour le motif {local!r}")
            for m in matches:
                out[str(m.relative_to(ROOT))] = m
        else:
            p = ROOT / local
            if not p.exists():
                sys.exit(f"Fichier local manquant : {local}")
            out[remote] = p
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Déploie un Space HuggingFace par liste blanche")
    parser.add_argument("space", choices=SPACES)
    parser.add_argument("--dry-run", action="store_true", help="affiche le plan sans rien envoyer")
    args = parser.parse_args()

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    spec = SPACES[args.space]
    wanted = _resolve(spec["files"])

    for remote, min_size in spec["min_size"].items():
        size = wanted[remote].stat().st_size
        if size < min_size:
            sys.exit(f"{remote} ne fait que {size} octets : pointeur LFS ? Lance `git lfs pull`.")

    api = HfApi(token=os.environ.get("HF_TOKEN"))  # None -> jeton de `hf auth login`
    remote_files = set(api.list_repo_files(spec["repo_id"], repo_type="space"))

    to_delete = sorted(remote_files - set(wanted) - NEVER_DELETE)
    ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for r, p in sorted(wanted.items())]
    ops += [CommitOperationDelete(path_in_repo=r) for r in to_delete]

    print(f"Space : {spec['repo_id']}")
    print(f"  {len(wanted)} fichiers envoyés :")
    for r in sorted(wanted):
        print(f"     + {r}")
    print(f"  {len(to_delete)} fichiers supprimés côté Space :")
    for r in to_delete:
        print(f"     - {r}")

    if args.dry_run:
        print("[dry-run] aucun commit envoyé.")
        return 0

    url = api.create_commit(
        repo_id=spec["repo_id"],
        repo_type="space",
        operations=ops,
        commit_message=f"Déploiement Somnia ({args.space}) : liste blanche, nettoyage des fichiers hors périmètre",
    )
    print(f"Commit : {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
