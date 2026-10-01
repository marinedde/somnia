"""
Garde-fous du dépôt (accord d'utilisation NSRR / SHHS).

Ces tests échouent si un fichier SHHS, un jeton NSRR ou un identifiant de
participant se retrouve suivi par git. Ils ne touchent pas aux données.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SHHS_PATTERNS = [
    re.compile(r"shhs[12]-\d{6}", re.I),      # shhs1-200001.edf, -nsrr.xml …
    re.compile(r"\.edf(\.gz)?$", re.I),
    re.compile(r"-nsrr\.xml$", re.I),
    re.compile(r"-profusion\.xml$", re.I),
]
TOKEN_PATTERN = re.compile(r"""NSRR_TOKEN\s*=\s*["'][A-Za-z0-9_\-]{12,}["']""")


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        return []  # pas un dépôt git (ex. image Docker) : rien à vérifier
    return out.stdout.splitlines()


def test_no_shhs_file_tracked():
    offenders = [
        f for f in _tracked_files()
        if any(p.search(Path(f).name) for p in SHHS_PATTERNS)
    ]
    assert offenders == [], f"Fichiers SHHS/EDF suivis par git : {offenders}"


def test_no_nsrr_token_in_sources():
    offenders = []
    for f in _tracked_files():
        p = ROOT / f
        if p.suffix not in {".py", ".ipynb", ".md", ".yml", ".yaml", ".toml", ".txt", ".env"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if TOKEN_PATTERN.search(text):
            offenders.append(f)
    assert offenders == [], f"Jeton NSRR écrit en dur dans : {offenders}"


def test_gitignore_blocks_shhs():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for needed in ("data/shhs/", "*.edf", "*-nsrr.xml"):
        assert needed in gitignore, f".gitignore doit contenir {needed!r}"
