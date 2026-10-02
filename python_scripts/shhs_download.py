#!/usr/bin/env python3
"""
Somnia — Téléchargement d'un sous-ensemble SHHS depuis le NSRR, robuste aux coupures.

Pourquoi ne plus passer par `sleepecg.download_nsrr` : il appelle `requests.get` sans délai
d'attente ni flux. Sur une connexion lente, une requête qui décroche reste bloquée pour
toujours (constaté le 2 octobre 2026 : 40 minutes sans un octet, processus vivant).
Ici : flux par morceaux, délai d'attente, 5 tentatives par fichier, écriture dans un
fichier .part renommé à la fin, vérification MD5, reprise : les fichiers déjà complets sont
sautés. Un fichier qui échoue cinq fois est noté et on passe au suivant.

Règles :
  - le jeton est lu dans la variable d'environnement NSRR_TOKEN, jamais écrit ni affiché ;
  - la destination est ~/data/shhs/raw (hors iCloud, hors dépôt), modifiable par SHHS_DIR ;
  - le script refuse d'écrire dans le dépôt ou dans un dossier iCloud.

Usage :
    export NSRR_TOKEN="..."                                   # https://sleepdata.org/token
    python python_scripts/shhs_download.py --dry-run
    python python_scripts/shhs_download.py                    # pilote : 9 nuits
    python python_scripts/shhs_download.py --pattern "*-200[0-2]??[.-]*"   # ~300 nuits

API NSRR : https://github.com/nsrr/sleepdata.org/wiki/api-v1-datasets
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = Path.home() / "data" / "shhs" / "raw"
API = "https://sleepdata.org/api/v1"
DB = "shhs"
SUBFOLDERS = {
    "edf": "polysomnography/edfs/{visit}",
    "xml": "polysomnography/annotations-events-nsrr/{visit}",
}
TIMEOUT = (15, 120)      # connexion, puis silence maximal entre deux morceaux (secondes)
CHUNK = 1 << 20          # 1 Mo
TENTATIVES = 5


def _check_destination(dest: Path) -> None:
    resolved = dest.resolve()
    if "Mobile Documents" in str(resolved) or "iCloud" in str(resolved):
        sys.exit(f"Refus : {resolved} est synchronisé iCloud. Utilise ~/data/shhs.")
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return
    sys.exit(f"Refus : {resolved} est dans le dépôt git. Utilise ~/data/shhs.")


def _md5(chemin: Path) -> str:
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(CHUNK), b""):
            h.update(bloc)
    return h.hexdigest()


def _session():
    try:
        import truststore  # sleepdata.org n'envoie pas son certificat intermédiaire
        truststore.inject_into_ssl()
    except ImportError:
        print("Conseil : pip install truststore (vérification TLS via les certificats macOS)")
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "somnia-shhs-download/1.0"
    return s


def _verifier_jeton(session, token: str) -> str:
    try:
        r = session.get(f"{API}/account/profile.json", params={"auth_token": token}, timeout=TIMEOUT)
        d = r.json()
    except Exception as e:  # le jeton ne doit jamais apparaître dans une trace
        sys.exit(f"Échec de connexion à sleepdata.org : {type(e).__name__}. Réseau, TLS ou jeton ?")
    if not d.get("authenticated"):
        sys.exit("Jeton NSRR refusé par sleepdata.org : régénère-le sur https://sleepdata.org/token")
    return d.get("username", "?")


def _lister(session, subfolder: str, pattern: str) -> list[tuple[str, str]]:
    """[(chemin_complet, md5)] des fichiers du dossier dont le nom correspond au motif."""
    r = session.get(f"{API}/datasets/{DB}/files.json", params={"path": subfolder}, timeout=TIMEOUT)
    r.raise_for_status()
    return [(i["full_path"], i["file_checksum_md5"]) for i in r.json()
            if i["is_file"] and fnmatch.fnmatch(i["file_name"], pattern)]


def _telecharger(session, token: str, chemin_distant: str, md5_attendu: str, cible: Path) -> str:
    """Renvoie 'present', 'ok' ou 'echec'. N'imprime jamais l'URL (elle contient le jeton)."""
    if cible.exists():
        if _md5(cible) == md5_attendu:
            return "present"
        cible.unlink()  # fichier partiel ou corrompu d'un essai précédent
    cible.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://sleepdata.org/datasets/{DB}/files/a/{token}/m/somnia/{chemin_distant}"
    partiel = cible.with_suffix(cible.suffix + ".part")
    for tentative in range(1, TENTATIVES + 1):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                if "content-disposition" not in r.headers:
                    sys.exit(f"Accès refusé à {DB} : vérifie que ta demande d'accès NSRR est approuvée.")
                with open(partiel, "wb") as f:
                    for bloc in r.iter_content(CHUNK):
                        f.write(bloc)
            if _md5(partiel) != md5_attendu:
                raise IOError("somme MD5 différente")
            partiel.rename(cible)
            return "ok"
        except SystemExit:
            raise
        except Exception as e:
            attente = min(60, 5 * tentative)
            print(f"    tentative {tentative}/{TENTATIVES} échouée ({type(e).__name__}), "
                  f"nouvel essai dans {attente} s", flush=True)
            partiel.unlink(missing_ok=True)
            time.sleep(attente)
    return "echec"


def main() -> int:
    parser = argparse.ArgumentParser(description="Télécharge un sous-ensemble SHHS")
    # Les identifiants vont de 200001 à 205804 (5 793 nuits). Le motif s'applique au nom de
    # fichier complet : "shhs1-200001.edf" ET "shhs1-200001-nsrr.xml", d'où le "[.-]".
    #   "*-20000?[.-]*"      -> 200001 à 200009 :  9 nuits (pilote)
    #   "*-200[0-2]??[.-]*"  -> 200001 à 200299 : ~300 nuits
    parser.add_argument("--pattern", default="*-20000?[.-]*",
                        help='motif de nom de fichier (défaut "*-20000?[.-]*" = 9 nuits)')
    parser.add_argument("--visit", default="shhs1", choices=["shhs1", "shhs2"])
    parser.add_argument("--dest", default=os.environ.get("SHHS_DIR", str(DEFAULT_DIR)),
                        help="dossier de destination (défaut : $SHHS_DIR ou ~/data/shhs/raw)")
    parser.add_argument("--only", choices=["edf", "xml"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="liste ce qui serait fait, sans télécharger")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser()
    _check_destination(dest)
    kinds = [args.only] if args.only else ["xml", "edf"]   # XML d'abord : petits, et nécessaires
    plan = [(k, SUBFOLDERS[k].format(visit=args.visit)) for k in kinds]

    print(f"Destination : {dest}")
    print(f"Visite      : {args.visit}   Motif : {args.pattern}")
    session = _session()
    fichiers = []
    for kind, sub in plan:
        liste = _lister(session, sub, args.pattern)
        print(f"  - {kind:3s} ← {DB}/{sub} : {len(liste)} fichiers")
        fichiers += liste
    if args.dry_run:
        print("[dry-run] rien téléchargé." + ("" if os.environ.get("NSRR_TOKEN") else " (NSRR_TOKEN non défini)"))
        return 0

    token = os.environ.get("NSRR_TOKEN")
    if not token:
        sys.exit('NSRR_TOKEN absent. Fais : export NSRR_TOKEN="..." (jeton sur sleepdata.org/token)')
    print(f"Authentifié sur sleepdata.org : {_verifier_jeton(session, token)}")

    bilan = {"present": 0, "ok": 0, "echec": []}
    debut = time.time()
    for i, (chemin_distant, md5) in enumerate(fichiers, 1):
        cible = dest / DB / chemin_distant
        statut = _telecharger(session, token, chemin_distant, md5, cible)
        if statut == "echec":
            bilan["echec"].append(chemin_distant)
        else:
            bilan[statut] += 1
        if statut != "present" or i % 25 == 0 or i == len(fichiers):
            ecoule = (time.time() - debut) / 60
            print(f"[{i}/{len(fichiers)}] {Path(chemin_distant).name} : {statut}   ({ecoule:.0f} min)", flush=True)

    n_edf = len(list(dest.rglob("*.edf")))
    n_xml = len(list(dest.rglob("*-nsrr.xml")))
    print(f"\nTerminé. Déjà présents : {bilan['present']}, téléchargés : {bilan['ok']}, "
          f"échecs : {len(bilan['echec'])}. EDF : {n_edf}   XML : {n_xml}   dans {dest}")
    if bilan["echec"]:
        print("Échecs (relance la même commande pour réessayer) :")
        for f in bilan["echec"]:
            print("   ", f)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
