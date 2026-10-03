#!/usr/bin/env python3
"""
Démonstration de la sortie par nuit sur une nuit PUBLIQUE de Sleep-EDF (jamais SHHS).

Le réseau de stades entraîné sur SHHS (C4-A1) est appliqué à une nuit Sleep-EDF (Fpz-Cz) :
c'est aussi une validation externe, honnête, sur un autre montage. On montre côte à côte
l'hypnogramme annoté par le technicien, celui proposé, la confiance, et la file de relecture.

Usage :
    python python_scripts/demo_nuit.py                     # première nuit de data/raw
    python python_scripts/demo_nuit.py --nuit SC4012E0 --seuil 0.6

Sorties : data/figures/demo_nuit.png, docs/demo_nuit.json (résumé, sans le signal)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from somnia.deep.data import normaliser_lot  # noqa: E402
from somnia.deep.model import CNN1D  # noqa: E402
from somnia.evaluation import metriques_stades  # noqa: E402
from somnia.nuit import STADES, analyser_nuit  # noqa: E402
from somnia.physionet import charger_epoques_eeg, paires_sleep_edf  # noqa: E402

MODELE = ROOT / "models" / "cnn" / "eeg_f1_s42.pt"
RAPPORT = ROOT / "models" / "cnn" / "eeg_f1_s42.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuit", default=None, help="ex. SC4012E0 (défaut : première nuit)")
    parser.add_argument("--seuil", type=float, default=0.6)
    args = parser.parse_args()

    paires = paires_sleep_edf(ROOT / "data" / "raw")
    ident, psg, hyp = next((p for p in paires if args.nuit and p[1].name.startswith(args.nuit)), paires[0])
    epoques, annotes = charger_epoques_eeg(psg, hyp)                  # volts, toute la cassette (~20 h)
    # On garde la période du premier au dernier sommeil annoté + 1 h de marge de chaque côté :
    # c'est ce qu'un enregistrement de nuit contient réellement.
    idx = np.flatnonzero(annotes != 0)
    a, b = max(0, idx[0] - 120), min(len(annotes), idx[-1] + 121)
    epoques, annotes = epoques[a:b], annotes[a:b]

    modele = CNN1D(5); modele.load_state_dict(torch.load(MODELE, map_location="cpu")); modele.eval()
    temperature = json.loads(RAPPORT.read_text(encoding="utf-8"))["calibration"]["temperature"]

    def predire(x_volts):
        with torch.no_grad():
            x = torch.from_numpy((x_volts * 1e6).astype(np.float32)).unsqueeze(1)   # µV comme à l'entraînement
            return modele(normaliser_lot(x)).numpy()

    res = analyser_nuit(epoques, predire, temperature=temperature, seuil=args.seuil)
    pred = np.array([{v: k for k, v in STADES.items()}[s] for s in res["hypnogramme"]])
    m = metriques_stades(annotes, pred)
    conf = np.array(res["confiance"])
    m_surs = metriques_stades(annotes[conf >= args.seuil], pred[conf >= args.seuil]) if (conf >= args.seuil).any() else {}

    resume = {
        "nuit": ident, "source": "Sleep-EDF (PhysioNet), canal Fpz-Cz ; modèle entraîné sur SHHS C4-A1 (validation externe)",
        "n_epoques": int(len(annotes)), "accord_avec_le_technicien": {k: round(v, 3) for k, v in m.items()},
        "accord_sur_les_epoques_gardees": {k: round(v, 3) for k, v in m_surs.items()},
        "indices_predits": res["indices"], "relecture": {k: v for k, v in res["relecture"].items() if k != "segments"},
        "segments_a_relire_en_premier": res["relecture"]["segments"][:5], "statut": res["statut"],
    }
    (ROOT / "docs" / "demo_nuit.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordre = {0: 4, 1: 2, 2: 1, 3: 0, 4: 3}     # affichage classique : N3 en bas, W en haut, REM entre
    t_h = np.arange(len(annotes)) * 30 / 3600
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})
    for ax, s, titre in ((axes[0], annotes, "Hypnogramme annoté par le technicien (référence)"),
                         (axes[1], pred, f"Hypnogramme proposé par le réseau — accord {m['accuracy']:.0%}, kappa {m['kappa']:.2f} — NON VALIDÉ")):
        ax.step(t_h, [ordre[int(v)] for v in s], where="post", color="#2E4057", lw=1)
        ax.set_yticks([0, 1, 2, 3, 4]); ax.set_yticklabels(["N3", "N2", "N1", "REM", "W"]); ax.set_title(titre, loc="left", fontsize=10); ax.grid(alpha=0.3)
    douteux = conf < args.seuil
    axes[1].fill_between(t_h, -0.5, 4.5, where=douteux, color="#E84855", alpha=0.25, step="post", label=f"à relire ({douteux.mean():.0%} des époques)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(t_h, conf, color="#048A81", lw=0.8); axes[2].axhline(args.seuil, color="#E84855", ls="--", lw=1)
    axes[2].set_ylim(0, 1); axes[2].set_ylabel("confiance"); axes[2].set_xlabel("heures depuis le début de la fenêtre"); axes[2].grid(alpha=0.3)
    axes[2].set_title(f"Confiance calibrée (T = {temperature:.2f}) ; sous {args.seuil}, l'époque part en relecture : "
                      f"{res['relecture']['temps_relecture_estime_min']:.0f} min à relire au lieu de {res['indices']['temps_enregistrement_min']:.0f}", loc="left", fontsize=10)
    fig.suptitle(f"Somnia — sortie par nuit, {ident} (Sleep-EDF, données publiques)", fontweight="bold")
    fig.tight_layout(); fig.savefig(ROOT / "data" / "figures" / "demo_nuit.png", dpi=130); plt.close(fig)
    print(json.dumps(resume, indent=2, ensure_ascii=False)[:1800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
