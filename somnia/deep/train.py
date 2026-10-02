"""
Boucle d'entraînement et d'évaluation (tâches 4.3 à 4.6), calibration et refus (roadmap §5.2, §5.5).

Principes :
  - graines fixées (random, numpy, torch) ; sur la puce graphique du Mac (MPS), certaines
    opérations restent non déterministes : on l'écrit, on ne le cache pas ;
  - arrêt anticipé sur la VALIDATION (kappa pour les stades, aire précision-rappel pour l'apnée) ;
    le test SHHS n'est jamais touché ici ;
  - le test du petit lot (`petit_lot`) vérifie que le réseau sait apprendre 32 exemples par
    cœur avant de lancer des heures d'entraînement ;
  - calibration : erreur de calibration attendue (ECE) avant et après mise à l'échelle par
    température, la température étant ajustée sur une MOITIÉ des personnes de validation et
    mesurée sur l'autre, puis l'inverse ;
  - courbe précision / couverture : que gagne-t-on en refusant les époques les moins sûres ?
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from somnia.deep.data import N_CLASSES, SignalDataset, Tableau, normaliser_lot, poids_de_classes
from somnia.deep.model import CNN1D, n_parametres
from somnia.evaluation import metriques_apnee, metriques_stades


def fixer_graines(graine: int) -> None:
    random.seed(graine); np.random.seed(graine); torch.manual_seed(graine)


def appareil() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class Historique:
    epoques: list[dict] = field(default_factory=list)
    meilleure_epoque: int = -1
    meilleur_score: float = -np.inf


def _predire(modele: nn.Module, loader: DataLoader, dev: torch.device, temperature: float = 1.0):
    """Renvoie (logits (n, C), y (n,)) sur CPU, en numpy."""
    modele.eval()
    logits, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = normaliser_lot(x.to(dev))
            logits.append((modele(x) / temperature).float().cpu()); ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _metriques(tache: str, y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    if tache == "eeg":
        return metriques_stades(y, pred)
    return metriques_apnee(y, pred, proba[:, 1])


def score_de_selection(tache: str, m: dict) -> float:
    return m["kappa"] if tache == "eeg" else m["auc_pr"]


def construire_loader(tableau: Tableau, tache: str, entrainement: bool, batch: int, graine: int = 42) -> DataLoader:
    g = torch.Generator(); g.manual_seed(graine)
    return DataLoader(SignalDataset(tableau, tache, entrainement), batch_size=batch, shuffle=entrainement,
                      num_workers=0, generator=g, drop_last=False)


def entrainer(tache: str, train: Tableau, val: Tableau, *, graine: int = 42, max_epoques: int = 15,
              patience: int = 3, batch: int = 256, lr: float = 1e-3, journal=print) -> tuple[CNN1D, Historique]:
    """Entraîne avec arrêt anticipé sur la validation ; renvoie le meilleur modèle et l'historique."""
    fixer_graines(graine)
    dev = appareil()
    modele = CNN1D(N_CLASSES[tache]).to(dev)
    journal(f"appareil {dev.type} | {n_parametres(modele):,} paramètres | train {len(train):,} | val {len(val):,}")
    poids = poids_de_classes(train.y, N_CLASSES[tache]).to(dev)
    perte = nn.CrossEntropyLoss(weight=poids)
    opt = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epoques)
    l_train = construire_loader(train, tache, True, batch, graine)
    l_val = construire_loader(val, tache, False, batch * 2)

    hist, meilleur_etat, sans_progres = Historique(), None, 0
    for ep in range(max_epoques):
        modele.train(); t0 = time.time(); total, n = 0.0, 0
        for x, y in l_train:
            x, y = normaliser_lot(x.to(dev)), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = perte(modele(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(y); n += len(y)
        sched.step()
        logits_v, y_v = _predire(modele, l_val, dev)
        proba_v = _softmax(logits_v)
        perte_val = float(nn.functional.cross_entropy(torch.from_numpy(logits_v), torch.from_numpy(y_v)))
        m = _metriques(tache, y_v, proba_v)
        score = score_de_selection(tache, m)
        hist.epoques.append({"epoque": ep + 1, "perte_train": total / n, "perte_val": perte_val,
                             "score_val": score, "duree_s": round(time.time() - t0, 1), **{k: m[k] for k in m}})
        journal(f"époque {ep + 1:2d} | perte train {total / n:.4f} | perte val {perte_val:.4f} | "
                f"{'kappa' if tache == 'eeg' else 'auc_pr'} val {score:.4f} | {time.time() - t0:.0f} s")
        if score > hist.meilleur_score + 1e-4:
            hist.meilleur_score, hist.meilleure_epoque, sans_progres = score, ep + 1, 0
            meilleur_etat = {k: v.detach().cpu().clone() for k, v in modele.state_dict().items()}
        else:
            sans_progres += 1
            if sans_progres >= patience:
                journal(f"arrêt anticipé : pas de progrès depuis {patience} époques")
                break
    modele.load_state_dict(meilleur_etat)
    return modele, hist


def petit_lot(tache: str, train: Tableau, n: int = 32, pas: int = 300, graine: int = 42, journal=print) -> dict:
    """Le réseau doit pouvoir apprendre n exemples par cœur : la perte doit tomber près de 0."""
    fixer_graines(graine)
    dev = appareil()
    rng = np.random.default_rng(graine)
    idx = rng.choice(len(train), size=n, replace=False)
    petit = Tableau(train.X[idx], train.y[idx], train.personne[idx])
    modele = CNN1D(N_CLASSES[tache]).to(dev)
    opt = torch.optim.Adam(modele.parameters(), lr=1e-3)
    perte = nn.CrossEntropyLoss()
    x = normaliser_lot(torch.from_numpy(petit.X.astype(np.float32)).unsqueeze(1).to(dev))
    y = torch.from_numpy(petit.y).to(dev)
    pertes = []
    modele.train()
    for _ in range(pas):
        opt.zero_grad(set_to_none=True)
        loss = perte(modele(x), y); loss.backward(); opt.step()
        pertes.append(loss.item())
    modele.eval()
    with torch.no_grad():
        exactitude = float((modele(x).argmax(1) == y).float().mean())
    journal(f"petit lot ({n} ex.) : perte {pertes[0]:.3f} -> {pertes[-1]:.4f}, exactitude {exactitude:.0%}")
    return {"n": n, "pas": pas, "perte_debut": pertes[0], "perte_fin": pertes[-1], "exactitude": exactitude,
            "ok": pertes[-1] < 0.1 and exactitude > 0.95}


# ── Calibration et refus ──────────────────────────────────────────────────
def ece(proba: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Erreur de calibration attendue : |confiance moyenne − exactitude| pondérée par bac."""
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    bords = np.linspace(0, 1, n_bins + 1); total = 0.0
    for a, b in zip(bords[:-1], bords[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return float(total)


def ajuster_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Une seule constante T > 0 qui minimise la log-vraisemblance négative de softmax(logits / T)."""
    lg, yt = torch.from_numpy(logits).double(), torch.from_numpy(y)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def fermeture():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / torch.exp(log_t), yt)
        loss.backward()
        return loss
    opt.step(fermeture)
    return float(torch.exp(log_t).item())


def calibration_croisee(logits: np.ndarray, y: np.ndarray, personnes: np.ndarray, graine: int = 42) -> dict:
    """ECE avant/après température, la température étant ajustée sur l'autre moitié des personnes."""
    pers = np.array(sorted(set(personnes)))
    rng = np.random.default_rng(graine); rng.shuffle(pers)
    moities = [set(pers[: len(pers) // 2]), set(pers[len(pers) // 2:])]
    avant, apres, temps = [], [], []
    for i in (0, 1):
        ajust = np.isin(personnes, list(moities[i])); mesure = ~ajust
        T = ajuster_temperature(logits[ajust], y[ajust])
        avant.append(ece(_softmax(logits[mesure]), y[mesure]))
        apres.append(ece(_softmax(logits[mesure] / T), y[mesure]))
        temps.append(T)
    return {"ece_avant": float(np.mean(avant)), "ece_apres": float(np.mean(apres)),
            "temperature": float(np.mean(temps)), "n_personnes_val": int(len(pers))}


def courbe_couverture(proba: np.ndarray, y: np.ndarray, tache: str,
                      couvertures=(1.0, 0.9, 0.8, 0.7, 0.5)) -> list[dict]:
    """On garde les époques les plus sûres : exactitude (ou F1 apnée) en fonction de la part gardée."""
    conf = proba.max(axis=1)
    ordre = np.argsort(-conf)
    out = []
    for c in couvertures:
        k = max(1, int(round(c * len(y))))
        idx = ordre[:k]
        m = _metriques(tache, y[idx], proba[idx])
        out.append({"couverture": c, "n": int(k), "seuil_confiance": float(conf[ordre[k - 1]]),
                    "accuracy": m["accuracy"], "kappa" if tache == "eeg" else "f1_apnee": m["kappa" if tache == "eeg" else "f1_apnee"]})
    return out


def evaluer(modele: CNN1D, tache: str, val: Tableau, batch: int = 512) -> dict:
    """Tout ce qu'on rapporte sur la validation : métriques, calibration, couverture, logits bruts."""
    dev = appareil()
    logits, y = _predire(modele, construire_loader(val, tache, False, batch), dev)
    proba = _softmax(logits)
    return {
        "metriques": _metriques(tache, y, proba),
        "calibration": calibration_croisee(logits, y, val.personne),
        "couverture": courbe_couverture(proba, y, tache),
        "logits": logits, "y": y, "personnes": val.personne,
    }
