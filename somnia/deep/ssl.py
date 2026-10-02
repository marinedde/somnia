"""
Pré-entraînement auto-supervisé contrastif de l'encodeur (étape 5).

Idée : on apprend au réseau à reconnaître « la même époque vue deux fois » (deux transformations
légères du même signal) parmi des époques d'autres personnes, sans aucune étiquette. Ensuite
seulement, on lui apprend les stades ou l'apnée avec peu d'exemples étiquetés.

Transformations (tâche 5.1), choisies avec un critère clinique : « un soignant jugerait-il le
tracé inchangé ? »
  - bruit gaussien léger        oui : un tracé un peu bruité reste le même tracé
  - amplitude × 0,8 à 1,2       oui : dépend de la pose des électrodes
  - décalage de quelques s      oui : la fenêtre de 30 s est une convention de scoreur
  - masquage d'un court passage oui : une électrode qui décroche une seconde
  - inversion de signe (ECG)    oui pour l'ECG : 80 % des nuits SHHS sont câblées à l'envers
  - inversion temporelle        NON : un QRS à l'envers n'existe pas
  - étirement temporel          NON : change la fréquence cardiaque, qui est le signe de l'apnée

Le piège des voisines (deux époques consécutives de la même personne se ressemblent) :
chaque lot contient UNE époque par personne (`LotsParPersonne`). Deux époques d'un même lot
viennent donc toujours de personnes différentes.

Perte : InfoNCE / NT-Xent (SimCLR) avec température 0,1.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from somnia.deep.data import Tableau, normaliser_lot
from somnia.deep.model import Encodeur


# ── Transformations, vectorisées sur un lot (B, 1, L) ─────────────────────
def bruit(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    return x + sigma * x.std(dim=-1, keepdim=True) * torch.randn_like(x)


def amplitude(x: torch.Tensor, lo: float = 0.8, hi: float = 1.2) -> torch.Tensor:
    f = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(lo, hi)
    return x * f


def decalage(x: torch.Tensor, max_points: int = 300) -> torch.Tensor:
    """Décalage circulaire, différent pour chaque exemple (±3 s à 100 Hz par défaut)."""
    B, _, L = x.shape
    s = torch.randint(-max_points, max_points + 1, (B, 1, 1), device=x.device)
    idx = (torch.arange(L, device=x.device).view(1, 1, L) - s) % L
    return x.gather(-1, idx.expand(B, 1, L))


def masquage(x: torch.Tensor, max_points: int = 300) -> torch.Tensor:
    """Met à zéro un passage de longueur aléatoire (jusqu'à 3 s) à un endroit aléatoire."""
    B, _, L = x.shape
    longueur = torch.randint(0, max_points + 1, (B, 1, 1), device=x.device)
    debut = torch.randint(0, L, (B, 1, 1), device=x.device)
    pos = torch.arange(L, device=x.device).view(1, 1, L)
    masque = (pos >= debut) & (pos < debut + longueur)
    return x.masked_fill(masque, 0.0)


def inversion_signe(x: torch.Tensor) -> torch.Tensor:
    s = torch.where(torch.rand(x.shape[0], 1, 1, device=x.device) < 0.5, -1.0, 1.0)
    return x * s


def vue(x: torch.Tensor, tache: str) -> torch.Tensor:
    """Une vue = le signal après les transformations admises, puis centré-réduit."""
    x = decalage(x)
    x = amplitude(x)
    x = bruit(x)
    x = masquage(x)
    if tache == "ecg":
        x = inversion_signe(x)
    return normaliser_lot(x)


# ── Perte InfoNCE ─────────────────────────────────────────────────────────
def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """NT-Xent : pour chaque vue, la vue jumelle doit être la plus proche parmi les 2B−1 autres.

    z1, z2 : (B, d), projections des deux vues du même lot, dans le même ordre.
    """
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)            # (2B, d)
    sim = z @ z.t() / temperature                                  # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))                              # pas soi-même
    cibles = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, cibles)


class ProjectionContrastive(nn.Module):
    """Encodeur + petite tête de projection (jetée après le pré-entraînement, comme dans SimCLR)."""

    def __init__(self, encodeur: Encodeur, dim_proj: int = 32):
        super().__init__()
        self.encodeur = encodeur
        self.proj = nn.Sequential(nn.ReLU(), nn.Linear(encodeur.dim, encodeur.dim), nn.ReLU(), nn.Linear(encodeur.dim, dim_proj))

    def forward(self, x):
        return self.proj(self.encodeur(x))


# ── Lots : une époque par personne ────────────────────────────────────────
class LotsParPersonne:
    """À chaque pas : un sous-ensemble de personnes, une époque tirée au hasard chez chacune."""

    def __init__(self, tableau: Tableau, taille_lot: int = 192, graine: int = 42):
        self.X = tableau.X
        pers = np.asarray(tableau.personne)
        self.indices_par_personne = [np.flatnonzero(pers == p) for p in sorted(set(pers))]
        self.taille_lot = min(taille_lot, len(self.indices_par_personne))
        self.rng = np.random.default_rng(graine)

    @property
    def n_personnes(self):
        return len(self.indices_par_personne)

    def lot(self) -> tuple[torch.Tensor, np.ndarray]:
        qui = self.rng.choice(len(self.indices_par_personne), size=self.taille_lot, replace=False)
        idx = np.array([self.rng.choice(self.indices_par_personne[q]) for q in qui])
        x = torch.from_numpy(self.X[idx].astype(np.float32)).unsqueeze(1)
        return x, qui


# ── Pré-entraînement ──────────────────────────────────────────────────────
def pre_entrainer(tache: str, train: Tableau, *, pas: int = 4000, taille_lot: int = 192, lr: float = 1e-3,
                  temperature: float = 0.1, graine: int = 42, journal=print, dev: torch.device | None = None):
    """Renvoie (encodeur pré-entraîné, courbe de perte). Aucune étiquette n'est lue."""
    from somnia.deep.train import appareil, fixer_graines

    fixer_graines(graine)
    dev = dev or appareil()
    modele = ProjectionContrastive(Encodeur()).to(dev)
    opt = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pas)
    lots = LotsParPersonne(train, taille_lot, graine)
    journal(f"pré-entraînement {tache} : {lots.n_personnes} personnes, {len(train):,} époques, "
            f"lots de {lots.taille_lot} (une époque par personne), {pas} pas, appareil {dev.type}")
    courbe = []
    modele.train()
    for i in range(1, pas + 1):
        x, _ = lots.lot()
        x = x.to(dev)
        z1, z2 = modele(vue(x, tache)), modele(vue(x, tache))
        loss = info_nce(z1, z2, temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        courbe.append(loss.item())
        if i % 200 == 0 or i == 1:
            journal(f"  pas {i:5d} | perte contrastive {np.mean(courbe[-200:]):.3f}")
    return modele.encodeur, courbe


def petit_lot_contrastif(tache: str, train: Tableau, n: int = 32, pas: int = 200, graine: int = 42, journal=print) -> dict:
    """Tâche 5.4 : sur 32 époques fixes, la perte contrastive doit descendre nettement."""
    from somnia.deep.train import appareil, fixer_graines

    fixer_graines(graine)
    dev = appareil()
    rng = np.random.default_rng(graine)
    idx = rng.choice(len(train), size=n, replace=False)
    x = torch.from_numpy(train.X[idx].astype(np.float32)).unsqueeze(1).to(dev)
    modele = ProjectionContrastive(Encodeur()).to(dev)
    opt = torch.optim.Adam(modele.parameters(), lr=1e-3)
    pertes = []
    for _ in range(pas):
        loss = info_nce(modele(vue(x, tache)), modele(vue(x, tache)))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        pertes.append(loss.item())
    hasard = float(np.log(2 * n - 1))
    journal(f"petit lot contrastif ({n} ex.) : perte {pertes[0]:.3f} -> {np.mean(pertes[-10:]):.3f} (hasard = {hasard:.2f})")
    return {"n": n, "pas": pas, "perte_debut": pertes[0], "perte_fin": float(np.mean(pertes[-10:])), "hasard": hasard,
            "ok": np.mean(pertes[-10:]) < 0.5 * pertes[0]}


# ── Sonde linéaire ────────────────────────────────────────────────────────
@torch.no_grad()
def encoder(encodeur: Encodeur, tableau: Tableau, dev: torch.device, batch: int = 1024) -> np.ndarray:
    """Vecteurs de l'encodeur gelé pour tout un tableau."""
    encodeur.eval()
    out = []
    for i in range(0, len(tableau), batch):
        x = torch.from_numpy(tableau.X[i:i + batch].astype(np.float32)).unsqueeze(1).to(dev)
        out.append(encodeur(normaliser_lot(x)).float().cpu().numpy())
    return np.concatenate(out)


def sonde_lineaire(encodeur: Encodeur, train: Tableau, val: Tableau, dev: torch.device, graine: int = 42):
    """Encodeur gelé, régression logistique dessus (tâche 5.6). Renvoie (proba_val, y_val)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Ztr, Zva = encoder(encodeur, train, dev), encoder(encodeur, val, dev)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=graine))
    clf.fit(Ztr, train.y)
    return clf.predict_proba(Zva), val.y
