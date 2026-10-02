# Résultats — Somnia

*Généré le 2026-10-03 par `python_scripts/evaluate_cv.py`. Ne pas éditer à la main.*

Deux chiffres par métrique : **avant** (découpage aléatoire par époque, celui des notebooks d'origine, fuité) et **après** (validation croisée par personne, 5 plis, moyenne ± écart-type). Même modèle, mêmes 16 caractéristiques, mêmes données. Seul le découpage change.

## Stades de sommeil (EEG, Sleep-EDF)

16 personnes, 28 enregistrements, 25,721 époques.

| Métrique | Avant : aléatoire par époque | Après : par personne (5 plis) |
|---|---|---|
| accuracy | 0.812 | 0.741 ± 0.063 |
| f1_macro | 0.735 | 0.651 ± 0.063 |
| kappa | 0.729 | 0.630 ± 0.084 |
| f1_Wake | 0.657 | 0.516 ± 0.137 |
| f1_N1 | 0.464 | 0.367 ± 0.057 |
| f1_N2 | 0.881 | 0.830 ± 0.072 |
| f1_N3 | 0.897 | 0.874 ± 0.035 |
| f1_REM | 0.774 | 0.667 ± 0.096 |

Détail par pli (personnes mises de côté) :

| Pli | n personnes | n époques | accuracy | f1_macro | kappa |
|---|---|---|---|---|---|
| 1 | 3 | 5,403 | 0.811 | 0.719 | 0.726 |
| 2 | 4 | 5,473 | 0.802 | 0.699 | 0.707 |
| 3 | 4 | 5,402 | 0.738 | 0.612 | 0.610 |
| 4 | 2 | 4,209 | 0.719 | 0.676 | 0.616 |
| 5 | 3 | 5,234 | 0.637 | 0.547 | 0.490 |

## Apnée (ECG, Apnea-ECG)

30 personnes, 31 enregistrements, 15,116 époques.

| Métrique | Avant : aléatoire par époque | Après : par personne (5 plis) |
|---|---|---|
| auc_roc | 0.969 | 0.838 ± 0.074 |
| auc_pr | 0.953 | 0.787 ± 0.066 |
| f1_apnee | 0.885 | 0.634 ± 0.107 |
| accuracy | 0.908 | 0.743 ± 0.042 |

Détail par pli (personnes mises de côté) :

| Pli | n personnes | n époques | auc_roc | auc_pr | f1_apnee |
|---|---|---|---|---|---|
| 1 | 6 | 3,364 | 0.927 | 0.886 | 0.746 |
| 2 | 6 | 2,975 | 0.893 | 0.826 | 0.727 |
| 3 | 6 | 3,007 | 0.838 | 0.761 | 0.469 |
| 4 | 6 | 2,854 | 0.711 | 0.690 | 0.550 |
| 5 | 6 | 2,916 | 0.819 | 0.772 | 0.680 |

## Lecture

- La baisse entre les deux colonnes mesure la **fuite** de l'ancien découpage : deux nuits de la même personne (Sleep-EDF) et des minutes voisines du même enregistrement (Apnea-ECG) se retrouvaient des deux côtés.
- Les écarts-types sont grands parce que les plis contiennent peu de personnes (3 à 4 pour l'EEG, 6 pour l'ECG). C'est la vraie incertitude de ce que l'on peut dire avec 16 et 30 personnes.
- Figures : `data/figures/fuite_roc_ecg.png` (vraies courbes ROC) et `data/figures/fuite_par_personne.png`.
