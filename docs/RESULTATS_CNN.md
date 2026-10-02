# Résultats — réseaux convolutifs (étape 4)

*Généré le 2026-10-03 par `python_scripts/cnn_report.py`. Validation SHHS uniquement (40 personnes) : le test n'a pas été ouvert. Une graine par ligne ; l'écart entre graines reste à mesurer.*

## Stades de sommeil (EEG, époques de 30 s)

| Modèle | Personnes étiquetées | Exemples | Meilleure époque | accuracy | kappa | f1_macro | f1_N1 | ECE avant → après | Durée |
|---|---|---|---|---|---|---|---|---|---|
| Classe majoritaire | — | — | — | 0.414 | 0.000 | 0.117 | 0.000 | — | — |
| Random Forest, 16 caract. | 192 (100 %) | 192,291 | — | 0.700 | 0.596 | 0.619 | 0.210 | — | — |
| CNN 1D, de zéro | 192 (100%) | 192,291 | 7 | 0.733 | 0.641 | 0.660 | 0.280 | 0.036 → 0.050 | 2.3 min |
| CNN 1D, de zéro | 19 (10%) | 19,009 | 6 | 0.676 | 0.560 | 0.595 | 0.231 | 0.054 → 0.085 | 0.5 min |
| CNN 1D, de zéro | 2 (1%) | 1,804 | 3 | 0.367 | 0.217 | 0.298 | 0.108 | 0.103 → 0.114 | 0.3 min |

Courbe précision / couverture (modèle à 100 %) : on ne classe que les époques les plus sûres.

| Part gardée | Seuil de confiance | accuracy | kappa |
|---|---|---|---|
| 100% | 0.25 | 0.733 | 0.641 |
| 90% | 0.46 | 0.773 | 0.691 |
| 80% | 0.53 | 0.810 | 0.739 |
| 70% | 0.61 | 0.844 | 0.784 |
| 50% | 0.77 | 0.894 | 0.849 |

## Apnée (ECG, fenêtres de 60 s en sommeil)

| Modèle | Personnes étiquetées | Exemples | Meilleure époque | auc_roc | auc_pr | f1_apnee | ECE avant → après | Durée |
|---|---|---|---|---|---|---|---|---|
| Classe majoritaire | — | — | — | 0.500 | 0.426 | 0.000 | — | — |
| Random Forest, 16 caract. | 192 (100 %) | 66,035 | — | 0.650 | 0.563 | 0.595 | — | — |
| CNN 1D, de zéro | 192 (100%) | 66,035 | 9 | 0.625 | 0.542 | 0.483 | 0.071 → 0.040 | 1.8 min |
| CNN 1D, de zéro | 19 (10%) | 6,567 | 1 | 0.480 | 0.406 | 0.537 | 0.072 → 0.034 | 0.3 min |
| CNN 1D, de zéro | 2 (1%) | 593 | 3 | 0.467 | 0.406 | 0.570 | 0.098 → 0.087 | 0.3 min |

Courbe précision / couverture (modèle à 100 %) : on ne classe que les époques les plus sûres.

| Part gardée | Seuil de confiance | accuracy | f1_apnee |
|---|---|---|---|
| 100% | 0.50 | 0.616 | 0.483 |
| 90% | 0.53 | 0.625 | 0.480 |
| 80% | 0.57 | 0.633 | 0.472 |
| 70% | 0.60 | 0.643 | 0.471 |
| 50% | 0.67 | 0.659 | 0.461 |

Par personne (validation), corrélation de Spearman entre la part de minutes prédites positives et :

- la part annotée : 0.16
- l'index clinique ahi_a0h3a : 0.07
- l'index clinique ahi_a0h4 : 0.02

Référence Random Forest sur la même mesure : 0.17 avec la part annotée.

## Lecture

- Les lignes « 10 % » et « 1 % » mesurent la faim d'étiquettes du réseau : c'est le trou que le pré-entraînement de l'étape 5 doit combler.
- L'ECE est mesuré sur une moitié des personnes de validation, la température étant ajustée sur l'autre, puis l'inverse.
- Figures : `data/figures/cnn_courbes.png`, `data/figures/cnn_etiquettes.png`.
