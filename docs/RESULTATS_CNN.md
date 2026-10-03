# Résultats — réseaux convolutifs (étape 4)

*Généré le 2026-10-03 par `python_scripts/cnn_report.py`. Validation SHHS uniquement (40 personnes) : le test a été ouvert une seule fois à la fin (`docs/RESULTATS_TEST.md`). Quand plusieurs graines existent, moyenne ± écart-type.*

## Stades de sommeil (EEG, époques de 30 s)

| Modèle | Personnes étiquetées | Exemples | Meilleure époque | accuracy | kappa | f1_macro | f1_N1 | ECE avant → après | Durée |
|---|---|---|---|---|---|---|---|---|---|
| Classe majoritaire | — | — | — | 0.414 | 0.000 | 0.117 | 0.000 | — | — |
| Random Forest, 16 caract. | 192 (100 %) | 192,291 | — | 0.700 | 0.596 | 0.619 | 0.210 | — | — |
| *Pré-entraînement contrastif (sans étiquette)* | 192 (signaux seuls) | 192,291 | 4000 pas | — | — | — | — | perte 5.87 → 0.19 | 1.6 min |
| CNN 1D, de zéro | 192 (100%) | 192,291 | 7 (3 graines) | 0.732 ± 0.006 | 0.642 ± 0.008 | 0.658 ± 0.006 | 0.282 ± 0.007 | 0.039 → 0.050 | 2.5 min |
| CNN 1D, de zéro | 19 (10%) | 19,033 | 6 (3 graines) | 0.693 ± 0.015 | 0.584 ± 0.019 | 0.607 ± 0.008 | 0.213 ± 0.022 | 0.060 → 0.107 | 0.6 min |
| CNN 1D, de zéro | 2 (1%) | 1,965 | 3 (3 graines) | 0.453 ± 0.087 | 0.254 ± 0.122 | 0.338 ± 0.118 | 0.118 ± 0.037 | 0.115 → 0.124 | 0.4 min |
| CNN 1D, sonde linéaire (encodeur gelé) | 192 (100%) | 192,291 | — (3 graines) | 0.676 ± 0.007 | 0.571 ± 0.010 | 0.607 ± 0.006 | 0.247 ± 0.004 | 0.054 → 0.082 | 0.2 min |
| CNN 1D, sonde linéaire (encodeur gelé) | 19 (10%) | 19,033 | — (3 graines) | 0.654 ± 0.001 | 0.542 ± 0.002 | 0.583 ± 0.003 | 0.220 ± 0.007 | 0.072 → 0.121 | 0.0 min |
| CNN 1D, pré-entraîné, affiné | 192 (100%) | 192,291 | 9 (3 graines) | 0.728 ± 0.014 | 0.638 ± 0.020 | 0.657 ± 0.014 | 0.273 ± 0.010 | 0.051 → 0.076 | 2.1 min |
| CNN 1D, pré-entraîné, affiné | 19 (10%) | 19,033 | 8 (3 graines) | 0.686 ± 0.006 | 0.581 ± 0.011 | 0.612 ± 0.009 | 0.230 ± 0.005 | 0.067 → 0.115 | 0.4 min |
| CNN 1D, pré-entraîné, affiné | 2 (1%) | 1,965 | 10 (3 graines) | 0.525 ± 0.088 | 0.369 ± 0.077 | 0.425 ± 0.082 | 0.104 ± 0.074 | 0.101 → 0.190 | 0.2 min |

Courbe précision / couverture (modèle à 100 %) : on ne classe que les époques les plus sûres.

| Part gardée | Seuil de confiance | accuracy | kappa |
|---|---|---|---|
| 100% | 0.23 | 0.724 | 0.631 |
| 90% | 0.47 | 0.763 | 0.680 |
| 80% | 0.55 | 0.798 | 0.724 |
| 70% | 0.63 | 0.833 | 0.769 |
| 50% | 0.81 | 0.878 | 0.828 |

## Apnée (ECG, fenêtres de 60 s en sommeil)

| Modèle | Personnes étiquetées | Exemples | Meilleure époque | auc_roc | auc_pr | f1_apnee | ECE avant → après | Durée |
|---|---|---|---|---|---|---|---|---|
| Classe majoritaire | — | — | — | 0.500 | 0.426 | 0.000 | — | — |
| Random Forest, 16 caract. | 192 (100 %) | 66,035 | — | 0.650 | 0.563 | 0.595 | — | — |
| *Pré-entraînement contrastif (sans étiquette)* | 192 (signaux seuls) | 66,035 | 4000 pas | — | — | — | perte 5.92 → 0.08 | 3.0 min |
| CNN 1D, de zéro | 192 (100%) | 66,035 | 5 (3 graines) | 0.615 ± 0.007 | 0.535 ± 0.005 | 0.484 ± 0.021 | 0.056 → 0.033 | 1.4 min |
| CNN 1D, de zéro | 19 (10%) | 6,567 | 1 | 0.480 | 0.406 | 0.537 | 0.072 → 0.034 | 0.3 min |
| CNN 1D, de zéro | 2 (1%) | 593 | 3 | 0.467 | 0.406 | 0.570 | 0.098 → 0.087 | 0.3 min |
| CNN 1D, sonde linéaire (encodeur gelé) | 192 (100%) | 66,035 | — | 0.550 | 0.473 | 0.499 | 0.074 → 0.033 | 0.0 min |
| CNN 1D, sonde linéaire (encodeur gelé) | 19 (10%) | 6,567 | — | 0.494 | 0.406 | 0.510 | 0.298 → 0.082 | 0.0 min |
| CNN 1D, pré-entraîné, affiné | 192 (100%) | 66,035 | 5 | 0.625 | 0.547 | 0.524 | 0.059 → 0.036 | 1.1 min |
| CNN 1D, pré-entraîné, affiné | 19 (10%) | 6,567 | 2 | 0.502 | 0.411 | 0.510 | 0.256 → 0.083 | 0.1 min |
| CNN 1D, pré-entraîné, affiné | 2 (1%) | 593 | 6 | 0.471 | 0.422 | 0.547 | 0.181 → 0.065 | 0.1 min |

Courbe précision / couverture (modèle à 100 %) : on ne classe que les époques les plus sûres.

| Part gardée | Seuil de confiance | accuracy | f1_apnee |
|---|---|---|---|
| 100% | 0.50 | 0.604 | 0.458 |
| 90% | 0.54 | 0.615 | 0.452 |
| 80% | 0.57 | 0.628 | 0.449 |
| 70% | 0.60 | 0.635 | 0.439 |
| 50% | 0.67 | 0.650 | 0.427 |

Par personne (validation), corrélation de Spearman entre la part de minutes prédites positives et :

- la part annotée : 0.18
- l'index clinique ahi_a0h3a : 0.06
- l'index clinique ahi_a0h4 : 0.06

Référence Random Forest sur la même mesure : 0.17 avec la part annotée.

## Lecture

- Les lignes « 10 % » et « 1 % » mesurent la faim d'étiquettes ; les lignes « pré-entraîné » disent ce que l'auto-supervisé en comble.
- Sonde linéaire : encodeur gelé + régression logistique ; affiné : tout le réseau réentraîné à partir de l'encodeur pré-entraîné.
- L'ECE est mesuré sur une moitié des personnes de validation, la température étant ajustée sur l'autre, puis l'inverse.
- Figures : `data/figures/cnn_courbes.png`, `data/figures/cnn_etiquettes.png`.
