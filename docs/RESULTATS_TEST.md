# Résultats sur le TEST SHHS — ouvert une seule fois

*2026-10-03. 40 personnes jamais vues, ni pour entraîner, ni pour choisir quoi que ce soit. La liste des modèles a été écrite avant d'exécuter ; ce fichier ne sera pas régénéré.*

## Stades (EEG) — 40 personnes, 40,276 lignes

| Modèle | accuracy | kappa | f1_macro | f1_N1 |
|---|---|---|---|---|
| Classe majoritaire | 0.433 | 0.000 | 0.121 | 0.000 |
| Random Forest SHHS, 16 caract. | 0.645 | 0.524 | 0.565 | 0.170 |
| Random Forest PhysioNet (externe) | 0.523 | 0.336 | 0.398 | 0.078 |
| CNN de zéro, 100 % | 0.680 | 0.572 | 0.603 | 0.181 |
| CNN pré-entraîné affiné, 10 % | 0.646 | 0.523 | 0.564 | 0.157 |
| CNN pré-entraîné affiné, 1 % | 0.404 | 0.263 | 0.308 | 0.000 |

## Apnée (ECG) — 40 personnes, 13,954 lignes

| Modèle | auc_roc | auc_pr | f1_apnee | spearman_ahi_a0h3a |
|---|---|---|---|---|
| Classe majoritaire | 0.500 | 0.423 | 0.000 | nan |
| Random Forest SHHS, 16 caract. | 0.672 | 0.585 | 0.588 | 0.322 |
| Random Forest PhysioNet (externe) | 0.569 | 0.506 | 0.258 | nan |
| CNN de zéro, 100 % | 0.624 | 0.535 | 0.497 | 0.314 |

## Lecture

À comparer aux chiffres de validation (`docs/RESULTATS_SHHS.md`, `docs/RESULTATS_CNN.md`) : un écart important entre validation et test signifierait que la validation a été trop regardée.
