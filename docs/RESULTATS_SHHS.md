# Résultats — références SHHS (étape 3)

*Généré le 2026-10-03 par `python_scripts/shhs_baselines.py`. Ne pas éditer à la main.*

Règle : le **test SHHS reste fermé**. Tout ce qui est mesuré sur SHHS l'est sur les 40 personnes de validation. PhysioNet est utilisé en entier pour la validation externe du modèle SHHS, et par son propre découpage par personne pour le sens inverse.

## Stades de sommeil (EEG)

SHHS : 192 personnes / 192,291 lignes en entraînement, 40 personnes / 40,020 lignes en validation (35,239 sans époque aberrante). PhysioNet : 16 personnes, 25,721 lignes.

| Modèle | Entraîné sur | Mesuré sur | accuracy | f1_macro | kappa | f1_Wake | f1_N1 | f1_N2 | f1_N3 | f1_REM |
|---|---|---|---|---|---|---|---|---|---|---|
| Classe majoritaire | SHHS train | SHHS val | 0.414 | 0.117 | 0.000 | 0.000 | 0.000 | 0.585 | 0.000 | 0.000 |
| Random Forest, 16 caract. | SHHS train | SHHS val | 0.700 | 0.619 | 0.596 | 0.769 | 0.210 | 0.723 | 0.764 | 0.632 |
| Random Forest, 16 caract. | SHHS train | SHHS val, sans aberrantes | 0.711 | 0.634 | 0.611 | 0.766 | 0.216 | 0.750 | 0.788 | 0.653 |
| Random Forest, 16 caract. | SHHS train | **PhysioNet, tout** (externe) | 0.548 | 0.482 | 0.401 | 0.498 | 0.159 | 0.546 | 0.657 | 0.549 |
| Random Forest, 16 caract. | PhysioNet train | PhysioNet val | 0.786 | 0.669 | 0.691 | 0.633 | 0.202 | 0.887 | 0.920 | 0.704 |
| Random Forest, 16 caract. | PhysioNet train | **SHHS val** (externe) | 0.565 | 0.432 | 0.389 | 0.681 | 0.103 | 0.639 | 0.354 | 0.380 |

## Apnée (ECG, fenêtres de 60 s en sommeil)

SHHS : 192 personnes / 66,035 lignes en entraînement, 40 personnes / 14,044 lignes en validation (13,011 sans époque aberrante). PhysioNet : 30 personnes, 15,116 lignes.

| Modèle | Entraîné sur | Mesuré sur | auc_roc | auc_pr | f1_apnee | accuracy |
|---|---|---|---|---|---|---|
| Classe majoritaire | SHHS train | SHHS val | 0.500 | 0.426 | 0.000 | 0.574 |
| Random Forest, 16 caract. | SHHS train | SHHS val | 0.650 | 0.563 | 0.595 | 0.594 |
| Random Forest, 16 caract. | SHHS train | SHHS val, sans aberrantes | 0.657 | 0.588 | 0.602 | 0.600 |
| Random Forest, 16 caract. | SHHS train | **PhysioNet, tout** (externe) | 0.706 | 0.595 | 0.625 | 0.630 |
| Random Forest, 16 caract. | PhysioNet train | PhysioNet val | 0.779 | 0.737 | 0.662 | 0.710 |
| Random Forest, 16 caract. | PhysioNet train | **SHHS val** (externe) | 0.572 | 0.492 | 0.349 | 0.592 |

Par personne (validation, 40 personnes) : part de minutes positives annotée contre prédite, corrélation 0.17, erreur absolue médiane 0.228 (moyenne 0.253).

## Lecture

- La ligne « classe majoritaire » est le plancher : un score qui ne la dépasse pas nettement ne vaut rien.
- Les deux lignes **externe** sont celles qui comptent en santé : un modèle entraîné dans un centre, testé dans un autre.
- Les écarts SHHS → PhysioNet et PhysioNet → SHHS ne sont pas symétriques : populations, appareils, et pour l'ECG définitions d'étiquettes différentes (minutes annotées contre paires d'époques).
