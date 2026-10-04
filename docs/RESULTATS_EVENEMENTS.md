# Résultats — détection d'événements respiratoires

*Généré le 2026-10-04 par `python_scripts/resp_report.py`. Validation SHHS (40 personnes) ; le test n'est pas ouvert pour cette tâche. 3 graine(s).*

## Ce qui est mesuré

Le réseau lit 5 minutes de flux, de ceintures thoracique et abdominale et de saturation (10 Hz), et rend pour chaque seconde : rien, apnée ou hypopnée. Les secondes positives sont regroupées en événements d'au moins 10 s. La référence est l'annotation du technicien (apnées obstructives, centrales, mixtes et hypopnées, toutes comptées).

Entraînement : 191 nuits. Réseau : 143,619 paramètres, meilleure époque 6, 2.9 min sur mps.

## Par événement

| Critère d'appariement | Précision | Rappel | F1 |
|---|---|---|---|
| Tout recouvrement | 0.654 ± 0.014 | 0.745 ± 0.023 | 0.696 ± 0.002 |
| Recouvrement IoU ≥ 0,3 (bornes exigeantes) | 0.634 ± 0.014 | 0.722 ± 0.022 | 0.674 ± 0.002 |

Rappel par type : apnées 0.836 ± 0.014 (1,388 dans la validation), hypopnées 0.736 ± 0.025 (8,808).

Par seconde : F1 « événement en cours » 0.631 ± 0.002 ; 18.5% des secondes sont en événement dans la référence, 21.1% dans la prédiction.

## Par personne : l'index

| Estimateur | Contre | Spearman | Erreur absolue médiane (/h) | Biais (/h) |
|---|---|---|---|---|
| Réseau (événements détectés / h de sommeil) | index annoté | 0.841 ± 0.008 | 5.9 ± 0.9 | 0.7 ± 1.7 |
| Réseau | index clinique `ahi_a0h3a` | 0.874 ± 0.014 | 15.9 ± 1.4 | 17.1 ± 1.7 |
| Réseau | index clinique `ahi_a0h4` | 0.771 ± 0.026 | 25.3 ± 2.0 | 25.7 ± 1.7 |
| Désaturations ≥ 3 % / h (référence simple) | index annoté | 0.588 | 22.9 | -20.4 |
| Désaturations ≥ 3 % / h (référence simple) | index clinique `ahi_a0h3a` | 0.776 | 5.6 | -4.0 |
| Désaturations ≥ 3 % / h (référence simple) | index clinique `ahi_a0h4` | 0.868 | 4.2 | +4.6 |
| Désaturations ≥ 4 % / h (référence simple) | index annoté | 0.597 | 28.4 | -27.9 |
| Désaturations ≥ 4 % / h (référence simple) | index clinique `ahi_a0h3a` | 0.793 | 10.4 | -11.5 |
| Désaturations ≥ 4 % / h (référence simple) | index clinique `ahi_a0h4` | 0.919 | 1.9 | -2.9 |

Accord réseau / index annoté (Bland-Altman, graine 1) : biais +0.5 événements par heure, limites d'accord à 95 % de -18.4 à +19.4.

## Lecture

- **Deux index, deux questions.** L'index annoté compte toutes les hypopnées marquées par le technicien ; l'index clinique SHHS ne garde que celles suivies d'une désaturation. Le réseau apprend le premier ; la référence par désaturation colle au second par construction.
- **La référence par désaturation est le chiffre à battre pour l'index clinique** : compter les chutes de saturation suffit presque à retrouver l'index clinique à 4 %. Ce que le réseau apporte en plus, c'est la **position** de chaque événement : c'est ce qui prend du temps à un lecteur.
- **Le F1 par événement est la mesure du temps gagné** : un événement bien placé est un événement à valider d'un clic au lieu de le chercher et de le marquer. Le rappel par type dit ce que le lecteur devra encore trouver seul.
- Figure : `data/figures/evenements_index.png`.
