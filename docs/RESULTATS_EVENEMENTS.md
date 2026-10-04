# Résultats — détection d'événements respiratoires

*Généré le 2026-10-04 par `python_scripts/resp_report.py`. Validation SHHS (40 personnes) ; le test n'est pas ouvert pour cette tâche. Moyenne ± écart-type quand plusieurs graines.*

## Ce qui est mesuré

Le réseau lit 5 minutes de flux, de ceintures thoracique et abdominale et de saturation (10 Hz), et rend pour chaque seconde : rien, apnée ou hypopnée. Les secondes positives sont regroupées en événements d'au moins 10 s. La référence est l'annotation du technicien (apnées obstructives, centrales, mixtes et hypopnées, toutes comptées).

Deux façons de mesurer :

- **tous les événements** du technicien, y compris ceux marqués pendant l'éveil ;
- **pendant le sommeil, de bout en bout** : la référence est limitée aux événements qui commencent pendant le sommeil du technicien (ceux qui comptent dans l'index) ; les propositions, à celles qui commencent pendant le sommeil **prédit** par le réseau de stades ; l'index est calculé sur le temps de sommeil prédit. Rien n'est emprunté au technicien.

## Par événement

| Modèle et mesure | Précision | Rappel | F1 | F1, bornes exigeantes (IoU ≥ 0,3) | Rappel apnées | Rappel hypopnées |
|---|---|---|---|---|---|---|
| v1, 4 canaux, tous les événements (3 graines) | 0.654 ± 0.014 | 0.745 ± 0.023 | 0.696 ± 0.002 | 0.674 ± 0.002 | 0.836 ± 0.014 | 0.736 ± 0.025 |
| v1, pendant le sommeil (1 graine) | 0.660 | 0.764 | 0.708 | 0.684 | 0.865 | 0.757 |
| **v2, pendant le sommeil** (3 graines) | 0.728 ± 0.013 | 0.725 ± 0.017 | 0.726 ± 0.003 | 0.712 ± 0.003 | 0.907 ± 0.007 | 0.704 ± 0.019 |
| v2, tous les événements (3 graines) | 0.724 ± 0.015 | 0.691 ± 0.015 | 0.707 ± 0.002 | 0.689 ± 0.003 | 0.778 ± 0.015 | 0.680 ± 0.018 |

v2 = v1 + un cinquième canal (la probabilité de sommeil prédite par le réseau de stades), une pondération des classes adoucie (racine carrée) et un gain aléatoire sur les capteurs à l'entraînement.

## Ce qui a été essayé, et ce que ça a donné

Diagnostic de la v1 sur la validation : **47 % des fausses propositions commençaient pendant l'éveil**, 44 % étaient à moins d'une minute d'un vrai événement, et un vote de trois graines n'apportait que +0,004 de F1.

| Expérience (une graine, mesure « pendant le sommeil ») | Précision | Rappel | F1 | Index : Spearman |
|---|---|---|---|---|
| A. référence, 4 canaux | 0.660 | 0.764 | 0.708 | 0.836 |
| B. + canal de sommeil prédit | 0.682 | 0.758 | 0.718 | 0.840 |
| C. + pondération adoucie (retenu : v2) | 0.722 | 0.725 | 0.724 | 0.818 |
| D. + réseau plus large et plus profond (96, 8 blocs) | 0.682 | 0.748 | 0.714 | 0.817 |
| E. comme B, avec 25 % des nuits d'entraînement | 0.663 | 0.750 | 0.704 | 0.777 |
| F. comme B, avec 50 % des nuits | 0.669 | 0.726 | 0.696 | 0.824 |

- **Le sommeil compte plus que la taille du réseau.** Dire au réseau si le patient dort améliore la précision ; l'élargir ne change rien.
- **Plus de nuits du même type n'aideraient pas** : avec un quart des nuits, le score est presque le même. La courbe d'apprentissage est plate.
- **Borne haute mesurée** : avec le sommeil du technicien à la place du sommeil prédit, le même réseau atteint un F1 de 0,77. L'écart restant vient donc du réseau de stades (accord éveil/sommeil de 91 % par seconde), pas du réseau d'événements.
- Lisser le sommeil prédit ou changer son seuil ne change rien (F1 entre 0,717 et 0,725). Les règles de regroupement (10 s minimum, trous de 3 s) et le seuil de décision (0,7) sont au bon endroit.
- Réserve : six expériences ont été comparées sur les mêmes 40 personnes de validation ; l'écart entre réglages voisins (0,01) est du même ordre que le bruit entre graines. Le test tranchera, une fois.

## Par personne : l'index

| Estimateur | Contre | Spearman | Erreur absolue médiane (/h) | Biais (/h) |
|---|---|---|---|---|
| Réseau v2, de bout en bout | index annoté | 0.834 ± 0.011 | 5.8 ± 0.3 | -0.7 ± 1.3 |
| Réseau v2, de bout en bout | index clinique `ahi_a0h3a` | 0.863 ± 0.009 | 13.9 ± 1.2 | 15.7 ± 1.3 |
| Réseau v2, de bout en bout | index clinique `ahi_a0h4` | 0.757 ± 0.018 | 23.3 ± 2.1 | 24.3 ± 1.3 |
| Réseau v1 (sommeil du technicien) | index annoté | 0.841 ± 0.008 | 5.9 ± 0.9 | 0.7 ± 1.7 |
| Réseau v1 (sommeil du technicien) | index clinique `ahi_a0h3a` | 0.874 ± 0.014 | 15.9 ± 1.4 | 17.1 ± 1.7 |
| Réseau v1 (sommeil du technicien) | index clinique `ahi_a0h4` | 0.771 ± 0.026 | 25.3 ± 2.0 | 25.7 ± 1.7 |
| Désaturations ≥ 3 % / h (référence simple) | index clinique `ahi_a0h3a` | 0.776 | 5.6 | -4.0 |
| Désaturations ≥ 3 % / h (référence simple) | index clinique `ahi_a0h4` | 0.868 | 4.2 | +4.6 |
| Désaturations ≥ 4 % / h (référence simple) | index clinique `ahi_a0h3a` | 0.793 | 10.4 | -11.5 |
| Désaturations ≥ 4 % / h (référence simple) | index clinique `ahi_a0h4` | 0.919 | 1.9 | -2.9 |

## Lecture

- **Deux index, deux questions.** L'index annoté compte toutes les hypopnées marquées par le technicien ; l'index clinique SHHS ne garde que celles suivies d'une désaturation. Le réseau apprend le premier ; la référence par désaturation colle au second par construction.
- **La référence par désaturation est le chiffre à battre pour l'index clinique** : compter les chutes de saturation suffit presque. Ce que le réseau apporte, c'est la **position** de chaque événement : c'est ce qui prend du temps à un lecteur.
- **Le prochain gain est dans le réseau de stades**, pas ici : mieux séparer éveil et sommeil rapprocherait de la borne de 0,77.
- Au-delà, la limite probable est l'annotation elle-même : marquer une hypopnée sans critère de désaturation est une décision où deux techniciens ne sont pas toujours d'accord. Hypothèse non vérifiée ici, faute de double scoring.
- Figure : `data/figures/evenements_index.png`.
