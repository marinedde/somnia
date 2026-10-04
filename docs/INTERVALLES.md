# Intervalles de confiance (horizon 1.6)

*Généré le 2026-10-04 par `python_scripts/intervalles.py`. Validation SHHS, 40 personnes. Chaque mesure est la moyenne de trois entraînements (graines 42, 1, 2). Intervalles à 95 % par bootstrap sur les personnes (1 000 à 2 000 tirages) ; différences calculées sur les mêmes personnes.*

Lecture : `estimation [borne basse ; borne haute]`. Une différence est dite établie quand son intervalle ne contient pas zéro.

## Stades : EEG seul contre EEG + yeux + menton

| Mesure | EEG seul | EEG + yeux + menton | Différence | Établie ? |
|---|---|---|---|---|
| Kappa | 0.708 [0.654 ; 0.749] | 0.731 [0.687 ; 0.768] | +0.023 [+0.008 ; +0.039] | oui |
| Exactitude | 0.782 [0.742 ; 0.813] | 0.799 [0.766 ; 0.828] | +0.017 [+0.006 ; +0.029] | oui |
| Accord éveil / sommeil | 0.920 [0.885 ; 0.946] | 0.943 [0.917 ; 0.960] | +0.023 [+0.009 ; +0.041] | oui |
| F1 du N1 | 0.416 [0.366 ; 0.463] | 0.414 [0.364 ; 0.461] | -0.002 [-0.019 ; +0.016] | non |
| F1 du REM | 0.838 [0.795 ; 0.875] | 0.850 [0.790 ; 0.893] | +0.012 [-0.021 ; +0.040] | non |

## Événements respiratoires : v3 contre v4 (pendant le sommeil, de bout en bout)

| Mesure | v3 | v4 | Différence | Établie ? |
|---|---|---|---|---|
| F1 par événement | 0.734 [0.695 ; 0.769] | 0.736 [0.696 ; 0.771] | +0.002 [-0.005 ; +0.009] | non |
| Précision | 0.716 [0.670 ; 0.756] | 0.734 [0.689 ; 0.773] | +0.018 [+0.007 ; +0.028] | oui |
| Rappel | 0.755 [0.694 ; 0.818] | 0.739 [0.676 ; 0.806] | -0.016 [-0.028 ; -0.002] | oui |
| Rappel des événements avec désaturation ≥ 3 | 0.864 [0.805 ; 0.909] | 0.871 [0.818 ; 0.913] | +0.008 [-0.009 ; +0.030] | non |

## Index clinique : le réseau contre le simple compte des désaturations

| Contre | Mesure | Réseau (apnées + hypopnées avec désaturation) | Désaturations par heure | Différence | Établie ? |
|---|---|---|---|---|---|
| `ahi_a0h3` | Spearman | 0.96 [0.90 ; 0.98] | 0.91 [0.80 ; 0.96] | +0.05 [-0.01 ; +0.14] | non |
| `ahi_a0h3` | Erreur absolue médiane (/h) | 2.3 [1.1 ; 3.1] | 2.0 [1.3 ; 4.3] | +0.3 [-1.8 ; +1.1] | non |
| `ahi_a0h4` | Spearman | 0.90 [0.80 ; 0.96] | 0.91 [0.83 ; 0.95] | -0.01 [-0.09 ; +0.07] | non |
| `ahi_a0h4` | Erreur absolue médiane (/h) | 1.7 [0.9 ; 2.7] | 1.9 [0.9 ; 3.7] | -0.2 [-1.8 ; +0.9] | non |

## Ce qu'on peut dire, et ce qu'on ne peut pas

- **Établi** : ajouter les yeux et le menton améliore les stades (kappa +0.023 [+0.008 ; +0.039]) et l'accord éveil / sommeil (+0.023 [+0.009 ; +0.041]). La cible de 0,95 reste dans l'intervalle du modèle multi-capteurs : on ne peut ni dire qu'elle est atteinte, ni qu'elle est manquée.
- **Non établi** : un gain sur le N1 ou le REM.
- **Établi** : le réseau d'événements v4 n'a pas un meilleur F1 que le v3 (+0.002 [-0.005 ; +0.009]). Il échange un peu de rappel contre un peu de précision.
- **Non établi** : que l'index clinique du réseau suive mieux `ahi_a0h3` que le simple compte des désaturations (+0.05 [-0.01 ; +0.14], 94% des tirages favorables au réseau). Les deux sont bons ; 40 personnes ne suffisent pas à les départager.
- **La largeur des intervalles est la leçon principale** : avec 40 personnes, un kappa est connu à ± 0,04 et un F1 par événement à ± 0,04. Tout écart plus petit entre deux réglages ne se lit que par une comparaison appariée, et encore.

## Repère : l'accord entre scoreurs humains

Un modèle ne peut pas être plus d'accord avec un technicien que deux techniciens entre eux. Chiffres publiés, **cités de mémoire : à vérifier à la source avant toute publication**. SHHS a été scoré selon les règles de Rechtschaffen et Kales, pas selon celles de l'AASM : la comparaison est indicative.

| Mesure | Entre scoreurs humains (littérature) | Somnia, validation |
|---|---|---|
| Stades, accord global | environ 83 % (Rosenberg et Van Hout, 2013, programme inter-scoreurs de l'AASM) | 80% |
| Stades, kappa | environ 0,76 (Danker-Hopfe et al., 2009) | 0.73 |
| N1 | accord d'environ 63 %, le plus bas de tous les stades (Rosenberg et Van Hout, 2013) | F1 0,41 (mesure différente) |
| REM | accord d'environ 90 % | F1 0,85 (mesure différente) |
| Hypopnées | accord d'environ 65 %, contre environ 77 % pour les apnées obstructives (Rosenberg et Van Hout, 2014) | rappel 0,72 ; apnées 0,94 |

Lecture : le N1 et les hypopnées sont les deux endroits où Somnia est faible, et ce sont aussi les deux endroits où les humains s'accordent le moins. Une partie du plafond est dans l'étiquette, pas dans le modèle. Pour le mesurer vraiment, il faudrait des nuits scorées deux fois, ce que SHHS ne fournit pas.

## Ce que ces intervalles ne couvrent pas

- **La variation entre entraînements** : les mesures sont moyennées sur trois graines, ce qui l'atténue sans la supprimer. L'index clinique est celui de la sortie par nuit, une seule graine.
- **Le choix sur la validation** : architectures, seuils et variantes ont été choisis sur ces 40 personnes. Les chiffres sont donc un peu optimistes, et l'intervalle ne corrige pas ce biais. Seul un test neuf le fera.
- **La population** : 40 adultes de SHHS, enregistrés à domicile entre 1995 et 1998.
