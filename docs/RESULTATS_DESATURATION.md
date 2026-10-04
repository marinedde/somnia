# Résultats — les règles de saturation contre l'annotation du technicien (horizon 2.1)

*Généré le 2026-10-04 par `python_scripts/desat_validation.py`. Validation SHHS, 40 nuits, 6,262 désaturations marquées. Agrégats seulement.*

SHHS marque toutes les chutes de saturation, même légères, et donne pour chacune son nadir et sa ligne de base. Profondeur des marques : médiane 2 point(s) ; 38% font au moins 3 points, 22% au moins 4. Toutes les comparaisons sont faites à profondeur égale.

## 1. Chaque désaturation : la règle contre le technicien

Règle : la saturation passe N points sous le maximum des deux minutes précédentes. Appariement un à un, par recouvrement.

| Profondeur minimale | Marquées | Trouvées par la règle | Précision | Rappel | F1 | Compte par nuit : Spearman |
|---|---|---|---|---|---|---|
| 2 points | 3,926 | 6,920 | 0.33 | 0.58 | 0.42 | 0.63 |
| 3 points | 2,352 | 5,283 | 0.31 | 0.70 | 0.43 | 0.74 |
| 4 points | 1,400 | 2,758 | 0.33 | 0.64 | 0.43 | 0.81 |

## 2. « Avec désaturation » : la règle par événement contre le technicien

Pour chaque événement respiratoire du technicien pendant le sommeil : y a-t-il une désaturation marquée d'au moins N points entre le début de l'événement et 45 s après sa fin ? La règle (`chute_de_saturation`) dit-elle la même chose ?

| Profondeur minimale | Événements | Technicien : avec désaturation | Règle : avec désaturation | Accord | Kappa |
|---|---|---|---|---|---|
| 2 points | 8,467 | 43% | 51% | 62% | 0.24 |
| 3 points | 8,467 | 28% | 42% | 72% | 0.39 |
| 4 points | 8,467 | 18% | 23% | 83% | 0.48 |

## 3. Oxymètre décollé : la règle contre les artefacts marqués

| | |
|---|---|
| Temps marqué « artefact » par le technicien | 4.3% de l'enregistrement |
| Temps « saturation invalide » pour la règle (hors de 50–100 %) | 5.3% |
| Part du temps invalide pour la règle qui est aussi marquée par le technicien | 65% |
| Part du temps marqué par le technicien que la règle retrouve | 79% |

## 4. La règle isolée du réseau : index à partir des événements du technicien

Événements et sommeil du technicien ; seule change la façon de décider qu'une hypopnée « a désaturé » (3 points). Contre `ahi_a0h3`.

| Lien hypopnée – désaturation | Spearman | Erreur absolue médiane | Biais |
|---|---|---|---|
| **Règle du code** (30 s avant, 45 s après) | 0.984 | 0.8 / h | +0.9 / h |
| Règle, 30 s après (sensibilité) | 0.984 | 0.9 / h | +0.2 / h |
| Règle, 20 s après (sensibilité) | 0.983 | 1.1 / h | -0.5 / h |
| Marques de désaturation de l'annotation | 0.921 | 1.7 / h | -2.6 / h |

## Lecture

- **Les temps et les profondeurs sont justes** : le nadir écrit dans l'annotation se retrouve dans le signal à la même seconde, et la profondeur mesurée sur le signal dans chaque marque est la même en médiane (-0.0 point).
- **La règle trouve environ deux fois plus de chutes que les marques**, et l'accord événement par événement est moyen (kappa 0.39 à 3 points). Les marques du fichier sont donc une référence incomplète : elles ne reprennent pas toutes les chutes visibles dans le signal.
- **C'est l'index qui tranche, et il donne raison à la règle** : avec les événements du technicien, lier les hypopnées aux désaturations par la règle redonne l'index officiel de SHHS presque exactement ; le faire avec les marques le sous-estime. La règle est donc validée indépendamment du réseau.
- **La fenêtre n'a pas été réglée** : 45 s après la fin était le choix de départ. Les lignes « sensibilité » montrent que le résultat y est peu sensible ; la fenêtre reste à 45 s pour ne pas régler sur la validation.
- **Oxymètre décollé** : la règle « hors de 50–100 % » retrouve quatre cinquièmes du temps marqué artefact ; un tiers de ce qu'elle signale n'est pas marqué.
