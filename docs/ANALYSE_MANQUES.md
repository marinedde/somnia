# Analyse des événements manqués (horizon 1.1)

*Généré le 2026-10-04 par `python_scripts/resp_manques.py`. Validation SHHS, 40 nuits, 8,467 événements du technicien pendant le sommeil. Réseau d'événements v3, sommeil prédit par le modèle de séquence. Agrégats seulement.*

## 1. Le compte

| Statut | Événements | Part |
|---|---|---|
| trouvé | 6,371 | 75.2% |
| possible | 616 | 7.3% |
| manqué | 1,480 | 17.5% |

## 2. Qui sont les manqués ?

| | Manqués | Trouvés |
|---|---|---|
| Hypopnées | 98% | 86% |
| Durée médiane | 15 s | 20 s |
| Durée de moins de 15 s | 46% | 18% |
| **Avec une désaturation ≥ 3 points** | **21%** | **49%** |
| Avec une désaturation ≥ 4 points | 9% | 27% |
| Chute de saturation médiane | 2.0 | 2.7 |

## 3. Pourquoi sont-ils manqués ?

| Cause (exclusive, dans cet ordre) | Part des manqués |
|---|---|
| Le réseau a vu quelque chose (P ≥ 0,7), mais moins de 10 s d'affilée | 32% |
| Signal faible (P entre 0,4 et 0,7), trop bref pour une zone « possible » | 29% |
| Le réseau n'a rien vu (P < 0,4 sur tout l'événement) | 23% |
| Le réseau de stades croyait le patient éveillé : la proposition a été écartée ou jamais faite | 16% |

## 4. Le rappel selon ce qui compte cliniquement

| Événements du technicien | Nombre | Trouvés | Trouvés ou possibles | Manqués |
|---|---|---|---|---|
| Tous | 8,467 | 75% | 83% | 17% |
| Avec désaturation ≥ 3 points | 3,559 | 88% | 91% | 9% |
| Avec désaturation ≥ 4 points | 1,913 | 90% | 93% | 7% |
| Sans désaturation (< 3 points) | 4,908 | 66% | 76% | 24% |
| Apnées | 949 | 95% | 97% | 3% |
| Hypopnées | 7,518 | 73% | 81% | 19% |
| Hypopnées avec désaturation ≥ 3 | 2,802 | 85% | 90% | 10% |
| Hypopnées sans désaturation | 4,716 | 65% | 76% | 24% |

## 5. Par stade et par nuit

| Stade (technicien) | Événements | Manqués |
|---|---|---|
| N1 | 620 | 20% |
| N2 | 4,679 | 18% |
| N3 | 726 | 21% |
| REM | 2,442 | 14% |

Concentration : les 5 nuits qui manquent le plus d'événements en totalisent 36% (sur 40 nuits) ; la médiane est de 21 manqués par nuit.

## Lecture

- **Les manqués sont presque tous des hypopnées courtes et sans désaturation.** Ce sont les événements les plus discutables de l'annotation : sans chute de saturation, ils ne comptent pas dans l'index clinique SHHS, et deux techniciens ne les marquent pas toujours de la même façon (hypothèse non vérifiée ici, faute de double scoring).
- **Le chiffre qui compte cliniquement est celui des événements avec désaturation** : c'est sur cette ligne du tableau 4 qu'il faut juger l'outil, pas sur le total.
- **La plupart des manqués ne sont pas invisibles** : dans la majorité des cas le réseau a réagi, mais trop brièvement pour franchir la règle des 10 s. Abaisser cette règle ferait remonter les fausses propositions (testé à l'étape précédente) ; la piste propre est d'élargir les zones « événement possible ».
- **Le réseau de stades n'explique qu'une petite part des manqués** : mieux séparer éveil et sommeil (horizon 1.3) aidera surtout la précision, peu le rappel.
- Réserve : la désaturation est mesurée ici par une règle simple (maximum des 30 s avant, minimum jusqu'à 45 s après), pas par l'annotation du technicien.
