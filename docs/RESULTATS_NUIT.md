# Résultats — la sortie par nuit complète

*Généré le 2026-10-04 par `python_scripts/nuit_complete.py` sur les 40 nuits de validation SHHS. Agrégats seulement : le détail d'une nuit reste hors dépôt. Le test n'est pas touché.*

Pour chaque nuit, deux réseaux tournent : celui des stades (EEG, yeux, menton) et celui des événements respiratoires (flux, ceintures, saturation, et le sommeil prédit par le premier). La sortie réunit l'hypnogramme, les événements proposés pendant le sommeil prédit avec leur confiance, l'index, et une file de relecture commune. La référence est l'ensemble des événements que le technicien a marqués pendant le sommeil : ceux qui comptent dans l'index.

## 1. Ce qu'il reste à relire

| Par nuit, médiane (quartiles) | |
|---|---|
| Durée de l'enregistrement | 510 (480–534) min |
| Époques dont le stade est à relire | 15 (13–22) % |
| Événements respiratoires de référence | 212 (119–302) |
| Événements proposés | 191 (119–278) dont sûrs 106 (58–161), à relire 70 (55–93) |
| « Événements possibles » signalés (non proposés) | 75 (49–93) |
| **Signal à relire, stades et événements réunis** | **191 (165–226) min, soit 38 (32–47) % de la nuit** |

Le signal à relire est l'union des passages concernés, avec 30 s de contexte autour de chaque événement. C'est une durée de signal, pas un temps de lecture : seul un chronomètre avec un lecteur dira le temps gagné.

## 2. La confiance trie-t-elle les événements ?

| Niveau | Événements proposés | Part qui recouvre un événement du technicien |
|---|---|---|
| Sûrs (confiance ≥ 0,85) | 5,171 | **91%** |
| À relire (confiance < 0,85) | 3,064 | 60% |

Plus on exige de confiance, plus les propositions sont justes, et moins il y en a :

| Confiance minimale | Part des événements proposés | Précision | Événements par nuit | Événements justes / événements de référence |
|---|---|---|---|---|
| 0.70 | 99% | 80% | 204 | 77% |
| 0.80 | 80% | 86% | 165 | 67% |
| 0.85 | 63% | 91% | 130 | 56% |
| 0.90 | 42% | 96% | 87 | 39% |
| 0.95 | 16% | 98% | 33 | 15% |

## 3. Où tombent les événements du technicien

| | Part des événements de référence |
|---|---|
| Dans un événement proposé comme sûr | 54% |
| Dans un événement proposé à relire | 20% |
| Dans un « événement possible » signalé | 7% |
| Nulle part : à trouver par le lecteur | **18%** |

## 4. L'index, de bout en bout

Sommeil prédit par le réseau de stades, événements prédits par le réseau respiratoire, contre l'index calculé avec le sommeil et les événements du technicien.

| Contre | Spearman | Erreur absolue médiane | Biais |
|---|---|---|---|
| Index annoté (technicien) | 0.81 | 6.5 / h | -2.0 / h |
| Tous les événements proposés, contre l'index clinique `ahi_a0h3a` | 0.86 | — | — |
| Tous les événements proposés, contre l'index clinique `ahi_a0h4` | 0.77 | — | — |
| **Index clinique estimé** (apnées + hypopnées avec désaturation ≥ 3 points), contre `ahi_a0h3` | **0.96** | 2.3 / h | +1.5 / h |
| **Index clinique estimé** (apnées + hypopnées avec désaturation ≥ 4 points), contre `ahi_a0h4` | **0.90** | 1.7 / h | +0.9 / h |

Temps de sommeil : prédit 6.3 (5.8–6.9) h, technicien 6.0 (5.5–6.7) h.

## Lecture

- Les événements « sûrs » sont ceux qu'un lecteur validerait d'un coup d'œil ; leur précision dit si on peut lui faire cette promesse.
- La ligne « nulle part » est la plus importante : ce sont les événements que l'outil ne signale d'aucune façon, et que le lecteur ne verra que s'il relit toute la nuit.
- L'index à donner au médecin est l'index clinique estimé : il applique la définition des index SHHS (hypopnées comptées seulement avec désaturation). Détail et comparaison au simple compte des désaturations : `docs/RESULTATS_CLINIQUE.md`.
- L'index de bout en bout dépend des deux réseaux : une erreur sur le temps de sommeil se retrouve dans l'index.
