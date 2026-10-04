# Résultats — la sortie par nuit complète

*Généré le 2026-10-04 par `python_scripts/nuit_complete.py` sur les 40 nuits de validation SHHS. Agrégats seulement : le détail d'une nuit reste hors dépôt. Le test n'est pas touché.*

Pour chaque nuit, deux réseaux tournent : celui des stades (EEG) et celui des événements respiratoires (flux, ceintures, saturation). La sortie réunit l'hypnogramme, les événements proposés avec leur confiance, l'index calculé sur le sommeil **prédit**, et une file de relecture commune.

## 1. Ce qu'il reste à relire

| Par nuit, médiane (quartiles) | |
|---|---|
| Durée de l'enregistrement | 510 (480–534) min |
| Époques dont le stade est à relire | 33 (26–42) % |
| Événements respiratoires de référence | 276 (143–349) |
| Événements proposés | 298 (221–372) dont sûrs 136 (98–210), à relire 136 (104–171) |
| « Événements possibles » signalés (non proposés) | 122 (84–144) |
| **Signal à relire, stades et événements réunis** | **315 (272–352) min, soit 63 (55–70) % de la nuit** |

Le signal à relire est l'union des passages concernés, avec 30 s de contexte autour de chaque événement. C'est une durée de signal, pas un temps de lecture : seul un chronomètre avec un lecteur dira le temps gagné.

## 2. La confiance trie-t-elle les événements ?

| Niveau | Événements proposés | Part qui recouvre un événement du technicien |
|---|---|---|
| Sûrs (confiance ≥ 0,85) | 6,821 | **83%** |
| À relire (confiance < 0,85) | 5,675 | 44% |

Plus on exige de confiance, plus les propositions sont justes, et moins il y en a :

| Confiance minimale | Part des événements proposés | Précision | Événements par nuit | Événements justes / événements de référence |
|---|---|---|---|---|
| 0.70 | 98% | 66% | 306 | 79% |
| 0.80 | 73% | 75% | 228 | 67% |
| 0.85 | 55% | 82% | 171 | 55% |
| 0.90 | 35% | 90% | 108 | 38% |
| 0.95 | 13% | 96% | 40 | 15% |

## 3. Où tombent les événements du technicien

| | Part des événements de référence |
|---|---|
| Dans un événement proposé comme sûr | 57% |
| Dans un événement proposé à relire | 24% |
| Dans un « événement possible » signalé | 6% |
| Nulle part : à trouver par le lecteur | **13%** |

## 4. L'index, de bout en bout

Sommeil prédit par le réseau de stades, événements prédits par le réseau respiratoire, contre l'index calculé avec le sommeil et les événements du technicien.

| Contre | Spearman | Erreur absolue médiane | Biais |
|---|---|---|---|
| Index annoté (technicien) | 0.83 | 6.0 / h | +3.3 / h |
| Index clinique `ahi_a0h3a` | 0.84 | — | — |
| Index clinique `ahi_a0h4` | 0.70 | — | — |

Temps de sommeil : prédit 6.0 (5.2–7.0) h, technicien 6.0 (5.5–6.7) h.

## Lecture

- Les événements « sûrs » sont ceux qu'un lecteur validerait d'un coup d'œil ; leur précision dit si on peut lui faire cette promesse.
- La ligne « nulle part » est la plus importante : ce sont les événements que l'outil ne signale d'aucune façon, et que le lecteur ne verra que s'il relit toute la nuit.
- L'index de bout en bout dépend des deux réseaux : une erreur sur le temps de sommeil se retrouve dans l'index.
