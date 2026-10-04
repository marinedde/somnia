# Résultats — la sortie par nuit complète

*Généré le 2026-10-04 par `python_scripts/nuit_complete.py` sur les 40 nuits de validation SHHS. Agrégats seulement : le détail d'une nuit reste hors dépôt. Le test n'est pas touché.*

Pour chaque nuit, deux réseaux tournent : celui des stades (EEG) et celui des événements respiratoires (flux, ceintures, saturation, et le sommeil prédit par le premier). La sortie réunit l'hypnogramme, les événements proposés pendant le sommeil prédit avec leur confiance, l'index, et une file de relecture commune. La référence est l'ensemble des événements que le technicien a marqués pendant le sommeil : ceux qui comptent dans l'index.

## 1. Ce qu'il reste à relire

| Par nuit, médiane (quartiles) | |
|---|---|
| Durée de l'enregistrement | 510 (480–534) min |
| Époques dont le stade est à relire | 33 (26–42) % |
| Événements respiratoires de référence | 212 (119–302) |
| Événements proposés | 190 (130–293) dont sûrs 100 (56–158), à relire 75 (57–106) |
| « Événements possibles » signalés (non proposés) | 78 (52–104) |
| **Signal à relire, stades et événements réunis** | **254 (227–282) min, soit 50 (44–58) % de la nuit** |

Le signal à relire est l'union des passages concernés, avec 30 s de contexte autour de chaque événement. C'est une durée de signal, pas un temps de lecture : seul un chronomètre avec un lecteur dira le temps gagné.

## 2. La confiance trie-t-elle les événements ?

| Niveau | Événements proposés | Part qui recouvre un événement du technicien |
|---|---|---|
| Sûrs (confiance ≥ 0,85) | 4,923 | **91%** |
| À relire (confiance < 0,85) | 3,248 | 57% |

Plus on exige de confiance, plus les propositions sont justes, et moins il y en a :

| Confiance minimale | Part des événements proposés | Précision | Événements par nuit | Événements justes / événements de référence |
|---|---|---|---|---|
| 0.70 | 99% | 78% | 203 | 75% |
| 0.80 | 78% | 85% | 160 | 64% |
| 0.85 | 60% | 91% | 124 | 53% |
| 0.90 | 40% | 96% | 81 | 37% |
| 0.95 | 15% | 98% | 31 | 15% |

## 3. Où tombent les événements du technicien

| | Part des événements de référence |
|---|---|
| Dans un événement proposé comme sûr | 52% |
| Dans un événement proposé à relire | 21% |
| Dans un « événement possible » signalé | 7% |
| Nulle part : à trouver par le lecteur | **21%** |

## 4. L'index, de bout en bout

Sommeil prédit par le réseau de stades, événements prédits par le réseau respiratoire, contre l'index calculé avec le sommeil et les événements du technicien.

| Contre | Spearman | Erreur absolue médiane | Biais |
|---|---|---|---|
| Index annoté (technicien) | 0.82 | 5.4 / h | -0.4 / h |
| Index clinique `ahi_a0h3a` | 0.86 | — | — |
| Index clinique `ahi_a0h4` | 0.75 | — | — |

Temps de sommeil : prédit 6.0 (5.2–7.0) h, technicien 6.0 (5.5–6.7) h.

## Lecture

- Les événements « sûrs » sont ceux qu'un lecteur validerait d'un coup d'œil ; leur précision dit si on peut lui faire cette promesse.
- La ligne « nulle part » est la plus importante : ce sont les événements que l'outil ne signale d'aucune façon, et que le lecteur ne verra que s'il relit toute la nuit.
- L'index de bout en bout dépend des deux réseaux : une erreur sur le temps de sommeil se retrouve dans l'index.
