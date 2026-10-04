# Résultats — qualité du signal et refus (horizon 1.5)

*Généré le 2026-10-04 par `python_scripts/qualite_report.py`. Règles et seuils : `somnia/qualite.py`, écrits avant de lancer ce script. Agrégats seulement. Le test n'est pas touché.*

## Les règles

Par époque de 30 s et par capteur, relativement à la nuit elle-même : **plat** (écart-type sous 5 % de la médiane de la nuit), **écrêté** (plus de 5 % des points à l'extrême de la nuit), **mouvement** (écart-type au-delà de 6 fois la médiane). Saturation : valeurs hors de 50–100 %.

Un passage est **inexploitable** quand : un capteur de la tête est plat ou écrêté ; le flux est plat ; les deux ceintures sont plates ou écrêtées ; la saturation est invalide. Au-delà de 25% de la nuit, l'outil refuse de rendre les indices de sommeil (EEG) ou l'index (respiration) ; si ce sont les yeux ou le menton, il passe au modèle EEG seul.

**Ce que la première version a donné, et pourquoi elle a été révisée.** Écrite avant de regarder, elle comptait aussi comme inexploitables les mouvements (tête) et tout capteur respiratoire écrêté. Mesurée sur la validation :

- les 850 époques d'EEG signalées étaient presque toutes de l'éveil avec mouvement, où le réseau est juste à 89% (contre 79% ailleurs) : la règle envoyait en relecture des époques faciles ;
- 23 nuits sur 231 étaient refusées pour l'index, alors que les propositions y étaient aussi justes qu'ailleurs (77% contre 79%) ;
- détail par cause : flux écrêté, propositions justes à 75 % (pas un problème) ; deux ceintures écrêtées, 51 % ; saturation invalide, 39 %.

La définition a donc été resserrée sur ce qui dégrade vraiment. Les seuils n'ont pas été touchés. **Ce choix a été fait sur les 40 personnes de validation** : c'est une hypothèse à confirmer sur d'autres nuits, pas un résultat.

## A. Ce que les règles signalent (231 nuits d'entraînement et de validation)

| Capteur | Époques inexploitables, moyenne | Médiane | 9e décile | Maximum | Nuits au-delà du quart | dont plat / écrêté / artefact (moyennes) |
|---|---|---|---|---|---|---|
| EEG | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 0.0% / 0.0% / 1.6% |
| EEG2 | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 0.0% / 0.0% / 1.6% |
| EOG-G | 0.1% | 0.0% | 0.0% | 25.7% | 1 | 0.0% / 0.1% / 3.5% |
| EOG-D | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 0.0% / 0.0% / 3.3% |
| EMG | 0.3% | 0.0% | 0.0% | 23.8% | 0 | 0.3% / 0.0% / 3.4% |
| flux | 0.0% | 0.0% | 0.0% | 6.3% | 0 | — |
| thorax | 9.3% | 7.4% | 18.4% | 50.1% | 11 | — |
| abdomen | 13.7% | 8.7% | 30.2% | 100.0% | 31 | — |
| sao2 | 3.2% | 0.5% | 9.7% | 29.7% | 3 | — |
| respiration | 8.3% | 6.6% | 18.3% | 36.8% | 8 | — |

Nuits refusées : 0 pour les stades, 8 pour l'index, sur 231. Rappel : la cohorte a déjà écarté, à la préparation, les nuits sans canal ou avec plus de 30 % de saturation invalide.

## B. Les règles trouvent-elles les erreurs ? (40 nuits de validation)

**Stades** (EEG + yeux + menton)

| Époques | Nombre | Part | Exactitude | Confiance médiane du réseau | Déjà sous le seuil de relecture |
|---|---|---|---|---|---|
| EEG exploitable | 40,020 | 100.0% | 79.6% | 0.85 | 17.8% |
| EEG inexploitable | 0 | 0.0% | — | — | — |
| Yeux inexploitables (EEG exploitable) | 0 | 0.0% | — | — | — |
| Menton inexploitable (EEG exploitable) | 125 | 0.3% | 62.4% | 0.66 | 36.8% |
| Tout exploitable | 39,895 | 99.7% | 79.6% | 0.85 | 17.7% |
| EEG : mouvement signalé (information, pas un refus) | 850 | 2.1% | 88.7% | 0.99 | 4.8% |

**Événements respiratoires** (v4)

| Zone | Événements proposés | Part qui recouvre un événement du technicien | Événements du technicien | Part retrouvée |
|---|---|---|---|---|
| Respiration exploitable | 8,107 | 79.5% | 8,374 | 74.4% |
| Respiration inexploitable | 127 | 64.6% | 93 | 58.1% |

**Effet sur la sortie par nuit**

| | Sans la qualité | Avec la qualité |
|---|---|---|
| Signal à relire, médiane | 38 % | 38 % |
| Index clinique contre `ahi_a0h3`, Spearman | 0.958 | 0.961 |
| Index clinique contre `ahi_a0h3`, erreur absolue médiane | 2.3 / h | 2.0 / h |

## C. Un capteur débranché toute la nuit (simulé)

Le canal est remplacé par des zéros sur toute la nuit, et le réseau multi-capteurs est appliqué tel quel.

| Situation | Kappa | Accord éveil / sommeil |
|---|---|---|
| Rien de débranché | 0.726 | 94.1% |
| Menton débranché | 0.702 | 92.6% |
| Yeux débranchés | 0.663 | 90.5% |
| Yeux et menton débranchés | 0.579 | 83.9% |
| Modèle EEG seul | 0.724 | 93.0% |

## Lecture

- **Dans cette cohorte, un capteur vraiment perdu est rare** : elle a déjà été triée à la préparation, et les enregistrements SHHS ont été contrôlés à l'époque. Les règles serviront surtout sur des nuits venues d'ailleurs.
- **Un mouvement n'est pas une panne** : 99.8% des époques d'EEG « en mouvement » sont de l'éveil, et le réseau y est juste à 88.7%. C'est une information utile au lecteur, pas un motif de refus.
- **Là où la respiration est inexploitable, les propositions sont moins justes** (64.6% contre 79.5%) : ne rien y proposer et le dire est le bon comportement. Le coût est du signal à relire en plus.
- **Un capteur débranché fait plus de mal que son absence** : sans les yeux et le menton, le réseau multi-capteurs tombe à un kappa de 0.58, alors que le modèle EEG seul fait 0.72. D'où la règle de repli : si les yeux ou le menton sont inexploitables sur plus d'un quart de la nuit, les stades sont calculés avec le modèle EEG seul.
- **Ce que le module ne voit pas** : un capteur mal posé mais qui bouge (EEG noyé dans l'ECG ou le secteur), une saturation plausible mais fausse, une inversion de canaux. Il faudrait des règles spectrales, ou un détecteur appris.
- **Limites** : le refus de l'index compte le temps inexploitable sur toute la nuit, éveil compris, alors que seul le sommeil compte ; et la comparaison des zones repose sur peu d'événements (une centaine en zone inexploitable).
- **Pas encore fait** : un refus par passage appris sur la confiance du réseau lui-même, et un essai sur des nuits réellement dégradées.
