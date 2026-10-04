# Protocole pour les nouvelles nuits SHHS (cohorte v2)

*Écrit le 4 octobre 2026, avant qu'une seule des nouvelles nuits soit préparée ou regardée.
Statut : proposition, à valider par l'autrice avant exécution.*

## Pourquoi

Avec 40 personnes de validation, un kappa ou un F1 n'est connu qu'à ± 0,04 (`docs/INTERVALLES.md`).
Tous les réglages de l'horizon 1 ont été choisis sur ces 40 personnes. Le test des stades (40 autres
personnes) a déjà été ouvert une fois. Il faut donc des personnes neuves, pour deux usages distincts.

## Les nuits

Enregistrements SHHS visite 1, identifiants 200300 à 200599 (300 nuits téléchargées, hors dépôt).
Aucun modèle, aucun seuil, aucune règle n'a vu ces personnes. Mêmes règles d'exclusion que la
cohorte v1, écrites dans `somnia/shhs_prepare.py` et `python_scripts/resp_prepare.py`, appliquées telles quelles.

## Le découpage, fixé d'avance

Par personne, tirage avec la graine 42, stratifié sur la sévérité estimée depuis les annotations
(moins de 5, 5 à 15, 15 à 30, 30 et plus), avec le même script que la v1 (`python_scripts/shhs_make_split.py`) :

| Ensemble | Part des nouvelles personnes | Usage |
|---|---|---|
| Validation 2 | 50 % | S'ajoute aux 40 personnes de validation : mesures plus fines, réglages futurs |
| Test 2 | 50 % | Jamais regardé avant le gel ; ouvert une seule fois |

L'entraînement n'est pas agrandi : la courbe d'apprentissage est plate (`docs/RESULTATS_EVENEMENTS.md`).
Si ce choix change plus tard, les personnes viendront d'un nouveau téléchargement, pas de ces deux ensembles.

## Première utilisation de la validation 2 : confirmer, sans rien régler

Avant tout nouveau réglage, rejouer tels quels sur la validation 2 les modèles et les règles actuels, et comparer
aux chiffres obtenus sur les 40 personnes d'origine. Sont à confirmer en priorité, parce qu'ils ont été choisis
ou révisés sur la validation :

1. le gain des yeux et du menton sur les stades (kappa +0,023) ;
2. la définition révisée d'un passage inexploitable (`somnia/qualite.py`) ;
3. le seuil de décision des événements (0,7), la confiance « sûre » (0,85), le réglage des zones « possibles » ;
4. l'index clinique estimé contre `ahi_a0h3` (Spearman 0,96) et son avantage, non établi, sur le compte des désaturations ;
5. les micro-éveils (F1 0,69) et l'index par position.

Un chiffre qui baisse nettement sur la validation 2 est un chiffre qui était optimiste. Il sera publié tel quel.

## Ouverture du test : une fois, sur une liste gelée

Le test 2 (stades, événements, micro-éveils, index) et le test v1 (événements seulement : il n'a jamais été ouvert
pour cette tâche) sont ouverts ensemble, une seule fois, quand la liste suivante est gelée dans un commit :

- les fichiers de poids et leurs graines ;
- les seuils et les règles (`somnia/nuit.py`, `somnia/qualite.py`, `somnia/resp.py`) ;
- la liste des mesures rapportées, avec leurs intervalles par bootstrap sur les personnes.

Le script d'ouverture refuse de tourner une seconde fois, comme `python_scripts/evaluate_test.py`.
Le test v1 des stades n'est pas rouvert.
