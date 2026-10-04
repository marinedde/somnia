# Résultats — stades à partir de plusieurs capteurs (horizon 1.3)

*Généré le 2026-10-04 par `python_scripts/multi_report.py`. Validation SHHS (40 personnes, 40,020 époques) ; entraînement sur 192 personnes. Moyenne ± écart-type sur les graines. Le test des stades n'est pas rouvert.*

## La question

Un technicien score le sommeil avec trois familles de signaux : l'EEG, les mouvements des yeux (EOG) et le tonus du menton (EMG). Le modèle de l'horizon 1.2 ne lisait que l'EEG. Que rapporte chaque capteur ajouté ? La cible fixée d'avance : un accord éveil / sommeil d'au moins 95%, parce que c'est lui qui plafonne le réseau d'événements respiratoires.

Toutes les variantes suivent exactement la même procédure (même architecture, mêmes réglages, mêmes personnes) ; seule l'entrée change. L'EEG est centré-réduit par époque ; les yeux et le menton sont réduits par nuit, pour garder leur niveau relatif (un tonus effondré doit rester petit).

## Ablation

| Capteurs | Graines | Exactitude | Kappa | Accord éveil / sommeil | Éveils reconnus | Sommeils reconnus | Kappa par nuit (médiane) |
|---|---|---|---|---|---|---|---|
| *Horizon 1.2, EEG seul (pour mémoire)* | 3 | 0.782 ± 0.009 | 0.708 ± 0.011 | 0.920 ± 0.007 | 0.845 ± 0.031 | 0.948 ± 0.011 | 0.710 ± 0.001 |
| EEG seul (témoin : refait l'horizon 1.2) | 3 | 0.782 ± 0.009 | 0.708 ± 0.011 | 0.920 ± 0.007 | 0.845 ± 0.031 | 0.948 ± 0.011 | 0.710 ± 0.001 |
| EEG + second EEG | 3 | 0.794 ± 0.003 | 0.722 ± 0.004 | 0.934 ± 0.002 | 0.873 ± 0.022 | 0.956 ± 0.007 | 0.729 ± 0.005 |
| EEG + menton (EMG) | 3 | 0.794 ± 0.002 | 0.722 ± 0.004 | 0.934 ± 0.003 | 0.891 ± 0.024 | 0.950 ± 0.009 | 0.723 ± 0.009 |
| EEG + yeux (EOG) | 3 | 0.792 ± 0.009 | 0.720 ± 0.010 | 0.932 ± 0.003 | 0.836 ± 0.010 | 0.968 ± 0.007 | 0.726 ± 0.020 |
| EEG + yeux + menton | 3 | 0.799 ± 0.003 | 0.731 ± 0.003 | 0.943 ± 0.002 | 0.866 ± 0.009 | 0.971 ± 0.003 | 0.734 ± 0.004 |
| Les cinq capteurs | 3 | 0.797 ± 0.004 | 0.726 ± 0.004 | 0.936 ± 0.003 | 0.853 ± 0.012 | 0.966 ± 0.006 | 0.722 ± 0.005 |

## Par stade (F1)

| Capteurs | Éveil | N1 | N2 | N3 | REM | F1 macro |
|---|---|---|---|---|---|---|
| EEG seul (témoin : refait l'horizon 1.2) | 0.851 ± 0.014 | 0.416 ± 0.013 | 0.791 ± 0.013 | 0.725 ± 0.005 | 0.838 ± 0.012 | 0.724 ± 0.009 |
| EEG + second EEG | 0.877 ± 0.005 | 0.419 ± 0.001 | 0.795 ± 0.010 | 0.731 ± 0.010 | 0.849 ± 0.005 | 0.734 ± 0.002 |
| EEG + menton (EMG) | 0.880 ± 0.006 | 0.412 ± 0.016 | 0.795 ± 0.010 | 0.734 ± 0.003 | 0.850 ± 0.010 | 0.734 ± 0.003 |
| EEG + yeux (EOG) | 0.869 ± 0.003 | 0.410 ± 0.011 | 0.805 ± 0.013 | 0.734 ± 0.003 | 0.836 ± 0.012 | 0.731 ± 0.007 |
| EEG + yeux + menton | 0.891 ± 0.003 | 0.414 ± 0.008 | 0.801 ± 0.002 | 0.737 ± 0.003 | 0.850 ± 0.006 | 0.739 ± 0.003 |
| Les cinq capteurs | 0.878 ± 0.005 | 0.418 ± 0.001 | 0.804 ± 0.009 | 0.736 ± 0.001 | 0.848 ± 0.004 | 0.737 ± 0.002 |

## Calibration et tri par la confiance

| Capteurs | ECE avant | ECE après température | Meilleure époque |
|---|---|---|---|
| EEG seul (témoin : refait l'horizon 1.2) | 0.048 ± 0.008 | 0.022 ± 0.004 | 5, 2, 12 |
| EEG + second EEG | 0.052 ± 0.005 | 0.021 ± 0.001 | 8, 9, 15 |
| EEG + menton (EMG) | 0.043 ± 0.006 | 0.021 ± 0.007 | 4, 9, 5 |
| EEG + yeux (EOG) | 0.050 ± 0.011 | 0.028 ± 0.006 | 15, 10, 1 |
| EEG + yeux + menton | 0.048 ± 0.004 | 0.018 ± 0.006 | 11, 7, 12 |
| Les cinq capteurs | 0.039 ± 0.005 | 0.021 ± 0.003 | 5, 7, 5 |

## Lecture

- **Le témoin est exact** : l'EEG seul, par le nouveau chemin de données, redonne les chiffres de l'horizon 1.2. Les écarts du tableau viennent donc des capteurs, pas d'un changement de préparation.
- **Chaque capteur ajouté aide un peu, les yeux et le menton ensemble aident le plus** : c'est la combinaison qu'utilise un technicien. Elle est aussi la plus stable d'une graine à l'autre.
- **La cible de 95% d'accord éveil / sommeil n'est pas atteinte.** Le gain porte sur l'éveil et le REM ; le N1 ne bouge pas, quel que soit le capteur.
- **Ajouter le second EEG par-dessus n'apporte rien de plus** : il redit ce que dit le premier.
- Réserves : la variante retenue a été choisie sur la validation, sur laquelle elle est aussi mesurée (six variantes comparées sur 40 personnes) ; le gain réel sera un peu plus faible. Les écarts entre variantes voisines sont de l'ordre du bruit entre graines. Seul un test neuf tranchera, et celui des stades a déjà été ouvert.
- Suite : le sommeil prédit par cette variante alimente le réseau d'événements v4 (`docs/RESULTATS_EVENEMENTS.md`).
