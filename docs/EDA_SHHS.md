# EDA — cohorte SHHS préparée

*Généré le 2026-10-03 par `python_scripts/shhs_eda.py` sur 272 nuits, 272,598 époques. Chiffres agrégés seulement ; le détail par nuit (avec identifiants) reste hors dépôt.*

## Stades et étiquettes

| Stade | Époques | Part |
|---|---|---|
| Wake | 76,792 | 28.2% |
| N1 | 9,239 | 3.4% |
| N2 | 113,773 | 41.7% |
| N3 | 36,120 | 13.3% |
| REM | 36,663 | 13.4% |
| Inconnu | 11 | 0.0% |

Époques de sommeil : 195,795 (71.8%). Part d'époques « apnée » pendant le sommeil : **30.1%** ; pendant l'éveil (à exclure de la tâche) : 18.6%.

### Par ensemble du découpage

| Ensemble | Nuits | Époques | Part N1 | Part N3 | Part REM | Part apnée (sommeil) |
|---|---|---|---|---|---|---|
| train | 192 | 192,298 | 3.3% | 13.5% | 13.2% | 29.8% |
| val | 40 | 40,020 | 3.8% | 12.9% | 14.9% | 30.9% |
| test | 40 | 40,280 | 3.2% | 12.5% | 13.4% | 31.0% |

## Amplitudes des signaux (par époque)

| Signal | p1 | p5 | médiane | p95 | p99 | max |
|---|---|---|---|---|---|---|
| EEG, écart-type (µV) | 5.5 | 7.4 | 14.9 | 63.5 | 119.2 | 124 |
| EEG, max absolu (µV) | 23 | 33 | 73 | 145 | 170 | 210 |
| ECG, écart-type (mV) | 0.021 | 0.030 | 0.109 | 0.306 | 1.150 | 1.24 |
| ECG, max absolu (mV) | 0.12 | 0.17 | 0.70 | 1.37 | 1.60 | 2.15 |

Écart-type médian de l'EEG par stade (attendu : N3 > N2 > N1 ≈ REM, Wake variable) :

- Wake : 17.5 µV
- N1 : 10.3 µV
- N2 : 14.2 µV
- N3 : 22.3 µV
- REM : 9.7 µV

## Valeurs aberrantes

Seuils fixés avant de regarder : EEG plat < 1.0 µV, EEG extrême > 200.0 µV, EEG saturé (> 5 % des échantillons à ±120 µV, plage physique ±125 µV), ECG plat, ECG saturé (> 5 % à ±1.24 mV).

| Critère | Époques | Part |
|---|---|---|
| EEG plat | 0 | 0.00% |
| EEG extrême | 0 | 0.00% |
| EEG saturé | 18,289 | 6.71% |
| ECG plat | 0 | 0.00% |
| ECG saturé | 7,116 | 2.61% |
| **Au moins un critère** | 20,375 | 7.47% |

Saturation EEG par stade (une saturation surtout en éveil = mouvements et yeux, pas des ondes lentes) :

- Wake : 16.6% des époques
- N1 : 2.7% des époques
- N2 : 3.1% des époques
- N3 : 2.1% des époques
- REM : 2.7% des époques

Nuits avec plus de 20% d'époques aberrantes : **19** (identifiants dans `eda_nuits.csv`, hors dépôt). Par ensemble : train 9, val 5, test 5.

## Ce qu'on en fait

- La saturation EEG touche surtout l'éveil : ce sont des artefacts de mouvement et d'yeux, pas des ondes lentes. Une poignée de nuits saturées à plus de 50 % (gain mal réglé) concentre une bonne part des époques aberrantes.
- Les époques aberrantes ne sont **pas** retirées des données : un modèle en production les verra. Elles sont marquées, et un futur module « qualité du signal » devra les refuser plutôt que les classer.
- Pour les références de l'étape 3, on entraîne sur tout et on rapporte aussi le score sans les époques aberrantes, pour mesurer leur poids.
- Figures : `data/figures/shhs_eda_amplitudes.png`, `data/figures/shhs_eda_stades.png`.
