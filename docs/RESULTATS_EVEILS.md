# Résultats — détection des micro-éveils (horizon 2.1)

*Généré le 2026-10-04 par `python_scripts/eveils_report.py`. Entraînement : 192 nuits ; validation : 40 nuits, 4,400 micro-éveils du technicien pendant le sommeil. 3 graines. Le test n'est pas touché.*

## Ce qui est mesuré

Le réseau lit l'EEG, le second EEG et le menton à 100 Hz, par fenêtres de 2 minutes, et rend pour chaque seconde la probabilité d'un micro-éveil. Les secondes au-dessus de 0,5 sont regroupées en événements d'au moins 3 s. Un événement proposé est juste s'il recouvre un micro-éveil du technicien (appariement un à un).

| Mesure | De bout en bout (sommeil prédit) | Avec le sommeil du technicien |
|---|---|---|
| Précision | 0.61 ± 0.01 | 0.70 ± 0.01 |
| Rappel | 0.79 ± 0.01 | 0.80 ± 0.01 |
| F1 par événement | 0.69 ± 0.00 | 0.75 ± 0.00 |
| Index par personne : Spearman | 0.74 ± 0.02 | 0.71 ± 0.03 |
| Index : erreur absolue médiane (/h) | 4.7 ± 0.4 | 3.8 ± 0.1 |
| Index : biais (/h) | 3.8 ± 0.6 | 2.3 ± 0.5 |

Index médian du technicien : 17 micro-éveils par heure de sommeil.

## Le seuil : précision contre rappel (de bout en bout)

| Seuil sur P(micro-éveil) | Précision | Rappel | F1 |
|---|---|---|---|
| 0.3 | 0.51 | 0.86 | 0.64 |
| 0.4 | 0.56 | 0.82 | 0.67 |
| 0.5 (retenu, fixé d’avance) | 0.61 | 0.79 | 0.69 |
| 0.6 | 0.66 | 0.74 | 0.70 |
| 0.7 | 0.71 | 0.67 | 0.69 |
| 0.8 | 0.76 | 0.58 | 0.66 |

## Lecture

- **Quatre micro-éveils sur cinq sont retrouvés, et six propositions sur dix sont justes.** C'est une première version, sans réglage.
- **L'écart entre les deux colonnes vient du réseau de stades** : avec le sommeil du technicien, la précision monte de près de dix points. Les fausses propositions sont en grande partie des éveils francs que le réseau de stades a pris pour du sommeil.
- **L'index par personne est surestimé de quelques unités par heure** et ordonne les personnes moyennement. Il ne remplace pas un comptage.
- **Repère** : les micro-éveils sont l'événement sur lequel les scoreurs humains s'accordent le moins. Une partie des « fausses » propositions sont probablement des micro-éveils qu'un autre technicien aurait marqués. Non vérifiable ici, faute de double scoring.
- **Dans la sortie par nuit**, les micro-éveils sont proposés avec leur confiance et leur index, mais ne sont pas ajoutés à la file de relecture : à ce niveau de précision, ils alourdiraient la relecture sans la guider.
- **Pistes** : donner au réseau le sommeil prédit et les événements respiratoires (un micro-éveil suit souvent une apnée), lier chaque hypopnée à son micro-éveil pour l'index `ahi_a0h3a`, et allonger le contexte.
