# Résultats — ce qui compte cliniquement (horizon 1.4)

*Généré le 2026-10-04 par `python_scripts/resp_clinique.py`. Validation SHHS, 40 nuits, de bout en bout (stades : EEG + yeux + menton ; événements : v4 ; sommeil prédit). Agrégats seulement. Le test n'est pas touché.*

## 1. Élargir les zones « événement possible »

Une zone « possible » est un passage où le réseau hésite : P(événement) au-dessus d'un plancher mais sous le seuil de décision. Trois réglages : le plancher, la durée minimale, et les trous qu'on comble. Le coût se lit dans le signal à relire.

| Plancher de P | Durée minimale | Trous comblés | Manqués, tous | **Manqués, avec désaturation** | Zones possibles par nuit | Signal à relire |
|---|---|---|---|---|---|---|
| 0.4 **(retenu)** | 10 s | 0 s | 18.3% | 8.6% | 75 | 38 % |
| 0.4 | 10 s | 3 s | 12.1% | 5.6% | 124 | 45 % |
| 0.4 | 5 s | 0 s | 12.7% | 5.8% | 210 | 46 % |
| 0.4 | 5 s | 3 s | 9.3% | 4.6% | 236 | 51 % |
| 0.3 | 10 s | 0 s | 13.6% | 6.4% | 133 | 46 % |
| 0.3 | 10 s | 3 s | 8.7% | 4.5% | 178 | 51 % |
| 0.3 | 5 s | 0 s | 8.8% | 4.2% | 300 | 53 % |
| 0.3 | 5 s | 3 s | 6.8% | 3.8% | 314 | 57 % |
| 0.2 | 10 s | 0 s | 9.2% | 4.7% | 200 | 52 % |
| 0.2 | 10 s | 3 s | 6.0% | 3.8% | 232 | 57 % |
| 0.2 | 5 s | 0 s | 6.0% | 3.5% | 402 | 60 % |
| 0.2 | 5 s | 3 s | 5.0% | 3.4% | 376 | 62 % |

Règle écrite avant de regarder : parmi les réglages qui n'ajoutent pas plus de 5 points de signal à relire, celui qui laisse le moins d'événements avec désaturation signalés nulle part.

## 2. Où tombent les événements du technicien (réglage retenu)

| Événements du technicien pendant le sommeil | Nombre | Proposé « sûr » | Proposé « à relire » | Zone « possible » | **Nulle part** |
|---|---|---|---|---|---|
| Tous | 8,467 | 54% | 20% | 7% | **18%** |
| Avec désaturation ≥ 3 points | 3,559 | 75% | 13% | 4% | **9%** |
| Avec désaturation ≥ 4 points | 1,913 | 82% | 9% | 2% | **6%** |
| Sans désaturation (< 3 points) | 4,908 | 39% | 26% | 10% | **25%** |
| Apnées | 949 | 89% | 4% | 1% | **5%** |

La désaturation associée est mesurée par une règle simple (maximum des 30 s avant le début, minimum jusqu'à 45 s après la fin), pas par l'annotation du technicien.

## 3. L'index clinique

Les index SHHS comptent toutes les apnées, et les hypopnées seulement si elles sont suivies d'une désaturation (3 points pour `ahi_a0h3`, 4 pour `ahi_a0h4`). La sortie par nuit donne maintenant le même calcul : chaque événement proposé reçoit sa désaturation associée. Classes de sévérité : moins de 5, 5 à 15, 15 à 30, 30 et plus par heure.

**Contre `ahi_a0h3`** (40 personnes)

| Estimateur | Spearman | Erreur absolue médiane | Biais | Même classe de sévérité | À une classe près |
|---|---|---|---|---|---|
| Réseau : apnées + hypopnées avec désaturation ≥ 3 | 0.96 | 2.3 / h | +1.5 / h | 85% | 100% |
| Réseau : tous les événements proposés | 0.82 | 19.3 / h | +18.0 / h | 22% | 70% |
| Référence simple : désaturations ≥ 3 points / h | 0.91 | 2.0 / h | -0.3 / h | 75% | 98% |

**Contre `ahi_a0h4`** (40 personnes)

| Estimateur | Spearman | Erreur absolue médiane | Biais | Même classe de sévérité | À une classe près |
|---|---|---|---|---|---|
| Réseau : apnées + hypopnées avec désaturation ≥ 4 | 0.90 | 1.7 / h | +0.9 / h | 82% | 100% |
| Réseau : tous les événements proposés | 0.77 | 23.4 / h | +22.9 / h | 8% | 50% |
| Référence simple : désaturations ≥ 4 points / h | 0.91 | 1.9 / h | -2.7 / h | 72% | 100% |

## Lecture

- **Zones possibles : le réglage ne change pas.** Tous les réglages plus larges coûtent plus de 5 points de signal à relire. Le compromis est dans le tableau : combler les trous de 3 s ramènerait les manqués avec désaturation de 9% à 6%, pour 7 points de signal en plus. C'est une décision de produit, à prendre avec un lecteur, pas une décision de modèle.
- **Les événements qui comptent sont bien mieux retrouvés que le total ne le laisse croire** : le tableau 2 est à lire ligne « avec désaturation », pas ligne « tous ».
- **L'index clinique estimé est le bon index à rendre** : en ne comptant que les hypopnées avec désaturation, l'index du réseau suit `ahi_a0h3` avec un Spearman de 0.96 et la même classe de sévérité dans 85% des cas. Compter tous les événements proposés surestimait l'index de près de 20 par heure : c'était une erreur de définition, pas de détection.
- **Il ne bat pas de façon établie le simple compte des désaturations** (0.91, 75%) : sur 40 personnes, l'intervalle de la différence contient zéro (`docs/INTERVALLES.md`). Ce que le réseau apporte en plus de l'index, c'est la position et le type de chaque événement.
- Réserves : 40 personnes ; la règle de désaturation (30 s avant, 45 s après) n'a pas été réglée, mais elle n'a pas non plus été validée contre l'annotation du technicien ; les intervalles de confiance sont dans `docs/INTERVALLES.md`.
