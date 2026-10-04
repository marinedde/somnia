# Résultats — type d'apnée : obstructive ou centrale

*Généré le 2026-10-04 par `python_scripts/type_apnee.py`. Appris sur les apnées du technicien des nuits d'entraînement (5,896 obstructives, 839 centrales), mesuré sur la validation (1,341 et 100). Agrégats seulement.*

## Le principe

Apnée obstructive : la gorge est fermée, le patient fait toujours l'effort de respirer, les ceintures bougent, souvent à contre-temps. Apnée centrale : la commande s'arrête, les ceintures sont plates. Deux méthodes : une **référence simple** (amplitude du thorax, de l'abdomen et du flux pendant l'événement, rapportée à la minute qui précède, opposition thorax–abdomen, régression logistique) et un **petit réseau** qui lit la forme du flux et des ceintures sur 90 s autour du début de l'apnée. Les apnées mixtes sont trop rares dans SHHS pour être apprises (19 dans l'entraînement).

## 1. Sur les apnées du technicien (validation)

| Mesure | Référence simple | Réseau (3 graines) |
|---|---|---|
| Aire sous la courbe ROC | 0.73 | **0.83 ± 0.03** |
| Apnées centrales reconnues (sensibilité) | 0.75 | 0.51 ± 0.05 |
| Apnées obstructives reconnues (spécificité) | 0.60 | 0.91 ± 0.04 |
| Parmi les apnées dites centrales, part qui l'est vraiment | 0.12 | 0.32 ± 0.08 |

Référence simple, trait par trait (aire sous la courbe) : log_thorax 0.79, log_abdomen 0.65, log_flux 0.65, opposition 0.50.

## 2. De bout en bout : index d'apnées centrales par personne

Apnées proposées par le réseau d'événements pendant le sommeil prédit, puis typées, contre les apnées centrales du technicien (40 personnes).

| Mesure | Référence simple | Réseau (graine 42) |
|---|---|---|
| Spearman | 0.52 | 0.52 |
| Erreur absolue médiane | 0.35 / h | 0.14 / h |
| Biais | +1.42 / h | +0.37 / h |
| Apnées centrales comptées : technicien / estimées | 85 / 424 | 85 / 168 |
| Personnes à 5 par heure ou plus : technicien / estimées / les deux | 0 / 5 / 0 | 0 / 2 / 0 |

## Lecture

- **Le réseau fait mieux que la règle, mais pas assez pour typer chaque apnée** : quand il dit « centrale », il a raison environ une fois sur trois. La raison est d'abord la rareté : une apnée sur quinze est centrale, donc même 9 obstructives sur 10 bien reconnues laissent beaucoup de fausses centrales.
- **De bout en bout, l'index d'apnées centrales n'est pas fiable** : il surestime, et désigne à tort des personnes au-dessus de 5 par heure alors qu'aucune ne l'est dans la validation.
- **Décision : le type n'entre pas dans la sortie par nuit.** Les apnées restent « apnée », à typer par le lecteur. Le réseau de typage est gardé comme point de départ (`somnia/deep/type_net.py`).
- **Ce qui manque** : des apnées centrales en nombre (une cohorte d'insuffisants cardiaques, par exemple), et de meilleures ceintures : celles de SHHS sont codées sur 8 bits et souvent écrêtées (`docs/RESULTATS_QUALITE.md`). Avec 85 apnées centrales dans la validation, la mesure elle-même est très bruitée.
