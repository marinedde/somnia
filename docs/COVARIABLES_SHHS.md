# Covariables SHHS — cohorte Somnia

*Généré le 2026-10-03 par `python_scripts/shhs_covariates.py` à partir de `shhs1-dataset-0.21.0.csv` (NSRR). Agrégats seulement ; aucune cellule de moins de 5 personnes.*

## 1. La cohorte, et chaque ensemble

| | Toutes | Entraînement | Validation | Test |
|---|---|---|---|---|
| Personnes | 272 | 192 | 40 | 40 |
| Âge, médiane (IQR) | 58.5 (50.0–68.0) | 59.0 (49.8–68.0) | 54.5 (47.8–66.0) | 62.5 (53.8–69.0) |
| Femmes | 45% | 45% | 38% | 52% |
| IMC, médiane (IQR) | 26.5 (24.1–29.5) | 26.4 (23.8–29.2) | 26.9 (24.1–30.8) | 26.9 (24.3–30.7) |
| Efficacité de sommeil %, médiane (IQR) | 85.1 (77.6–90.6) | 85.0 (77.0–90.1) | 85.0 (78.5–91.5) | 85.1 (81.1–90.6) |
| Epworth, médiane (IQR) | 7.0 (4.0–11.0) | 7.0 (5.0–10.0) | 7.0 (4.0–11.0) | 8.0 (4.0–12.0) |
| IAH clinique ≥ 3 % ou éveil (ahi_a0h3a), médiane (IQR) | 13.8 (7.3–24.7) | 13.2 (6.4–25.1) | 14.2 (10.2–22.1) | 14.2 (7.4–24.4) |
| IAH clinique ≥ 4 % (ahi_a0h4), médiane (IQR) | 5.4 (1.7–14.1) | 5.0 (1.5–14.1) | 7.2 (3.4–10.4) | 6.4 (1.8–15.3) |
| Mon index annoté (événements / h), médiane (IQR) | 39.6 (23.6–57.2) | 39.5 (23.8–57.5) | 43.9 (23.3–59.2) | 33.8 (24.1–51.1) |
| Sévérité (ahi_a0h4) : < 5 (normal) | 47% | 49% | 45% | 40% |
| Sévérité (ahi_a0h4) : 5-15 (légère) | 28% | 27% | 32% | 32% |
| Sévérité (ahi_a0h4) : 15-30 (modérée) | 16% | 15% | 18% | 18% |
| Sévérité (ahi_a0h4) : ≥ 30 (sévère) | 8% | 9% | 5% | 10% |
| Sévérité (ahi_a0h3a) : < 5 (normal) | 17% | 19% | 8% | 15% |
| Sévérité (ahi_a0h3a) : 5-15 (légère) | 39% | 36% | 48% | 40% |
| Sévérité (ahi_a0h3a) : 15-30 (modérée) | 26% | 26% | 25% | 28% |
| Sévérité (ahi_a0h3a) : ≥ 30 (sévère) | 18% | 18% | 20% | 18% |

Âge minimal : 40 ans (SHHS recrute à partir de 40 ans). Qualité globale de la PSG (overall_shhs1, 1 à 7) : médiane 6.

## 2. Mon index annoté contre l'index clinique

Mon index compte tous les événements annotés (apnées + toutes les hypopnées) par heure de sommeil. L'index clinique SHHS ne compte une hypopnée qu'avec une désaturation (≥ 3 % ou micro-éveil pour `ahi_a0h3a`, ≥ 4 % pour `ahi_a0h4`).

| Comparaison | Corrélation (Spearman) | Rapport médian annoté / clinique | Accord de classe de sévérité |
|---|---|---|---|
| annoté vs ahi_a0h3a | 0.82 | ×2.7 | 24% |
| annoté vs ahi_a0h4 | 0.73 | ×6.6 | 10% |

Lecture : l'ordre des personnes est bien conservé (corrélation), mais le niveau ne l'est pas : mon index surestime la sévérité clinique. La stratification du découpage (faite sur l'index annoté) reste valide pour répartir ; les classes de sévérité à publier sont celles de l'index clinique.

## 3. Références de l'étape 3 par sous-groupe (validation, 40 personnes)

Cellules de moins de 5 personnes masquées. Avec 40 personnes, ces chiffres indiquent une direction, pas une certitude : l'intervalle d'une AUC sur 5 personnes est énorme.

### Stades (EEG)

| Sous-groupe | Personnes | Lignes | accuracy | kappa | f1_macro |
|---|---|---|---|---|---|
| Âge : 53–66 ans | 12 | 12,130 | 0.651 | 0.526 | 0.567 |
| Âge : < 53 ans | 17 | 17,100 | 0.736 | 0.646 | 0.646 |
| Âge : ≥ 66 ans | 11 | 10,790 | 0.700 | 0.590 | 0.613 |
| Sexe : femme | 15 | 15,175 | 0.743 | 0.653 | 0.659 |
| Sexe : homme | 25 | 24,845 | 0.675 | 0.562 | 0.595 |
| IMC : 25–30 | 16 | 15,740 | 0.662 | 0.543 | 0.589 |
| IMC : < 25 | 13 | 13,231 | 0.749 | 0.658 | 0.653 |
| IMC : ≥ 30 | 11 | 11,049 | 0.698 | 0.597 | 0.621 |
| Sévérité clinique (ahi_a0h4) : 15-30 (modérée) | 7 | 7,077 | 0.734 | 0.627 | 0.624 |
| Sévérité clinique (ahi_a0h4) : 5-15 (légère) | 13 | 12,921 | 0.667 | 0.557 | 0.601 |
| Sévérité clinique (ahi_a0h4) : < 5 (normal) | 18 | 18,161 | 0.718 | 0.620 | 0.627 |
| Sévérité clinique (ahi_a0h4) : ≥ 30 (sévère) | 2 | — | masqué | masqué | masqué |

### Apnée (ECG)

| Sous-groupe | Personnes | Lignes | auc_roc | auc_pr | f1_apnee |
|---|---|---|---|---|---|
| Âge : 53–66 ans | 12 | 4,067 | 0.626 | 0.486 | 0.598 |
| Âge : < 53 ans | 17 | 6,434 | 0.657 | 0.488 | 0.544 |
| Âge : ≥ 66 ans | 11 | 3,543 | 0.653 | 0.728 | 0.660 |
| Sexe : femme | 15 | 5,404 | 0.631 | 0.485 | 0.504 |
| Sexe : homme | 25 | 8,640 | 0.653 | 0.596 | 0.636 |
| IMC : 25–30 | 16 | 5,429 | 0.653 | 0.551 | 0.593 |
| IMC : < 25 | 13 | 4,711 | 0.661 | 0.581 | 0.609 |
| IMC : ≥ 30 | 11 | 3,904 | 0.636 | 0.566 | 0.579 |
| Sévérité clinique (ahi_a0h4) : 15-30 (modérée) | 7 | 2,129 | 0.668 | 0.796 | 0.705 |
| Sévérité clinique (ahi_a0h4) : 5-15 (légère) | 13 | 4,553 | 0.636 | 0.546 | 0.564 |
| Sévérité clinique (ahi_a0h4) : < 5 (normal) | 18 | 6,762 | 0.642 | 0.424 | 0.538 |
| Sévérité clinique (ahi_a0h4) : ≥ 30 (sévère) | 2 | — | masqué | masqué | masqué |

Par personne (validation) : part de minutes prédites positives contre les index de référence :

- corrélation de Spearman avec la part annotée : 0.21
- avec l'IAH clinique ahi_a0h3a : 0.16
- avec l'IAH clinique ahi_a0h4 : 0.17

C'est la mesure clinique : un bon modèle d'époques qui ordonne mal les personnes ne sert à rien à un médecin.
