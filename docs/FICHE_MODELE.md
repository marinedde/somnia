# Fiche modèle — Somnia

*Version du 3 octobre 2026. À relire à chaque réentraînement.*

## Ce que c'est

Deux familles de modèles, un même cadre :

| | Stades de sommeil | Apnée / hypopnée |
|---|---|---|
| Entrée | 30 s d'EEG, une dérivation, 100 Hz | 60 s d'ECG, une dérivation, 100 Hz (deux époques de sommeil) |
| Sortie | 5 stades (W, N1, N2, N3, REM) avec probabilités | apnée ou non, avec probabilité |
| Modèles | Random Forest sur 16 caractéristiques (API) ; réseau convolutif 1D (recherche) | Random Forest sur 16 caractéristiques (API) ; réseau convolutif 1D (recherche) |
| Entraînement | SHHS visite 1, 192 personnes ; PhysioNet Sleep-EDF, 16 personnes | SHHS visite 1, 192 personnes ; PhysioNet Apnea-ECG, 30 groupes |

## Usage prévu

Aide à la **relecture** d'un enregistrement du sommeil par un professionnel formé : proposer un
hypnogramme, des indices, et une **file de relecture** des époques les moins sûres. L'humain
relit et décide ; le modèle trie. Recherche et formation.

## Usages exclus

- Diagnostic sans relecture humaine. Dépistage sans avis médical. Décision thérapeutique.
- Enfants et adolescents (aucune donnée de moins de 40 ans à l'entraînement SHHS).
- Enregistrements avec stimulateur cardiaque ou en fibrillation auriculaire : un modèle fondé
  sur la variabilité du rythme a toutes les raisons d'échouer ; **non mesuré**.
- Tout usage sur de vrais patients en dehors d'un protocole de recherche : ce logiciel n'est
  pas un dispositif médical et n'a fait l'objet d'aucune évaluation réglementaire.

## Sur qui il a été mesuré

| Cohorte | Personnes | Âge | Femmes | IMC | Particularités |
|---|---|---|---|---|---|
| SHHS visite 1 (entraînement 192, validation 40, test 40) | 272 | médiane 58,5 ans, minimum 40 | 45 % | médiane 26,5 | population générale américaine, enregistrement à domicile, 1995-1998 ; IAH clinique médian 5 (critère 4 %) |
| Sleep-EDF (validation externe) | 16 | 25 à 101 ans d'après la documentation | non vérifié | — | volontaires sains, cassette à domicile, 1987-1991 |
| Apnea-ECG (validation externe) | 30 groupes | 27 à 63 ans d'après la documentation | non vérifié | — | patients et témoins sélectionnés |

Populations **non couvertes** : moins de 40 ans, grossesse, porteurs de stimulateur,
fibrillation auriculaire, maladies neurologiques, enregistrements de laboratoire récents
(pression nasale au lieu de thermistance), autres pays et autres appareils.

## Performance (validation SHHS, 40 personnes, test fermé jusqu'à l'ouverture finale)

| Tâche | Modèle | Métrique | Valeur |
|---|---|---|---|
| Stades | Random Forest | exactitude / kappa | 0,70 / 0,60 |
| Stades | réseau convolutif | exactitude / kappa | 0,73 / 0,64 |
| Stades | réseau convolutif, externe sur Sleep-EDF | exactitude / kappa | voir `docs/demo_nuit.json` |
| Stades | Random Forest, externe sur Sleep-EDF | exactitude / kappa | 0,55 / 0,40 |
| Apnée | Random Forest | AUC-ROC / aire précision-rappel | 0,65 / 0,56 |
| Apnée | réseau convolutif | AUC-ROC / aire précision-rappel | 0,63 / 0,54 |
| Apnée, par personne | Random Forest | Spearman avec l'index clinique | 0,17 |

Par sous-groupe (âge, sexe, IMC, sévérité) : `docs/COVARIABLES_SHHS.md`. Aucun effondrement
net observé, mais 40 personnes : une direction, pas une preuve. Calibration du réseau de
stades : ECE 0,036. En ne classant que la moitié des époques les plus sûres, l'exactitude
atteint 0,89 : c'est le principe de la file de relecture.

## Limites connues

- Une seule dérivation par signal, pas de contexte entre époques : le N1 et les transitions
  sont mal reconnus (F1 N1 ≈ 0,28), comme chez les scoreurs humains mais davantage.
- L'étiquette « apnée » compte toutes les hypopnées annotées, sans critère de désaturation :
  elle vaut 2,7 à 6,6 fois l'index clinique SHHS. Le modèle d'apnée **n'ordonne pas** les
  personnes par sévérité (Spearman 0,17) : il ne doit pas servir à estimer un index.
- 7,5 % des époques SHHS sont saturées (EEG ±125 µV, ECG ±1,25 mV) ; gardées à l'entraînement,
  marquées, mais le modèle les classe au lieu de les refuser. Un module de qualité du signal
  manque.
- La validation externe divise le kappa des stades par 1,5 (autre dérivation, autres appareils).
- SHHS mesure le flux par thermistance : les critères modernes de l'hypopnée reposent sur la
  pression nasale.
- Résultats à une graine ; l'écart entre graines est en cours de mesure.

## Données et droits

SHHS est fournie par le National Sleep Research Resource sous accord d'utilisation : aucune
donnée, aucun identifiant, aucun signal SHHS ne figure dans ce dépôt ni dans la démonstration.
Les signaux de démonstration viennent de PhysioNet (licence ouverte).

## Statut réglementaire

Projet de recherche et de formation. **N'est pas un dispositif médical.** Un logiciel de
pré-scoring utilisé pour une décision diagnostique relèverait vraisemblablement du règlement
européen 2017/745 (classe IIa, règle 11) ; rien ici n'a été fait dans ce cadre.
