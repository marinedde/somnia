# Audit des données et du code — Somnia

*3 octobre 2026. Chiffres agrégés seulement : aucun identifiant de participant SHHS, aucun
signal. Les vérifications sont reproductibles avec les scripts cités.*

Pourquoi ce document : les données sont des données de santé, les résultats serviront à
décider de la suite du projet, et deux défauts (ECG inversé, détecteur de pics bruité) ont été
trouvés en route alors qu'ils auraient dû l'être avant. Cet audit répond à trois questions :
d'où viennent les données, sont-elles saines, et le code laisse-t-il passer quelque chose.

## Sommaire

1. Provenance des trois jeux
2. Comptages : personnes, enregistrements, nuits, époques
3. Étiquettes et déséquilibre des classes
4. Qualité des signaux et valeurs aberrantes
5. Intégrité des annotations SHHS
6. L'ECG inversé de SHHS : ce que c'est, ce que ce n'est pas
7. Cohérence entre sources (unités, caractéristiques)
8. Découpages
9. Audit du code
10. Décisions prises, écarts connus, à faire

---

## 1. Provenance des trois jeux

| | Sleep-EDF Expanded | Apnea-ECG | SHHS visite 1 |
|---|---|---|---|
| Source | PhysioNet (Kemp et al.) | PhysioNet (Penzel et al., Challenge 2000) | NSRR, Sleep Heart Health Study |
| Époque | enregistrements 1987-1991 | années 1990 | 1995-1998 |
| Population | volontaires sains, 25 à 101 ans, cassette ambulatoire à domicile (sous-ensemble `SC`) | patients et témoins sélectionnés pour le défi, 27 à 63 ans | population générale, 40 ans et plus, polysomnographie à domicile (Compumedics P-Series) |
| Ce que nous utilisons | EEG Fpz-Cz, 100 Hz, hypnogramme R&K par époque de 30 s | ECG une dérivation, 100 Hz, une étiquette apnée / normal par minute | EEG C4-A1 (canal `EEG`) et ECG, 125 Hz natifs, XML NSRR : stades R&K et événements horodatés |
| Licence | ouverte (PhysioNet) | ouverte (PhysioNet) | accord d'utilisation NSRR : pas de redistribution, stockage hors dépôt, citation |
| Vérifié par nous | fréquences, canaux, unités (volts), 32 annotations non mappées | fréquences, unités (mV), doublon c05 = c06 confirmé par corrélation 1,000 à 80 s de décalage | fréquences par canal lues dans les en-têtes, structure XML, documentation sleepdata.org |
| Non vérifié | âges et sexes (fichier `SC-subjects.xls` non téléchargé) | correspondance enregistrement → personne (non publiée) | — (covariables jointes le 3 octobre : `docs/COVARIABLES_SHHS.md`) |

Les descriptions de population (âges, sélection) viennent de la documentation des bases, pas
de nos fichiers. À confirmer sur les tables de covariables avant toute analyse par sous-groupe.

## 2. Comptages

| | Sleep-EDF | Apnea-ECG | SHHS |
|---|---|---|---|
| Fichiers présents | 30 PSG, 28 hypnogrammes | 35 annotés, 31 avec signal | 297 EDF, 297 XML |
| Enregistrements utilisés | 28 | 31 | 272 (25 exclus : < 4 h de sommeil scoré) |
| **Personnes** | **16** (12 avec deux nuits) | 31 enregistrements, **30 groupes** (c05 = c06) ; 32 personnes réelles d'après PhysioNet, correspondance inconnue | **272** (une nuit par personne en visite 1) |
| Durée d'enregistrement | 21,4 à 23,8 h (cassette ambulatoire : la journée est enregistrée) | 7 à 10 h (428 à 577 minutes) | 3 à 9 h (360 à 1 086 époques, médiane 1 020) |
| Unité étiquetée | époque de 30 s | minute | époque de 30 s |
| Unités par nuit avant / après traitement | 2 056 à 2 858 époques annotées → 552 à 1 153 après retrait de l'éveil hors sommeil | 428 à 577 minutes | 360 à 1 086 époques, toutes gardées |
| Total utilisé | 25 721 époques | 15 116 minutes | 272 598 époques EEG ; 94 033 fenêtres ECG de 60 s en sommeil |
| Sommeil scoré par nuit | — | — | 1,6 à 8,1 h, médiane 5,9 h (avant exclusion) |

Deux conventions différentes à garder en tête :

- Sur Sleep-EDF, l'éveil avant l'endormissement et après le réveil est retiré (sinon 68 %
  d'éveil, la journée étant enregistrée). Sur SHHS, tout est gardé : 28 % d'éveil, une vraie nuit.
- Sur SHHS, la minute ECG est une paire d'époques de 30 s **toutes deux en sommeil**,
  positive si l'une des deux l'est. Les fenêtres contenant de l'éveil sont écartées de la
  tâche apnée (une apnée se définit pendant le sommeil), d'où 94 033 fenêtres et non 136 299.

## 3. Étiquettes et déséquilibre des classes

### Stades (EEG)

| Stade | Sleep-EDF (après élagage) | SHHS |
|---|---|---|
| Wake | 6,3 % | 28,2 % |
| N1 | 8,0 % | 3,4 % |
| N2 | 49,8 % | 41,7 % |
| N3 | 15,3 % | 13,3 % |
| REM | 20,6 % | 13,4 % |
| Non scoré | 0 (32 annotations « ? » ignorées à la lecture) | 11 époques sur 296 890 |

Le N1 est rare partout : c'est la classe que tous les modèles et tous les scoreurs reconnaissent
mal. SHHS a trois fois plus d'éveil, population âgée enregistrée à domicile.

### Apnée (ECG)

| | Apnea-ECG | SHHS (fenêtres de 60 s en sommeil) |
|---|---|---|
| Part positive | 39,6 % | 41,1 % (train 40,5 / val 42,6 / test 42,3) |
| Par enregistrement | de 0 % (témoins `c`) à 96 % (`a01`) : trois groupes construits pour le défi | part d'époques positives pendant le sommeil par nuit : de 2 % à 98 %, médiane autour de 25 % |
| Définition | minute marquée par un expert | époque couverte ≥ 10 s par une apnée (obstructive, centrale, mixte) **ou une hypopnée** |

Le point qui pèse le plus : dans SHHS, **les hypopnées font 85 % des événements comptés**
(62 641 hypopnées, 9 017 apnées obstructives, 1 149 centrales, 23 mixtes sur 297 nuits).
Un modèle « apnée » sur SHHS est donc surtout un modèle « hypopnée », événement plus
discret, défini par une baisse de flux partielle. Sur Apnea-ECG, les minutes positives
viennent de patients à apnées franches. Les deux tâches ne sont pas la même tâche, et c'est
une des raisons de l'écart 0,84 (Apnea-ECG) contre 0,65 (SHHS).

Événements NSRR non comptés dans l'étiquette, par choix : désaturations (43 791), artefacts
SpO2 (8 139), micro-éveils (39 502 + 218 + 56), « Unsure » (155).

## 4. Qualité des signaux et valeurs aberrantes

Détail : `docs/EDA_SHHS.md`, généré par `python_scripts/shhs_eda.py`. Seuils fixés avant de regarder.

| Critère | Sleep-EDF | SHHS |
|---|---|---|
| EEG plat (écart-type < 1 µV) | non mesuré | 0 époque |
| EEG saturé (> 5 % des échantillons à la butée ±125 µV) | non applicable (plage ±192 µV) | **6,7 % des époques**, 16,6 % en éveil contre 2 à 3 % en sommeil : mouvements et yeux |
| ECG plat | non mesuré | 0 époque |
| ECG saturé (> 5 % à ±1,25 mV) | non mesuré | 2,6 % des époques |
| Au moins un critère | — | 7,5 % des époques ; 19 nuits au-delà de 20 % (train 9, val 5, test 5) |
| Fenêtres ECG où aucun battement n'est détecté | 14 sur 15 116 (0,09 %) | 0 |
| Nuits à moins de 4 h de sommeil scoré | — | 25 sur 297, exclues |
| ECG échantillonné à 50 Hz au lieu de 125 | — | **1 nuit** (rééchantillonnée à 100 Hz, qualité de détection moindre) : voir partie 10 |

Décision : les époques aberrantes restent dans les données, **marquées**. Un modèle en
production les verra ; les retirer donnerait un score flatteur. Les scores sont donnés avec et
sans elles (`docs/RESULTATS_SHHS.md`) : l'écart est de 0,01 à 0,03.

## 5. Intégrité des annotations SHHS (297 nuits)

| Vérification | Résultat |
|---|---|
| Durée couverte par les stades = durée de l'EDF | écart 0 s sur les 297 nuits |
| Époques non scorées | 11 sur 296 890 |
| Événements de durée nulle ou négative, débuts négatifs | 0 |
| Apnées et hypopnées de moins de 10 s | 59 (0,08 %) : ne peuvent pas couvrir 10 s d'une époque, donc sans effet |
| Apnées et hypopnées de plus de 120 s | 70 (0,10 %), maximum 197 s : gardées, invraisemblables mais annotées |
| Événements finissant après la fin de l'EDF | 61, dont 17 apnées ou hypopnées : tronqués à la dernière époque |
| Chevauchements entre apnées et hypopnées | 0 |
| Nuits sans aucun événement respiratoire | 0 |
| Montages | 11 variantes de liste de canaux ; `EEG` (C4-A1) et `ECG` présents dans les 297 ; `EEG(sec)` absent de 4 ; le flux s'appelle `NEW AIR` ou `AIRFLOW` |
| Fréquences natives | EEG, EMG 125 Hz ; EOG 50 Hz ; respiration, flux 10 Hz ; SaO2, position, lumière 1 Hz ; ECG 125 Hz sauf une nuit à 50 Hz |
| Plages physiques | EEG ±125 µV (297), ECG ±1,25 mV (297) |

## 6. L'ECG inversé de SHHS : ce que c'est, ce que ce n'est pas

Mesure : signe du pic R aux battements détectés, 60 époques réparties sur chaque nuit.

| Nuits | Résultat |
|---|---|
| 217 | pic R négatif sur plus de 90 % des époques |
| 8 | pic R positif sur plus de 90 % des époques |
| 47 | mixtes, presque toutes entre 70 et 90 % de négatif (bruit de détection sur quelques époques) |

Une polarité **constante toute la nuit** et **partagée par 80 % des personnes** n'est ni un
parasite (qui varie), ni une pathologie (un bloc de branche ou une séquelle d'infarctus ne
touche pas 80 % d'une population dans la même dérivation). C'est la **convention de câblage**
de la dérivation ECG unique de SHHS : deux électrodes thoraciques, et celle qui est branchée
sur l'entrée « + » décide du sens, comme aVR donne un QRS négatif parce qu'elle regarde le
cœur à l'opposé de DII. Les 8 nuits positives correspondent à des électrodes permutées à la
pose, ou à de vraies différences d'axe. Apnea-ECG est à l'endroit (0 % d'époques négatives
sur le premier enregistrement vérifié).

Conséquences :

- Les intervalles RR ne dépendent pas du signe : une fois la détection rendue insensible à la
  polarité (fait le 3 octobre), les caractéristiques de variabilité sont valides.
- L'amplitude QRS est maintenant lue en valeur absolue.
- Pour un réseau convolutif sur le signal brut, il faudra soit normaliser la polarité, soit
  présenter les deux signes à l'entraînement (augmentation), sinon le réseau apprendra que
  « négatif » veut dire « SHHS ».

## 7. Cohérence entre sources (unités, caractéristiques)

Après les corrections du 3 octobre, les mêmes extracteurs tournent sur les deux sources.
Médianes (p5 – p95) par époque :

| Caractéristique | PhysioNet | SHHS | Lecture |
|---|---|---|---|
| EEG écart-type (µV) | 14,3 (7,5 – 36) | 14,9 (7,4 – 64) | mêmes unités ; queue haute SHHS = saturation et éveil |
| EEG puissance delta | 29,9 | 29,1 | cohérent |
| EEG puissance alpha | 1,2 | 3,8 | **dérivation différente** : C4-A1 (centrale) voit plus d'alpha et de fuseaux que Fpz-Cz (frontale) |
| EEG rapport delta | 0,84 | 0,70 | idem : explique en partie la chute en validation externe |
| Fréquence cardiaque (bpm) | 64 (54 – 83) | 64 (50 – 85) | identiques après correction (avant : 82 contre 111) |
| Écart-type RR (s) | 0,062 | 0,038 | plausible (avant : 0,17 et 0,15, bruit du détecteur) |
| Amplitude QRS | 1,38 | 0,53 | gains différents ; sans incidence sur les RR |

Aucune ligne de caractéristiques entièrement nulle sur SHHS, 0,09 % sur Apnea-ECG (segments
plats), aucun NaN.

## 8. Découpages

| Fichier | Tâche | train / val / test | Doublons | Ensembles disjoints | Personnes effectivement vues dans les tableaux |
|---|---|---|---|---|---|
| `data/splits/physionet_v1.json` (versionné) | EEG | 12 / 2 / 2 personnes | 0 | oui | — |
| idem | ECG | 22 / 4 / 4 groupes | 0 | oui | — |
| `data/splits/shhs_v1.json` (hors git, identifiants) | les deux | 192 / 40 / 40 personnes | 0 | oui | 192 / 40 / 40, intersections vides |

Répartition des stades et part d'apnée identiques dans les trois ensembles SHHS (partie 3 et
`docs/EDA_SHHS.md`). Le test SHHS n'a jamais été ouvert. Stratification SHHS sur la charge
d'événements annotés par heure de sommeil, qui dépasse l'index clinique SHHS (celui-ci exige
une désaturation pour compter une hypopnée) : elle sert à répartir, pas à diagnostiquer.

## 9. Audit du code

Méthode : relecture indépendante de tout le code de la chaîne (un second agent, en lecture
seule, sans accès aux données SHHS), puis vérification de chaque point et correction le jour
même. Le relecteur a aussi confirmé par lui-même : MNE convertit bien `uV` et `mV` en volts
(code source 1.13.2), les 28 paires Sleep-EDF ont la même heure de début dans les deux en-têtes,
et les annotations Apnea-ECG tombent toutes exactement sur la grille des minutes.

**Aucune fuite de données trouvée** : normalisation dans le pipeline (ajustée sur l'entraînement
seul), découpages par personne figés, test SHHS fermé, référence de dérive sur l'entraînement.

### Défauts trouvés et ce qui en a été fait

| # | Gravité | Défaut | Effet possible | État |
|---|---|---|---|---|
| 1 | importante | Reproductibilité : à la prédiction, la forêt additionnait les arbres en parallèle dans un ordre dépendant des threads (écart 4·10⁻¹⁶) ; une égalité parfaite pouvait basculer, deux exécutions ne donnaient pas toujours la même métrique au bit près | mesures non reproductibles à la 3e décimale | **corrigé** : entraînement parallèle, inférence séquentielle partout (évaluation, réentraînement, références, API). Test lancé 5 fois : identique |
| 2 | importante | Époques SHHS sans signal (annotations plus longues que l'EDF) complétées par des zéros mais gardant leur stade et leur étiquette : elles seraient apprises | signal nul étiqueté N2 ou apnée | **corrigé** : marquées stade −1, apnée 0 ; test. Non déclenché sur les 272 nuits (durées EDF = XML) |
| 3 | importante | ECG sans battement détectable : valeurs par défaut « plausibles » (RR 0,8 s, 60 bpm) : un segment saturé ressemblait à un cœur normal | une époque illisible classée comme une époque saine | **corrigé** : zéros explicites (0 bpm est impossible, donc reconnaissable) ; test |
| 4 | importante | L'erreur « signal plat » de sleepecg n'était pas attrapée : les 16 caractéristiques tombaient à zéro par le filet générique, avec un `print` | 14 fenêtres Apnea-ECG, 0 sur SHHS | **corrigé** : plat détecté avant l'appel, erreur attrapée, `logging` + compteur d'échecs |
| 5 | importante | Annotations SHHS : un bloc de stade chevauchant le précédent aurait décalé tous les stades suivants par rapport aux événements, sans erreur ; un `Start` manquant était pris pour 0 | désalignement stade / apnée / signal | **corrigé** : refus du fichier dans les deux cas, contrôle final de la durée ; tests. Non observé sur les 297 nuits |
| 6 | importante | L'API ne disait pas l'unité attendue (EEG en volts, ECG en mV) et ne contrôlait aucune plage : des µV envoyés par erreur donnaient une prédiction sur des valeurs ×10⁶ | prédiction absurde sans avertissement | **corrigé** : unités documentées dans le schéma, refus 422 hors plage ou signal plat ; tests. Les tests de l'API utilisaient eux-mêmes des signaux de 1 V : corrigés |
| 7 | mineure | NaN (asymétrie d'une époque constante) remplacés par 0 à l'entraînement mais pas dans l'API : deux chemins pour la même donnée | incohérence entraînement / inférence | **corrigé** : remplacement dans les extracteurs, une seule fois pour tous |
| 8 | mineure | `epoch × 1e6` sur un tableau float16 déborde en NaN ; protégé seulement par un cast en amont | NaN si un appelant passe du float16 | **corrigé** : cast float64 en tête des deux extracteurs ; test |
| 9 | mineure | Une nuit exclue pour « fichier illisible » (téléchargement partiel) n'était jamais réessayée sans `--force` | nuit perdue silencieusement | **corrigé** : seules les exclusions stables sont réutilisées |
| 10 | mineure | F1 macro moyenné sur les classes présentes, F1 par stade sur les 5 classes fixes | deux conventions dans un même tableau | **corrigé** : 5 classes fixes partout |
| 11 | mineure | Variabilité fréquentielle (LF/HF) calculée sur un axe temporel comprimé quand un battement manque | LF/HF approximatif | **différé**, documenté dans le code : à refaire avec les instants réels des pics |
| 12 | mineure | ECG stocké en float16 : pas de 0,001 mV entre 1 et 2 mV (~1,6 pas du convertisseur SHHS) | sans effet sur les caractéristiques actuelles | **différé** : à remesurer avant le réseau convolutif ; passer en float32 coûte 3 Mo par nuit |
| 13 | mineure | Règle d'exclusion « fréquence native < 100 Hz » absente | une nuit à ECG 50 Hz dans l'entraînement de shhs_v1 | **ajoutée** pour les prochaines cohortes ; v1 inchangé, écart documenté |

### Tests

104 tests, dont 25 ajoutés par l'audit. Couverture des modules de la chaîne : `subjects` 100 %,
`evaluation` 100 %, `shhs` 96 %, `splits` 82 %, `shhs_prepare` et `physionet` : les briques
pures (rééchantillonnage, découpage, unités, alignement, fenêtres de 60 s, élagage, appariement)
sont testées sur des signaux synthétiques ; la lecture des fichiers réels ne l'est pas en CI,
par construction (aucune donnée SHHS dans le dépôt).

Propriétés désormais verrouillées par un test :

- identité de personne (3 jeux, doublon c05 = c06) ; découpage disjoint, reproductible, indépendant de l'ordre ; refus de l'ancien découpage ;
- règle des 10 s, désaturations non comptées, trous de stades comblés, chevauchements et blocs incomplets refusés ;
- 125 → 100 Hz (longueur, fréquence, amplitude) ; volts → µV / mV à 1 % près ; alignement époque / signal ; époques sans signal marquées ; fenêtres de 60 s (paires, étiquette max, éveil exclu) ;
- polarité ECG ; aucune valeur inventée sans battement ; float16 sans débordement ; valeurs finies sur une époque constante ;
- validation croisée : chaque personne testée une fois, prédiction hors pli pour chaque époque, reproductible ; l'ancien découpage est bien plus optimiste ;
- API : unités contrôlées, signal plat refusé ; aucun fichier SHHS ni jeton dans le dépôt.

Non couvert, à ajouter : un test bout en bout `preparer_nuit` sur un EDF synthétique écrit sur
disque (demande d'écrire un EDF, à faire avec `edfio`), et un test de grille des annotations
Apnea-ECG qui s'exécute quand les données locales sont présentes.

Après ces corrections, toute la chaîne numérique (PhysioNet et SHHS) a été relancée avec le
code audité ; les chiffres de `docs/RESULTATS.md` et `docs/RESULTATS_SHHS.md` sont ceux de ce
code.

## 10. Décisions prises, écarts connus, à faire

**Décisions figées**

- Étiquette apnée : apnées de tous types et hypopnées, couverture ≥ 10 s par époque, éveil exclu.
- 100 Hz partout, pas d'autre filtre que l'anti-repliement.
- Époques aberrantes gardées et marquées, pas retirées.
- Test SHHS fermé jusqu'à la fin ; toute mesure SHHS est sur la validation.
- Détection des pics R par `sleepecg`, insensible à la polarité, même code à l'entraînement et dans l'API.

**Écarts connus, assumés**

- Une nuit SHHS a un ECG natif à 50 Hz (rééchantillonné à 100 Hz). Elle est dans l'ensemble
  d'entraînement de `shhs_v1`. Une règle « ECG natif < 100 Hz » est ajoutée pour les prochaines
  cohortes ; `shhs_v1` n'est pas modifié (un découpage ne se modifie pas), l'écart est documenté.
- 70 apnées ou hypopnées de plus de 120 s (0,1 %) gardées telles qu'annotées.
- Apnea-ECG : découpage par enregistrement faute de correspondance personne ; une fuite
  résiduelle entre deux enregistrements d'une même personne reste possible.
- Covariables jointes (`docs/COVARIABLES_SHHS.md`) : âge médian 58,5 ans (minimum 40), 45 % de femmes,
  IMC médian 26,5 ; les trois ensembles se ressemblent. **Mon index annoté vaut 2,7 fois l'index clinique
  `ahi_a0h3a` et 6,6 fois `ahi_a0h4`** (toutes les hypopnées comptées, sans critère de désaturation) ;
  l'ordre des personnes est conservé (Spearman 0,82) mais pas le niveau. Les classes de sévérité à
  publier sont celles de l'index clinique : avec `ahi_a0h4`, 47 % de la cohorte est « normale » et 8 %
  « sévère » ; la validation et le test n'ont que 2 personnes sévères chacun.

**À faire**

- Décider, avant l'étape 4, si l'étiquette « apnée » doit se rapprocher de la définition clinique
  (hypopnée comptée seulement avec une désaturation associée), ou rester celle des annotations.
  Les désaturations sont dans les XML mais le NSRR prévient qu'elles peuvent manquer ou être décalées.
- Module « qualité du signal » qui refuse une époque saturée plutôt que de la classer.
- Pour le réseau convolutif : normalisation ou augmentation de polarité ECG.
