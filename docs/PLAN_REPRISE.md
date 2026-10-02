# Somnia — Plan de reprise (1er octobre 2026)

> Ce document remplace la lecture séparée de `ROADMAP_V3.md` (juillet, orienté produit/EEG)
> et du plan « 11 · Reprendre Somnia avec SHHS » (septembre, orienté ECG/Implicity).
> Il part de l'état **vérifié** du dépôt au 1er octobre, pas de ce que les deux documents
> supposaient. Les parties 4 et 5 sont à suivre dans l'ordre.

## Sommaire

1. Ce que le dépôt contient vraiment
2. Les trois fils, et pourquoi ils ne se contredisent pas
3. Le plan réconcilié
4. Étape 0 : ce qui est fait, ce qui te reste
5. Étapes 1 à 6 : ajustements par rapport au plan de septembre
6. Décisions à prendre, et quand
7. Ce que je n'ai pas vérifié

---

## 1. Ce que le dépôt contient vraiment

Vérifié le 1er octobre, fichier par fichier.

### 1.1 Les données et les modèles

| Point | Ce qu'affichent le README et l'API | Ce qui est vrai |
|---|---|---|
| Sleep-EDF (EEG) | « 28 sujets » | **16 personnes**, 28 enregistrements. 12 personnes ont deux nuits (SC4001 et SC4002 = personne 00, nuits 1 et 2). Deux PSG sans hypnogramme (SC4032, SC4122) ignorés |
| Apnea-ECG | « 35 sujets » | **31 enregistrements** présents en local (a19, c07, c09 sans fichier `.dat`). La base PhysioNet en compte 35, issus de **32 personnes** d'après sa description : certains enregistrements sont la même personne. À vérifier sur la page PhysioNet avant de dire « découpage par personne » |
| Découpage EEG | — | `train_test_split` **sur les époques**, stratifié par stade (`02_preprocessing.ipynb`, cellule 12). Fuite confirmée : les deux nuits d'une même personne, et des époques voisines de la même nuit, sont réparties entre train, val et test |
| Découpage ECG déployé | AUC 0,967 | Même `train_test_split` sur les minutes (cellule 21). Le modèle servi par l'API (`somnia_ecg_pipeline.joblib`, 4 juin, renommé depuis) est entraîné par `retrain.py` sur ce découpage |
| ECG par enregistrement | « note : AUC 0,70 inter-sujets » | Mesuré une fois dans le notebook 03 (19 enregistrements en train, 12 en test) : **AUC 0,70, exactitude 0,69**. Un essai antérieur avec 4 enregistrements en train donnait 0,55. Aucune validation croisée, aucun écart-type |
| EEG par personne | — | **Jamais mesuré.** Le 0,834 est le seul chiffre qui existe |
| Métriques enregistrées | `/model-info` | `training_metrics.json` : `"eeg": null`. Le modèle EEG date du notebook du 7 mai et n'est jamais passé par `retrain.py`. Les chiffres EEG de l'API sont codés en dur dans `ml_model.py` |
| Référence de dérive | — | `baseline_stats.json` ne contient que `apnea`. `/monitoring/drift/features?task=sleep_stage` répond « baseline absente » |
| Courbe ROC « avant » | figure `leakage_validation.png` | Les points de la courbe « split aléatoire » sont **écrits à la main** (`fpr_rand = [0, 0.02, 0.05, 0.10, 1]`, notebook 03b, cellule 4) |

### 1.2 Le code et l'infrastructure

| Point | État |
|---|---|
| Emplacement | `~/dev/somnia`, **hors iCloud**. L'ancienne copie iCloud n'existe plus. J0 de la roadmap de juillet est fait |
| Code SHHS | Aucun. Dossiers `~/data/shhs/raw` et `processed` créés le 28 juillet, vides |
| Script de téléchargement de juin | Jeton en dur dans le code (placeholder), destination `./data/raw/shhs` **à l'intérieur du dépôt**. Remplacé (voir 4.1) |
| Python | 3.13.5 système, sans environnement virtuel. `mne`, `wfdb`, `scikit-learn` présents ; **`torch` et `sleepecg` absents** |
| Machine | Apple M4, 24 Go, 117 Go libres. Suffisant pour 300 à 500 nuits SHHS prétraitées |
| Tests | 38 tests d'API. **Aucun test sur les données ni sur le découpage** |
| CI | Deux workflows de déploiement contradictoires : `ci.yml` (HuggingFace) et `deploy.yml` (Render, mort). Deux fichiers de dépendances avec des versions différentes (`scikit-learn` 1.5.2 contre 1.7.1) |
| Dépôt | 174 Mo de signaux Apnea-ECG suivis par git (licence ouverte, mais lourd), la présentation `.pptx`, des modèles doublons dans `notebooks/models/`, `mlruns/` sur disque (607 Mo, ignoré) |
| Démo en ligne | Les signaux de démo (`data/demo/*.npy`) viennent de PhysioNet. Rien à changer pour l'accord NSRR |

### 1.3 Ce que ça veut dire

La phrase « j'ai reconstruit le pipeline avec un découpage par patient » n'est vraie pour aucun des deux modèles déployés. Elle est vraie pour **une expérience** ECG dans un notebook, avec un seul tirage et un chiffre affiché à côté d'une courbe dessinée à la main. C'est exactement le genre de détail qu'un examinateur chez Implicity trouvera en dix minutes. L'étape 0 existe pour ça.

---

## 2. Les trois fils, et pourquoi ils ne se contredisent pas

Tu as trois raisons de reprendre Somnia, qui tirent en apparence dans trois directions.

| Fil | D'où il vient | Ce qu'il demande |
|---|---|---|
| **Le produit** | Ton idée de départ : un généraliste formé à la polysomnographie met 4 h à lire une nuit | Une sortie **par nuit** (hypnogramme, événements, index), et surtout une **file de relecture triée** : « 47 époques douteuses, 15 minutes » au lieu de 960 époques |
| **L'EEG** | Le jury AIA : les logiciels d'analyse EEG vieillissent | Un modèle de stades **robuste au montage**, et à terme des mesures descriptives d'EEG quantitatif |
| **Implicity** | Candidature ingénieure deep learning ; leur métier : signaux cardiaques, beaucoup de patients, peu d'étiquettes | Un réseau convolutif sur ECG, un **pré-entraînement auto-supervisé**, un découpage par patient irréprochable, une fiche modèle |

**Ce qui les réunit :** SHHS enregistre l'EEG et l'ECG **de la même nuit**, chez des milliers de personnes, à domicile. Un seul lecteur de fichiers, un seul découpage par personne, un seul tableau de résultats servent les trois fils. La différence est l'ordre dans lequel tu branches les têtes de modèle :

```
                 ┌── tête EEG : stades  ───────► hypnogramme ──┐
 SHHS ─► lecteur ─┤                                             ├─► sortie par nuit ─► file de relecture
 (EDF + XML)      └── tête ECG : apnée  ───────► index ─────────┘        (le produit)
                        ▲
                        └── pré-entraînement contrastif (Implicity)
```

Le plan de juillet voulait construire d'abord la sortie par nuit et l'interface. Le plan de septembre voulait d'abord le pré-entraînement ECG. **Les deux ont raison sur le fond et tort sur l'ordre :** avant l'un ou l'autre, il faut des modèles mesurés honnêtement sur SHHS. C'est l'objet des étapes 0 à 4. La sortie par nuit (fil produit) coûte peu une fois les deux têtes entraînées : c'est une boucle sur les époques plus un calcul d'index. Elle vient à l'étape 6.

**Ce qui est reporté, explicitement :** Supabase, Next.js, Vercel, hébergement HDS, Databricks, module EEG quantitatif, rapport rédigé par un LLM. Rien de tout ça ne fait progresser un modèle. Tu y reviendras si le freelance se concrétise. La roadmap de juillet reste la référence pour ce jour-là, en particulier ses parties 2 (séparation interface / calcul), 5 (calibration, refus, sous-groupes) et 7 (réglementaire).

---

## 3. Le plan réconcilié

Les sept étapes du plan de septembre sont conservées. Trois ajouts viennent de juillet, parce qu'ils coûtent peu et comptent beaucoup devant un clinicien : la **calibration** (étape 4), le **refus de prédire** quand le signal est mauvais (étape 4), la **sortie par nuit** (étape 6).

| Étape | Contenu | Chiffre produit | Sert surtout |
|---|---|---|---|
| 0 | Mesurer et corriger Somnia sur les données actuelles | Exactitude, kappa, AUC **par personne**, avec écart-type | Les trois |
| 1 | Pilote SHHS : 10 nuits lues, tracées, comptées | Tableau de 10 lignes | Les trois |
| 2 | Cohorte 200 à 500 personnes, découpage figé, diagramme de cohorte | Effectifs | Les trois |
| 3 | Références : classe majoritaire, Random Forest, validation externe PhysioNet ↔ SHHS | Deux lignes du tableau final | Implicity, produit |
| 4 | Réseau convolutif supervisé, EEG et ECG, avec 100 %, 10 %, 1 % des étiquettes, **plus calibration et courbe précision/couverture** | Quatre lignes | Implicity, EEG |
| 5 | Pré-entraînement contrastif de l'encodeur ECG | Deux lignes et une conclusion | Implicity |
| 6 | Sortie par nuit (hypnogramme, index, file triée par incertitude), fiche modèle, README | Une démo honnête | Produit, EEG |

Le tableau de résultats unique du plan de septembre (partie 4) reste le livrable central. Ajoute-lui une colonne « tâche » (stades / apnée) puisque tu auras les deux.

---

## 4. Étape 0 : ce qui est fait, ce qui te reste

> **Mise à jour du 2 octobre : l'étape 0 est terminée**, à la demande de Marine et écrite par
> l'assistant (donc hors règle rouge : chaque module ci-dessous est à refaire de mémoire).
>
> | Tâche | Où | Résultat |
> |---|---|---|
> | 0.1 | `somnia/subjects.py` | 16 personnes Sleep-EDF, 30 groupes Apnea-ECG |
> | 0.2 | `somnia/physionet.py`, `python_scripts/prepare_features.py` | 25 721 époques EEG, 15 116 minutes ECG, avec la personne |
> | 0.3 | `somnia/evaluation.py`, `python_scripts/evaluate_cv.py` | EEG : exactitude 0,74 ± 0,06, kappa 0,63 ± 0,08 (avant : 0,81 / 0,73) |
> | 0.4 | idem | ECG : AUC 0,77 ± 0,08, aire PR 0,70 ± 0,07 (avant : 0,97 / 0,95) |
> | 0.5 | `somnia/splits.py`, `python_scripts/make_split.py`, `retrain.py`, notebook 02 | `data/splits/physionet_v1.json`, plus de `train_test_split` dans la chaîne |
> | 0.6 | `tests/test_split.py` | 8 tests, dont un qui refuse l'ancien découpage |
> | 0.7 | `python_scripts/retrain.py`, `app/model_metrics.py` | `/model-info` lit `training_metrics.json` : chiffres par personne, écart-type, chiffres « avant » |
> | 0.8 | `retrain.py` | Seuils : kappa EEG ≥ 0,50, AUC ECG ≥ 0,65 |
> | 0.9, 0.10 | `README.md` | Réécrit : état réel, tableau de résultats, section « Fuite de données » |
> | 0.11 | `models/baseline_stats.json` | Référence de dérive EEG et ECG |
> | 0.12 | CI | Fait le 1er octobre |
> | 0.13 | `data/figures/fuite_roc_ecg.png` | Les deux courbes ROC sont calculées |
>
> Non fait : la diapositive 3 (la présentation est sortie du dépôt). Le tableau de résultats
> vit dans `docs/RESULTATS.md`, généré par `evaluate_cv.py`.
>
> **Test de la page blanche, dans l'ordre conseillé :** `subjects.py` → `splits.py` →
> `evaluation.py` (la validation croisée) → `physionet.py`. Un par jour.

### 4.1 Fait pendant la séance du 1er octobre (zone verte)

| Quoi | Où |
|---|---|
| Environnement de recherche dédié, avec PyTorch, MNE, sleepecg, MLflow, pytest | `.venv/`, `requirements-research.txt`. Activer : `source .venv/bin/activate` |
| Garde-fous SHHS : `data/shhs/`, `*.edf`, `*-nsrr.xml`, `*.npz`, `.env` ignorés par git | `.gitignore` |
| Tests qui échouent si un fichier SHHS ou un jeton NSRR est suivi par git | `tests/test_repo_hygiene.py` |
| Script de téléchargement : jeton lu dans `NSRR_TOKEN`, destination `~/data/shhs/raw`, refus d'écrire dans le dépôt ou iCloud, mode `--dry-run` | `python_scripts/shhs_download.py` |
| Suppression du déploiement Render (`deploy.yml`, `render.yaml`) ; un seul `requirements.txt` pour l'API ; CI et README mis à jour | tâche 0.12 |
| Journal de bord | `JOURNAL.md` |

Rien n'est commité. Relis le `git diff`, puis commite toi-même.

### 4.2 Ce qui te reste (zone rouge, dans l'ordre)

Les numéros sont ceux du plan de septembre. J'ajoute ce que l'audit change.

| N° | Tâche | Ce que l'audit ajoute |
|---|---|---|
| 0.1 | Fonction `identifiant_personne(nom_fichier)` | Sleep-EDF : `SC4ssN…` → personne `ss` (caractères 3 et 4). Apnea-ECG : PhysioNet **ne publie pas** la correspondance enregistrement → personne (vérifié le 1er octobre dans `additional-information.txt`), mais signale que **c05 et c06 sont le même enregistrement** décalé de 80 s. Découpe donc par enregistrement, mets c05 et c06 dans le même groupe, et écris cette limite dans le README |
| 0.2 | Colonne `personne` à côté de chaque époque | Les fichiers `X_*.npy` actuels n'ont pas cette information : il faut relancer l'extraction depuis `data/raw` et `data/raw_apnea`. Sauvegarde un seul tableau `(features, étiquette, personne)` par tâche, le découpage se fait après |
| 0.3 | Validation croisée par personne, EEG | `StratifiedGroupKFold(5)` ou `LeaveOneGroupOut` sur 16 personnes. Rapporte exactitude, F1 macro, **kappa**, F1 par stade, moyenne ± écart-type |
| 0.4 | Idem ECG | AUC, aire précision-rappel, F1 apnée. Attention : les enregistrements `c` n'ont presque aucune apnée, un pli peut n'en contenir aucune. `StratifiedGroupKFold` répartit les classes ; vérifie quand même |
| 0.5 | Remplacer `train_test_split` dans le notebook 02 **et** dans `retrain.py` | `retrain.py` lit des `.npy` déjà découpés : il faut qu'il reçoive le tableau complet et le fichier de découpage par personne |
| 0.6 | Test automatique : aucune personne commune entre train et test | `ensemble_train.isdisjoint(ensemble_test)`. Mets-le dans `tests/`, il tournera en CI avec les autres |
| 0.7 | Réentraîner, mettre à jour `/model-info` | Les chiffres EEG sont codés en dur dans `app/ml_model.py` (`MODEL_METADATA`) : fais-les lire depuis `training_metrics.json` comme le ferait l'ECG |
| 0.8 | Abaisser les seuils de `retrain.py` | Actuels : exactitude EEG ≥ 0,75, AUC ECG ≥ 0,65. À régler après 0.3 et 0.4 |
| 0.9 | Corriger README, diapositive, fiches | 16 personnes / 28 enregistrements ; 31 enregistrements ECG ; CNN Keras `cnn_best_model.h5` d'octobre 2025 à ne plus mentionner |
| 0.10 | Section README « Fuite de données » | Deux fuites : le découpage par époque, et les deux nuits par personne. Les deux chiffres avant/après |
| 0.11 | Référence de dérive EEG | `retrain.py --task eeg` l'écrit tout seul une fois 0.5 fait |
| 0.13 | Vraie courbe ROC « avant » | Refais le découpage aléatoire une fois, pour la figure, et écris dans la légende que c'est le découpage fuité |

**Critère de fin de l'étape 0 :** `/model-info` affiche des chiffres par personne, la CI est verte avec le test 0.6, le README ne contient plus un chiffre sans source.

### 4.3 À quoi t'attendre

L'ECG l'a déjà montré : AUC 0,97 → 0,70 en passant au découpage par enregistrement. L'EEG suivra probablement le même chemin, peut-être moins fort : les caractéristiques spectrales dépendent moins de la personne que la variabilité cardiaque. Si tu obtiens 0,70 à 0,78 d'exactitude et un kappa de 0,55 à 0,65, c'est un Random Forest honnête sur 16 personnes. Note les deux chiffres, avant et après : c'est ta deuxième histoire de fuite, et celle-là tu l'auras trouvée seule.

---

## 5. Étapes 1 à 6 : ajustements par rapport au plan de septembre

Le plan de septembre reste la référence pour le détail de chaque étape. Voici seulement ce que l'audit et la roadmap de juillet y changent.

### Étape 1 : pilote

- Le script est prêt : `python python_scripts/shhs_download.py --dry-run`, puis sans `--dry-run` une fois `NSRR_TOKEN` exporté. Dix nuits SHHS pèsent de l'ordre de 500 Mo à 1 Go.
- Lis **les deux** canaux dès le pilote : l'EEG (C4-A1 ou C3-A2) et l'ECG. Le lecteur d'annotations extrait stades et événements d'un seul coup. C'est ce qui fait que les trois fils partagent le même code.
- La roadmap de juillet le disait et c'est vrai : **les noms de canaux sont le vrai problème.** Note dans ton journal le nom exact et la fréquence de chaque canal dans les dix fichiers. S'ils diffèrent d'un fichier à l'autre, tu écris une petite table de correspondance ; c'est le début du `MontageMapper` dont parlait juillet.

### Étape 2 : cohorte

- Un seul fichier de découpage pour les deux tâches. Stratifie au moins sur la sévérité (index d'apnées de SHHS, disponible dans les tables de covariables) : sinon ton test peut n'avoir aucun cas sévère.
- Le fichier contient des identifiants de participants : garde-le dans `data/splits/` (ignoré par git) tant que tu n'as pas relu l'accord. Publie les effectifs et la graine.

### Étape 3 : références

- La validation externe va dans les deux sens et pour les deux tâches : SHHS → PhysioNet et PhysioNet → SHHS. Pour l'ECG, les étiquettes ne sont pas définies pareil (une par minute contre des événements horodatés) : il faudra ramener SHHS à des minutes pour comparer, et le dire.

### Étape 4 : réseau supervisé

- Deux ajouts de juillet, un jour chacun : **calibration** (diagramme de fiabilité, erreur de calibration attendue, puis mise à l'échelle par température ajustée sur la validation) et **courbe précision / couverture** (que gagne-t-on en refusant les 10 % d'époques les moins sûres ?). Ces deux figures sont la base de la file de relecture triée. Sans calibration, trier par incertitude trie du bruit.
- Pour les stades, une époque isolée plafonne : le contexte de la nuit compte. Ne t'en occupe pas à l'étape 4 ; note-le comme piste et traite-le à l'étape 6 si le temps le permet (une couche récurrente ou convolutive sur la séquence de vecteurs d'époques).

### Étape 5 : pré-entraînement

- Rien à changer. C'est l'étape Implicity. Si tu passes un entretien avant d'y être, dis que c'est la prochaine étape et explique les transformations que tu as choisies et pourquoi.

### Étape 6 : sortie par nuit et documentation

- C'est ici que l'idée de départ redevient visible : une fonction qui prend une nuit entière, applique les deux modèles époque par époque, et renvoie hypnogramme, liste d'événements, index estimé, temps de sommeil, et la liste des époques dont la probabilité calibrée est en dessous d'un seuil. Quelques dizaines de lignes, pas une plateforme.
- La démo en ligne reste sur PhysioNet. Une nuit complète de Sleep-EDF suffit pour montrer la file de relecture.
- Fiche modèle : reprends le gabarit de septembre. Ajoute la ligne « population non couverte : fibrillation auriculaire, stimulateur cardiaque » et dis en entretien que tu veux la mesurer.

---

## 6. Décisions à prendre, et quand

| Décision | Quand | Mon avis |
|---|---|---|
| Option A (3 demi-journées par semaine) ou B (1 à 2) | Fin de la semaine 2, après l'étape 0 | Les semaines 1 et 2 sont identiques dans les deux cas. Décide avec les réponses d'Implicity en main |
| Événements comptés comme « apnée » | Étape 1, après avoir lu un XML | Apnées de tous types et hypopnées, comme l'index clinique. Note le choix |
| Seuil de recouvrement pour étiqueter une époque | Étape 1 | Au moins 10 s sur 30 |
| 125 Hz ou 100 Hz | Étape 1 | 100 Hz, pour la validation externe sur PhysioNet et pour réutiliser ton code |
| Découpage par enregistrement ou par personne pour Apnea-ECG | Étape 0, tâche 0.1 | Par personne si PhysioNet publie la correspondance ; sinon par enregistrement, en l'écrivant |
| Publier le fichier de découpage SHHS | Étape 2 | Non, tant que l'accord n'est pas relu |
| Reprendre la roadmap produit (Supabase, interface, HDS) | Après l'étape 6, ou si une mission freelance l'exige | Pas avant |

---

## 7. Ce que je n'ai pas vérifié

- **Les termes de ton accord NSRR.** Les règles supposées (pas de redistribution, stockage sécurisé, citation) sont celles qu'on trouve d'habitude. Relis-le avant le premier téléchargement.
- **Les chiffres SHHS** (effectifs, fréquences, noms de canaux, structure XML) : ils restent à confirmer sur tes fichiers à l'étape 1, comme le disait le plan de septembre.
- **Le téléchargement réel** : la signature de `download_nsrr` a été vérifiée dans le code source de `sleepecg` (`db_slug, subfolder, pattern, shallow, data_dir`) et les fichiers arrivent sous `data_dir/shhs/polysomnography/…`. Le script n'a tourné qu'en `--dry-run` ; les noms de sous-dossiers NSRR (`edfs/shhs1`, `annotations-events-nsrr/shhs1`) sont ceux de ton script de juin, à confirmer sur l'onglet *Files* de sleepdata.org.

Vérifié, en revanche : Apnea-ECG ne fournit pas de correspondance enregistrement → personne ; seul le doublon c05/c06 est documenté.
- **Les notebooks n'ont pas été réexécutés.** Les chiffres de la partie 1 viennent des sorties enregistrées dedans.
