# Journal de bord — Somnia

Cinq lignes par séance. Le but n'est pas d'être exhaustive mais de pouvoir
répondre à « comment travaillez-vous ? » avec des faits datés.

Règle des zones (voir `docs/PLAN_REPRISE.md`, partie 2) :
**rouge** = j'écris seule · **orange** = relecture après coup · **verte** = plomberie, aide libre.

---

## 2026-10-01
- Fait : état des lieux du dépôt par l'assistant (zone verte) ; environnement `.venv` avec PyTorch ;
  garde-fous SHHS (`.gitignore`, tests) ; script de téléchargement lisant `NSRR_TOKEN` ;
  suppression de `deploy.yml`/`render.yaml` ; un seul `requirements.txt` pour l'API.
- Résultat : rien de mesuré encore. Chiffres actuels toujours fuités (EEG 0,83 ; ECG AUC 0,97 en découpage aléatoire, 0,70 par enregistrement).
- Bloquée sur : —
- Appris : Sleep-EDF local = 16 personnes pour 28 enregistrements ; Apnea-ECG local = 31 enregistrements sur 35, sans correspondance personne publiée, sauf c05 = c06 (même enregistrement).
- Écrit sans aide : non. Tâche 0.1 (`somnia/subjects.py` + tests) écrite par l'assistant à ma demande, donc hors règle rouge : à refaire de mémoire demain (test de la page blanche) avant de passer à 0.2.

## 2026-10-02
- Fait : étape 0 complète (0.2 → 0.13), écrite par l'assistant à ma demande : extraction avec la personne,
  découpage par personne versionné, validation croisée par personne, réentraînement, API qui lit ses chiffres,
  README et tableau de résultats, vraies courbes ROC.
- Résultat : EEG exactitude 0,74 ± 0,06 et kappa 0,63 ± 0,08 (avant 0,81 / 0,73) ; ECG AUC 0,77 ± 0,08
  (avant 0,97). Le N1 reste à 0,37 de F1.
- Bloquée sur : —
- Appris : la fuite ECG était bien plus grosse que la fuite EEG ; les écarts-types sont larges avec 16 et 30 personnes.
- Écrit sans aide : non. Page blanche à faire, dans l'ordre : subjects.py, splits.py, evaluation.py, physionet.py.

## 2026-10-02 (suite) — étape 1, pilote SHHS
- Fait : jeton en place (régénéré après une fuite dans une trace), TLS via truststore, motif de fichiers corrigé
  (identifiants à partir de 200001), 9 nuits téléchargées (335 Mo). Inspection des en-têtes EDF, lecture d'un vrai
  XML, lecteur d'annotations validé, tableau des 9 nuits, 6 figures par nuit.
- Résultat : deux montages dès 9 fichiers (flux `NEW AIR` ou `AIRFLOW`) ; fréquences conformes à la doc ;
  IAH estimé de 10 à 63/h ; hypopnées 10 fois plus nombreuses que les apnées obstructives.
- Bloquée sur : —
- Appris : les noms de canaux changent de casse et d'espaces d'un fichier à l'autre ; MNE masque les fréquences
  natives (lire l'en-tête EDF soi-même) ; sleepdata.org n'envoie pas son certificat intermédiaire.
- Écrit sans aide : non (assistant). Page blanche à faire : `somnia/shhs.py` (lecteur XML), après ceux de l'étape 0.
- Décisions prises : apnée = apnées obstructives, centrales, mixtes + hypopnées ; époque positive si ≥ 10 s
  couvertes ; éveil exclu de la tâche apnée. À revoir : garder 125 Hz ou revenir à 100 Hz (conseil : 100 Hz).

## 2026-10-02 (suite) — étape 2, lancement
- Fait : décisions 300 nuits / 100 Hz ; téléchargement des nuits 200001-200300 lancé (~10 h) ; chaîne EDF + XML →
  npz par nuit écrite et validée sur le pilote (8 retenues / 9) ; découpage stratifié par sévérité prêt.
- Résultat : ~10 Mo par nuit préparée ; EEG ~20 µV d'écart-type par époque, ECG ~0,05 mV ; 1 178 époques
  d'apnée sur 8 140 (14 %).
- Bloquée sur : le téléchargement (il tourne).
- Appris : 3 identifiants manquent entre 200001 et 200299 (296 fichiers, pas 299) ; une nuit sur 9 a moins de
  4 h de sommeil scoré.
- Écrit sans aide : non (assistant). Page blanche à faire : `somnia/shhs_prepare.py`.

## 2026-10-03 — étape 2 terminée
- Fait : 297 nuits téléchargées (téléchargeur réécrit après un blocage : flux, délai, reprise, MD5) ; préparation
  des 297 nuits en 10 min ; découpage par personne stratifié, figé en v1 (hors git) ; diagramme de cohorte dans le README.
- Résultat : 272 nuits retenues, 25 exclues (< 4 h de sommeil scoré) ; 272 598 époques, 27 % avec apnée/hypopnée ;
  192 / 40 / 40 personnes.
- Bloquée sur : —
- Appris : compter des époques d'apnée par heure surestime la sévérité (183 « sévères » sur 272) ; même par
  événements, l'index annoté NSRR (médiane 40/h) dépasse l'index clinique SHHS, qui exige une désaturation.
  La connexion du soir divise le débit par 10 : lancer les gros téléchargements l'après-midi.
- Écrit sans aide : non (assistant). Page blanche à faire : `shhs_make_split.py` (stratification) après `shhs_prepare.py`.

## 2026-10-03 (suite) — EDA SHHS avant l'étape 3
- Fait : EDA des 272 nuits préparées (`shhs_eda.py` → `docs/EDA_SHHS.md`, deux figures agrégées) : amplitudes par
  époque, stades par nuit, équilibre par ensemble, valeurs aberrantes avec seuils écrits avant de regarder.
- Résultat : stades Wake 28 % / N1 3 % / N2 42 % / N3 13 % / REM 13 %, même répartition dans train, val et test ;
  30 % d'époques « apnée » pendant le sommeil ; EEG saturé (butée ±125 µV) sur 6,7 % des époques, ECG saturé sur
  2,6 % ; 19 nuits avec plus de 20 % d'époques aberrantes.
- Bloquée sur : —
- Appris : mon premier seuil « EEG extrême > 200 µV » ne pouvait rien voir, la plage physique est ±125 µV : il faut
  lire l'en-tête EDF avant de fixer un seuil. La saturation EEG est à 17 % en éveil contre 2-3 % en sommeil :
  mouvements et yeux, pas ondes lentes. Les époques aberrantes restent dans les données, marquées.
- Écrit sans aide : non (assistant).

## 2026-10-__
- Fait :
- Résultat :
- Bloquée sur :
- Appris :
- Écrit sans aide :
