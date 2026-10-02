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

## 2026-10-__
- Fait :
- Résultat :
- Bloquée sur :
- Appris :
- Écrit sans aide :
