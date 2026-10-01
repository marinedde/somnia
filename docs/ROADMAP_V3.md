# Somnia — Roadmap v3 (bascule SHHS)

> **Statut** : document de cadrage, écrit le 28/07/2026, au moment de l'accès accordé
> au dataset SHHS (NSRR). Révisé après arbitrage : nom **Somnia**, entraînement
> **en local**, pas d'hébergement HDS à ce stade.
>
> **Objectif affiché du projet, tel que tranché :** *améliorer le modèle grâce à SHHS,
> jusqu'à un niveau de fiabilité crédible en santé.* Pas de diffusion publique des
> données, pas de mise en production clinique pour l'instant.

---

## 0. La rupture à comprendre avant tout le reste

L'archi v2 raisonne en **fenêtres** : 3000 points d'EEG entrent, un stade sort.
SHHS raisonne en **nuits** : un EDF de 8 h, un XML d'annotations, un sujet.

Ce n'est pas un détail d'implémentation. Ça change l'unité de travail de tout le système :

| | v2 (aujourd'hui) | v3 (cible) |
|---|---|---|
| Objet manipulé | une fenêtre de 30 s | un **enregistrement** (nuit complète) |
| API | `POST /predict` synchrone, réponse en ms | **job asynchrone** : upload → analyse → résultat |
| Sortie | un label + probas | **hypnogramme complet + liste d'événements horodatés + indices (AHI, TST, efficacité)** |
| Métrique | accuracy sur fenêtres | **kappa de Cohen par nuit**, erreur sur l'AHI, F1 par événement |
| Interface | une prédiction affichée | **timeline de nuit navigable**, revue clinicien |
| Stockage | rien à stocker | signaux volumineux + métadonnées + corrections |

Tant qu'on n'a pas acté ça, tout le reste (choix de stack, choix de modèle) est mal posé.
**C'est le vrai changement de la v3.** Le passage RF → CNN, à côté, est secondaire.

Conséquence immédiate : le classifieur de fenêtre ne disparaît pas, il devient un
**composant interne** appelé en boucle sur la nuit, pas le produit.

---

## 1. Contraintes matérielles — à régler AVANT le premier téléchargement

Trois blocages concrets, mesurés sur ta machine :

**1.1 — Le repo est dans iCloud Drive.**
`~/Library/Mobile Documents/com~apple~CloudDocs/JEDHA/somnia` — 5,9 Go déjà synchronisés.
Si tu télécharges SHHS ici, iCloud tente de pousser des centaines de Go vers le cloud,
et les fichiers « évincés » (icône nuage) font échouer les scripts au milieu d'un
entraînement. **Sortir le code ET les données d'iCloud** est le prérequis n°1.

**1.2 — 138 Go libres, SHHS1 complet pèse de l'ordre de 250–400 Go en EDF.**
(≈ 5 800 nuits × 40–70 Mo — à confirmer sur l'onglet *Files* de NSRR.)
Tu ne prends pas tout. Tu prends un sous-ensemble, et tu le convertis tout de suite
dans un format compact.

**1.3 — En revanche, la machine suffit.**
Apple M4, 10 cœurs, **24 Go de mémoire unifiée**. PyTorch tourne sur le backend MPS, et
24 Go de mémoire unifiée, c'est davantage que les 16 Go d'un T4 gratuit sur Kaggle.
Un encodeur CNN + couche séquentielle sur 500 nuits en HDF5 compact s'entraîne ici,
de l'ordre de **quelques heures par run** — à mesurer sur les 50 premiers sujets avant
de dimensionner la suite.

**Décision : on entraîne en local.** Ça résout trois problèmes d'un coup :
- pas de quota cloud, pas de transfert de dizaines de Go à chaque itération ;
- **plus aucune question de DUA NSRR** — rien ne sort de la machine, donc pas de
  redistribution, pas de stockage chez un tiers à faire valider ;
- le cloud reste disponible plus tard, si un modèle plus lourd le justifie.

> **Pourquoi pas Google Drive.** Drive n'est pas un système de fichiers à accès
> aléatoire : une boucle d'entraînement qui fait des milliers de petites lectures par
> époque sur un Drive monté rampe ou casse. Et c'est le problème iCloud recréé — un
> client de synchro qui se bat contre toi. Drive reste acceptable comme *parking* d'un
> gros fichier HDF5 unique à recopier en local au début d'une session Colab. Jamais
> comme source de lecture d'entraînement.

**Le plan qui découle de ces trois points :**

```
SSD externe (ou ~/data hors iCloud)
   └── shhs/
        ├── raw/          EDF bruts — sous-ensemble seulement, jamais versionné
        └── processed/    HDF5 compact — c'est CE dossier qui monte sur le GPU
```

Budget de conversion, canaux utiles rééchantillonnés à 100 Hz en `float16` :
- staging seul (EEG×2, EOG, EMG) → ≈ 17 Mo/nuit → **500 nuits ≈ 9 Go**
- staging + respiratoire (+ ECG, flux, 2 ceintures, SpO₂) → ≈ 29 Mo/nuit → **500 nuits ≈ 15 Go**

À cette taille, tout tient en local sur le M4. Kaggle (T4 gratuit, ~30 h/semaine) reste
le plan B si un modèle plus lourd le justifie plus tard.

**Ordre de grandeur : 300–500 sujets suffisent largement pour un premier modèle
séquentiel honnête.** Monter à 2 000 se fait après, quand le pipeline est rodé.

---

## 2. Stack — réponse directe à « Vercel + Supabase, ou Scalingo ? »

**Les deux, mais pas pour la même chose.** La bonne réponse n'est pas un hébergeur,
c'est une **séparation en deux plans** — et c'est cette séparation qui est l'actif
architectural, pas le choix du fournisseur.

### Plan R — Recherche & portfolio (à construire maintenant)

Ne voit **que des données publiques dé-identifiées** : SHHS, Sleep-EDF, MESA.
Aucune donnée de patient réel n'y entre, jamais.

| Brique | Choix | Pourquoi |
|---|---|---|
| Front revue clinicien | **Next.js sur Vercel** | rapide à sortir, gratuit, excellent en portfolio |
| Auth + métadonnées + file de jobs | **Supabase** (Postgres + Auth + RLS) | tu obtiens en une soirée ce qui prendrait deux semaines à écrire |
| Stockage signaux | **Supabase Storage** (ou S3-like) | jamais iCloud, jamais git |
| **Calcul** | **worker Python séparé** (Modal, Scalingo, RunPod, ou ta machine) | ⚠️ voir ci-dessous |
| Entraînement | **local, M4 / MPS** | rien ne sort de la machine, pas de quota, pas de DUA à arbitrer |
| Tracking | MLflow (conservé) | comparaison RF vs séquentiel |

> ⚠️ **Le piège Vercel.** Les fonctions Vercel sont serverless, courtes, sans GPU, avec
> une limite de taille de bundle. Elles ne peuvent pas charger MNE + PyTorch et
> mouliner un EDF de 300 Mo. **Vercel sert l'interface, jamais l'analyse.**
> L'analyse tourne dans un worker Python long-running qui pioche dans une file de jobs.
> Cette coupure interface/worker est exactement la couture qui te permettra de
> déménager vers un hébergeur HDS sans réécrire le front.

### « Ne vaudrait-il pas mieux prendre Scalingo d'emblée ? » — non

Question posée, réponse argumentée, parce que l'intuition est bonne mais vise le
mauvais axe.

**1. L'obligation HDS ne se déclenche pas ici.** Elle porte sur l'hébergement de
données de santé de personnes **identifiables**, **pour le compte d'un tiers**. SHHS
est une cohorte de recherche dé-identifiée, mise à ta disposition pour entraîner un
modèle. Tu n'héberges rien pour personne. L'obligation naîtra le jour où un médecin
déposera l'enregistrement d'un vrai patient — pas avant.

**2. Scalingo ne fait rien pour la fiabilité du modèle.** C'est une plateforme
d'exécution d'applications, pas d'entraînement : pas de GPU, pas de vocation à
mouliner des centaines de nuits. Prendre l'offre HDS maintenant, c'est payer un cadre
juridique qui ne s'applique pas encore, sur un axe qui n'est pas celui qu'on cherche à
améliorer. La fiabilité clinique ne vient d'aucun choix d'hébergeur — voir §5.

**3. Ce qui est vrai dans l'intuition** : la crainte de devoir tout refaire. Elle se
traite par l'**abstraction**, pas par l'hébergeur. Une couche d'accès aux données —
`storage`, `db`, `queue` — avec une implémentation Supabase remplaçable. Deux jours de
travail maintenant contre une réécriture plus tard. **C'est ça, faire « d'emblée ».**

**Le déclencheur à surveiller**, celui qui fera basculer vers Scalingo HDS : *la
première fois qu'une donnée de patient réel entre dans le système*. À ce moment-là,
et seulement là, l'offre HDS devient obligatoire. (À noter aussi : pour un pilote avec
un médecin, traiter les fichiers **en local, sans jamais les héberger**, évite
entièrement la question.)

### « Et Databricks ? » — pas maintenant, même logique

Databricks résout un problème de **passage à l'échelle** : traiter distribué, sur un
cluster, ce qui ne tient pas sur une machine. Or l'échelle n'est pas le goulot ici.

- **Volume** : 500 nuits en HDF5 compact ≈ 15 Go, et 10 cœurs pour le prétraitement.
  Spark distribue un travail qui tient déjà largement en local — on paierait la
  complexité d'un cluster sans en tirer le bénéfice.
- **Nature du calcul** : l'entraînement d'un réseau séquentiel sur signaux 1D est du
  **mono-GPU**, pas du Spark. Databricks sait le faire (cluster mono-nœud GPU), mais
  c'est alors une machine louée avec beaucoup de cérémonie autour.
- **Coût** : DBU + compute cloud, facturés à l'heure. L'édition gratuite est très
  limitée et sans GPU (l'offre a changé récemment — à vérifier si le sujet revient).
- **Données** : y monter SHHS recrée exactement la question de stockage chez un tiers
  qu'on vient d'éliminer en restant en local.
- **MLflow** : déjà en place, et il tourne très bien hors Databricks. On ne gagne rien
  de ce côté-là.

**Le seul argument sérieux est un argument de CV**, et il est légitime : Databricks est
très demandé en mission. Mais alors il faut l'assumer comme tel — c'est un objectif de
compétence, pas un besoin du projet — et le traiter sur un jeu de données jouet, pas en
y déversant SHHS.

**Quand ça deviendrait justifié** : passer aux ~5 800 nuits de SHHS1 *plus* plusieurs
cohortes NSRR, avec un prétraitement EDF → features massivement parallèle. Là, le
« embarrassingly parallel » sur des milliers de fichiers est un vrai cas d'usage Spark.
C'est J6+, pas maintenant.

### Plan C — Clinique (à ne PAS construire maintenant, mais à prévoir)

Le jour où un vrai médecin dépose l'enregistrement d'un vrai patient :

- **Hébergement HDS obligatoire** en France (art. L.1111-8 du Code de la santé
  publique) dès lors que tu héberges des données de santé pour le compte d'un tiers.
  **Scalingo est certifié HDS** — c'est un bon choix, PaaS français, ergonomie proche
  de Heroku. Alternatives : OVHcloud, Clever Cloud, Outscale, ou les offres HDS
  d'AWS/Azure/GCP en région France.
- **Vercel et Supabase managé ne sont pas HDS.** Ils restent parfaits pour le plan R.
  Pour le plan C, la brique équivalente devient un Postgres managé chez l'hébergeur HDS
  (ou un Supabase auto-hébergé, plus lourd).
- Pseudonymiser ne suffit pas : un EEG de nuit reste une donnée de santé au sens de
  l'art. 9 du RGPD, et l'obligation HDS porte sur l'hébergement, pas sur l'identification.

**Ce que ça implique concrètement dès maintenant :** aucune ligne de code du plan R ne
doit supposer « Supabase » en dur. Une couche d'accès aux données (`storage`, `db`,
`queue`) avec une implémentation Supabase — remplaçable. C'est 2 jours de travail
aujourd'hui, contre une réécriture plus tard.

### Sur la licence des données — point à ne pas rater

Le *Data Use Agreement* de la NSRR **interdit la redistribution** des données SHHS.
Or ton déploiement actuel est un **Space HuggingFace public**.

→ Jamais de signal SHHS, même un extrait, dans le repo, dans une image Docker, ou dans
un Space public. Les signaux de démo (`data/demo/*.npy`) doivent rester issus de
Sleep-EDF (licence ouverte) ou être synthétiques. Et il faut citer la NSRR + SHHS dans
toute présentation ou publication.

---

## 3. Architecture cible v3

```
┌───────────────────────────────────────────────────────────────────────────┐
│ C5 · EXTENSIONS PRODUIT — prévu, non construit                             │
│    comptes multi-praticiens · facturation · multi-centres · Plan C (HDS)   │
├───────────────────────────────────────────────────────────────────────────┤
│ C4 · REVUE CLINICIEN            [Next.js / Vercel]                         │
│    hypnogramme navigable · file de revue triée par incertitude             │
│    valider / corriger / rejeter · rapport validé (Claude API) · export     │
├───────────────────────────────────────────────────────────────────────────┤
│ C3 · ANALYSE                    [worker Python, GPU/CPU]                   │
│    ├── STAGING     : encodeur par époque + modèle séquentiel sur la nuit   │
│    ├── RESPIRATOIRE: détection d'événements (apnée / hypopnée) → AHI       │
│    ├── QUALITÉ     : artefacts, canaux morts, signal exploitable ou non    │
│    └── sortie = probas + incertitude + horodatage, jamais un verdict       │
├───────────────────────────────────────────────────────────────────────────┤
│ C2 · PRÉTRAITEMENT              [Python / MNE]                             │
│    lecture EDF · mapping de montage · rééchantillonnage · normalisation    │
│    PAR SUJET · découpage en époques · parsing XML NSRR → labels            │
├───────────────────────────────────────────────────────────────────────────┤
│ C1 · INGESTION & DONNÉES        [Supabase (plan R) | HDS (plan C)]         │
│    upload EDF · file de jobs · sujets / enregistrements / annotations      │
│    audit trail des corrections cliniciens                                  │
└───────────────────────────────────────────────────────────────────────────┘
```

**La couche 2 est la clé de la polyvalence PSG / EEG seul / ECG seul que tu veux.**
Un `MontageMapper` qui dit « ce fichier contient tel canal, à telle fréquence, sous tel
nom » découple les modèles du format d'entrée. C'est ce qui permet à un même modèle de
staging de tourner sur SHHS (C4-A1 à 125 Hz), sur du Sleep-EDF (Fpz-Cz à 100 Hz), ou
sur un EEG ambulatoire, sans réécrire le modèle. **Les noms de canaux sont le vrai
cauchemar du domaine — c'est là qu'il faut mettre la rigueur.**

---

## 4. Les modèles — ce qu'on entraîne sur SHHS

### 4.1 Staging (stades du sommeil)

**Le point technique décisif : une époque isolée ne suffit pas.** Un humain qui score
une nuit regarde le contexte (ce qui précède, ce qui suit, les transitions plausibles).
Un classifieur qui ne voit qu'une fenêtre de 30 s plafonne — c'est très exactement le
plafond ~0,70 que tu as rencontré, et il ne se casse pas en changeant de famille de
modèle.

Architecture cible, en deux temps :
1. **Encodeur par époque** — CNN 1D sur le signal brut (EEG + EOG + EMG) → un vecteur
   par époque de 30 s.
2. **Modèle séquentiel sur la nuit** — bi-LSTM ou convolutions dilatées / transformer
   sur la suite des vecteurs → un stade par époque, avec le contexte.

C'est le principe des références du domaine (DeepSleepNet, SeqSleepNet, U-Sleep).
U-Sleep est particulièrement pertinent pour toi : il est conçu pour être **agnostique
au montage**, ce qui rejoint directement ton objectif « PSG mais aussi EEG seul ».

**Métriques honnêtes, et c'est un différenciateur :**
- kappa de Cohen **par nuit** (pas seulement l'accuracy globale, qui est gonflée par
  la surreprésentation du N2)
- F1 par stade — le N1 est toujours le point faible, chez les modèles *comme chez les
  humains*
- **Comparer au désaccord inter-scoreurs.** L'accord entre deux techniciens humains
  tourne autour de 80–83 % sur ce type de données. Un modèle à 84 % n'est pas
  « imparfait », il est *dans la zone du désaccord humain*. Positionner tes résultats
  contre ce plafond, c'est le genre de rigueur qui te crédibilise instantanément
  auprès d'un clinicien — et c'est la suite naturelle de ta correction de fuite de
  données.

### 4.2 Événements respiratoires

Formulation correcte : **détection d'événements** (début, durée, type) sur les canaux
respiratoires + SpO₂, pas classification binaire de fenêtres. La sortie utile est
l'**AHI**, plus une liste d'événements horodatés que le médecin peut relire.

**Limite scientifique à assumer d'emblée :** SHHS1 mesure le flux par **thermistance**,
pas par pression nasale. Les critères modernes de l'hypopnée reposent sur la pression
nasale. Un modèle entraîné sur SHHS1 ne se transposera donc pas tel quel sur un
enregistrement de labo actuel. Ce n'est pas rédhibitoire — c'est à savoir, à écrire,
et à tester le jour où tu ajoutes un second dataset.

Métriques : F1 par type d'événement, **erreur absolue sur l'AHI**, et un
**Bland-Altman AHI modèle vs AHI de référence** — c'est la figure que lit un
pneumologue.

### 4.3 Ce qui reste de la v2

La RF sur features **reste, comme baseline explicite** dans MLflow. « J'ai comparé,
voilà l'écart, voilà pourquoi » vaut mieux que « j'ai pris le modèle le plus à la
mode ». Le `feature_extractor.py` (16 features, correction V → µV) est réutilisé tel quel.

---

## 5. Fiabilité — le protocole, puisque c'est l'objectif

> « Je veux un modèle fiable **car** données de santé. »
>
> La fiabilité clinique ne vient d'aucun choix d'hébergeur, d'aucune certification,
> d'aucune architecture. Elle vient d'un **protocole décidé avant d'entraîner**.
> Décidé après, il ne prouve plus rien : on aura choisi les règles en regardant les
> résultats. Cette section est donc à figer **maintenant**, avant J2.

### 5.1 Le découpage, gelé une fois pour toutes

- **Split par sujet, jamais par époque.** Deux époques de la même nuit se ressemblent
  trop : les mélanger entre train et test, c'est reproduire exactement ta fuite de
  données à une autre échelle.
- Train 60 % / val 20 % / test 20 %, **tiré une seule fois, avec une seed écrite**, et
  la liste des sujets sauvegardée dans un fichier versionné du repo.
- **Stratifier** le tirage sur la sévérité (AHI) et l'âge — sinon on se retrouve avec
  un test sans cas sévères, et un score qui ne veut rien dire.
- **Le test ne s'ouvre qu'à la fin.** Chaque coup d'œil intermédiaire est une fuite
  lente : on finit par sélectionner le modèle qui plaît au test.

### 5.2 La calibration — le point qui fait tenir toute l'architecture

C'est le plus important, et le plus souvent oublié.

Ta couche 4 **trie par incertitude** : on montre au médecin les cas où le modèle
hésite. Ce tri n'a de sens que si « 0,82 » signifie vraiment « juste 82 fois sur 100 ».
Or les réseaux profonds sont **typiquement sur-confiants** : ils sortent 0,95 sur des
cas où ils ont raison 7 fois sur 10. Avec des probabilités non calibrées, ton tri par
incertitude classe du bruit, et tu envoies le clinicien regarder les mauvais segments —
la promesse de gain de temps s'effondre en silence, sans qu'aucune métrique
d'accuracy ne le signale.

- **Mesurer** : diagramme de fiabilité, ECE (*expected calibration error*), score de Brier.
- **Corriger** : *temperature scaling* ajusté sur le set de validation. Un seul
  paramètre, quelques lignes, très efficace.
- **Ce n'est pas un raffinement.** C'est la condition de validité de la couche 4.

### 5.3 La validation externe — la seule preuve qui compte en santé

Un score sur le test SHHS mesure la performance **sur SHHS** : mêmes appareils, même
protocole, mêmes scoreurs, même population. En santé, ce qui compte est la performance
sur un autre centre, un autre matériel, une autre population.

→ **Tenir une seconde cohorte NSRR (MESA ou MrOS) entièrement hors de l'entraînement**,
comme test externe, ouvert une seule fois.

> « Kappa 0,79 sur SHHS, 0,71 sur MESA jamais vue » est une phrase infiniment plus
> crédible que « kappa 0,82 ». **La chute entre les deux *est* le résultat
> intéressant** — c'est elle qui dit à un clinicien ce qui se passera chez lui.

### 5.4 Les métriques par sous-groupe

Une moyenne globale masque les défaillances. Découper systématiquement les résultats
par **âge, sexe, IMC, sévérité de l'AHI, qualité du signal**.

Un modèle à 84 % global qui tombe à 65 % chez les plus de 70 ans est dangereux
précisément là où on voudrait l'utiliser. SHHS embarque toutes ces covariables : le
tableau ne coûte rien à produire, et c'est typiquement ce qui distingue un travail
sérieux d'une démo.

### 5.5 Le droit de dire « je ne sais pas »

Deux mécanismes de refus, à des étages différents :

- **En amont** — qualité du signal insuffisante (canal plat, saturation, artefacts
  massifs) → **pas de prédiction du tout** sur ce segment.
- **En aval** — incertitude au-dessus d'un seuil → l'époque part en revue humaine
  *sans proposition*, plutôt qu'avec une proposition douteuse.

Mesurer la **courbe précision / couverture** : si le modèle passe de 84 % à 93 % en
refusant 15 % des époques, c'est une information directement exploitable par le
clinicien, et un argument de vente honnête.

> Un modèle qui ne se tait jamais n'est pas fiable. Il est bavard.

### 5.6 Le plafond humain comme référentiel

L'accord entre deux techniciens humains tourne autour de **80–83 %** sur ce type de
données. Deux conséquences :

- Ne pas viser 95 %. Un tel score serait le **signe d'une fuite**, pas d'une réussite.
- Le N1 est mal classé par les modèles **comme par les humains**. Le dire explicitement
  vaut mieux que de le cacher dans une moyenne.

### 5.7 Traçabilité

Chaque run MLflow doit porter : hash git du code, hash de la liste des sujets du split,
seed, version du jeu de données prétraité, hyperparamètres, métriques complètes.
Sans ça, un résultat n'est pas un résultat — c'est une anecdote qu'on ne saura pas
reproduire dans trois mois.

### 5.8 Les pièges spécifiques à SHHS

- **Déséquilibre massif** : le N2 représente environ la moitié des époques, le N1
  autour de 5 %. L'accuracy seule ment mécaniquement → kappa + F1 macro + F1 par stade.
- **Normalisation par sujet, statistiques calculées sur le train uniquement.** Une
  normalisation globale estimée sur l'ensemble du dataset est une fuite — subtile,
  fréquente, et elle gonfle les scores.
- **Règles d'exclusion écrites avant de voir les résultats** : nuits tronquées, canaux
  manquants, durée d'enregistrement insuffisante. Exclure après coup, c'est choisir
  ses données.
- **Thermistance vs pression nasale** (cf. §4.2) : limite à documenter, pas à masquer.

### 5.9 Définition de « fiable » — critères de sortie proposés

À ajuster une fois la baseline mesurée, mais à écrire **avant** d'entraîner :

| Critère | Cible |
|---|---|
| Kappa de Cohen, test interne SHHS | ≥ 0,75 |
| Kappa de Cohen, cohorte externe jamais vue | ≥ 0,70 |
| ECE après calibration | < 0,05 |
| Pire sous-groupe (kappa) | ≥ 0,60 |
| Erreur absolue médiane sur l'AHI | < 5 événements/h |

Un modèle qui atteint ça, sur une cohorte externe, avec des probabilités calibrées et
un tableau par sous-groupe, est défendable devant un clinicien. C'est ça, l'objectif —
pas un chiffre d'accuracy isolé.

---

## 6. Le parcours « centre de dermato » appliqué au sommeil

Ton intuition est juste, et elle se transpose presque terme à terme :

| Dermato | Somnia |
|---|---|
| Photo prise par un opérateur | Nuit enregistrée par le technicien |
| Analyse IA immédiate | Pré-scoring automatique (staging + événements + qualité) |
| Tri : lésion suspecte ou non | **Tri par incertitude × sévérité** |
| Le dermatologue ne revoit que ce qui compte | Le médecin ne relit que les segments douteux |
| Compte-rendu signé par le médecin | **Rapport validé = ce que le médecin a validé** |

**Ce que ça donne en interface, concrètement :**

1. **Dépôt** → l'EDF arrive, un job part.
2. **Rapport de qualité d'abord.** Avant tout résultat : « canal EMG plat après 2 h,
   3 % du signal saturé ». Un modèle qui annonce d'emblée ce qu'il ne peut pas lire
   gagne une confiance qu'aucune métrique n'achète.
3. **Pré-scoring** → hypnogramme + événements + AHI estimé, **tout marqué comme
   non validé**.
4. **File de revue triée.** Pas « voici 960 époques ». Plutôt : *« 47 époques où le
   modèle hésite, 12 événements limites — 15 minutes de relecture »*. C'est là que se
   trouve le gain de temps réel, et c'est ce que tu vends.
5. **Validation en un geste** — valider / corriger / rejeter, chaque action tracée
   (ton `ValidationStore` fait déjà ça, il faut juste le passer en base et l'attacher
   à un enregistrement).
6. **Rapport final** — rédigé par Claude **à partir des seuls éléments validés**.
   Le LLM rédige, il ne décide pas.

**La différence importante avec la dermato, à ne pas gommer :** dans les centres de
dépistage, l'IA fait du **tri de population**. L'équivalent qui a le plus de valeur
opérationnelle chez toi, ce n'est pas « remplacer le scoring », c'est **prioriser la
file d'attente d'un labo** : quels enregistrements lire en premier quand il y a six
mois de retard. Cette valeur-là est réelle, mesurable, et n'émet aucune information
diagnostique — c'est de l'organisation de flux.

---

## 7. Réalité réglementaire — correction d'un point de la v2

L'archi v2 pose que « le clinicien valide → pas de certification requise ». C'est
**partiellement vrai, et il vaut mieux le savoir maintenant.**

Sous le règlement européen MDR 2017/745 et le guide MDCG 2019-11, un logiciel qui
fournit une information **utilisée pour une décision diagnostique** est généralement un
dispositif médical, et la règle 11 le classe le plus souvent en **classe IIa**. La
présence d'un humain dans la boucle **réduit le risque, mais n'exonère pas
automatiquement**. Un logiciel de pré-scoring PSG vendu à un labo tomberait
vraisemblablement dans ce périmètre.

Ce qui te garde réellement hors du champ **aujourd'hui** :
- **La destination revendiquée.** Somnia est un outil de recherche et un démonstrateur,
  non destiné au diagnostic — à écrire noir sur blanc dans l'interface et le README.
- **Les données.** Tant que tu ne traites que des cohortes de recherche
  dé-identifiées, tu n'es ni sur du soin, ni sur du patient identifiable.
- **La nature de la sortie.** Un outil de *priorisation de file d'attente* ou de
  *contrôle qualité du signal* n'émet pas d'information diagnostique.

Ce qu'il faut donc faire : **avancer, sans changer la revendication**. Et le jour où
un labo veut l'utiliser sur de vrais patients, deux voies s'ouvrent — restreindre les
revendications au flux de travail, ou assumer un parcours MDR IIa. Cette décision se
prend à ce moment-là, pas maintenant. *(Je ne suis pas juriste : à faire confirmer
avant tout usage réel.)*

Le mérite de cette lecture, c'est qu'elle ne change rien à la roadmap — elle change
seulement la phrase que tu écris sur ta page d'accueil.

---

## 8. Roadmap — six jalons

> Principe conservé de la v2 : **un front à la fois**.
> Estimations à temps partiel.

### J0 — Assainir le terrain · ½ journée
- [x] ~~Trancher le nom~~ → **Somnia**, acté.
- [ ] Sortir le repo d'iCloud → `~/dev/somnia`.
- [ ] Purger le dépôt : `venv/` (2,9 Go), `mlruns/` (607 Mo), `htmlcov/`, `.coverage`,
      `.DS_Store`, la présentation (déplacée en `docs/presentation_AIA_v3.pptx`). Les modèles (319 Mo) → Git LFS ou
      releases.
- [ ] Créer `~/data/shhs/` hors iCloud (ou SSD externe), vérifier l'espace.
- [ ] Récupérer le token NSRR (profil NSRR) et repérer la commande exacte de
      téléchargement sur l'onglet *Files*.

### J1 — Socle données SHHS · 1–2 semaines
- [ ] Lire `10-montage-and-sampling-rate.md` et `05-polysomnography-intro` du dataset :
      noter précisément canaux, fréquences, unités. **Ne rien coder avant.**
- [ ] Télécharger un sous-ensemble : **50 sujets d'abord** (validation du pipeline),
      puis 300–500.
- [ ] `shhs_loader.py` : EDF + XML `annotations-events-nsrr` → HDF5 compact
      (époques 30 s, labels stades, événements horodatés, métadonnées sujet).
- [ ] **Figer le protocole de fiabilité du §5 AVANT d'entraîner** : split par sujet
      stratifié (AHI, âge), seed écrite, liste des sujets versionnée dans le repo,
      règles d'exclusion écrites.
- [ ] Normalisation par sujet, statistiques estimées **sur le train uniquement**.
- [ ] EDA : distribution des stades, distribution des AHI, qualité par sujet,
      sujets à exclure.
- [ ] Re-loguer la baseline RF sur SHHS dans MLflow → **la nouvelle référence**.

### J2 — Staging séquentiel · 2–3 semaines
- [ ] Encodeur CNN par époque, puis couche séquentielle sur la nuit.
- [ ] Entraînement **en local** (PyTorch MPS) depuis le HDF5 — mesurer le temps par run
      sur 50 sujets avant de lancer sur 500.
- [ ] Évaluation : kappa par nuit, F1 par stade, matrice de confusion,
      **positionnement vs désaccord inter-scoreurs**.
- [ ] **Calibration** (§5.2) : diagramme de fiabilité + ECE, puis *temperature scaling*
      sur la validation. Sans ça, le tri par incertitude de la couche 4 ne vaut rien.
- [ ] **Tableau par sous-groupe** (§5.4) : âge, sexe, IMC, sévérité.
- [ ] **Courbe précision / couverture** (§5.5) : que gagne-t-on en refusant N % ?
- [ ] Comparaison honnête RF vs séquentiel dans MLflow.

### J3 — Événements respiratoires · 2–3 semaines
- [ ] Détection d'événements sur canaux respiratoires + SpO₂.
- [ ] Calcul de l'AHI, Bland-Altman vs référence SHHS.
- [ ] Documenter la limite thermistance vs pression nasale.

### J4 — Plateforme v3 · 3–4 semaines
- [ ] Schéma Supabase : `subjects`, `recordings`, `jobs`, `epochs`, `events`,
      `validations`.
- [ ] Worker Python : consomme la file, analyse la nuit, écrit les résultats.
- [ ] Couche d'abstraction stockage/DB (le point de bascule vers HDS).
- [ ] Front Next.js sur Vercel : upload, suivi de job, résultats.
- [ ] Streamlit **archivé** en l'état comme démo du bloc Jedha — pas maintenu en
      parallèle.

### J5 — Couche revue clinicien · 2–3 semaines
- [ ] Hypnogramme interactif + navigation dans le signal.
- [ ] File de revue triée par incertitude × sévérité.
- [ ] Valider / corriger / rejeter, tracé en base.
- [ ] Rapport validé généré par Claude, export PDF/JSON.
- [ ] Rapport de qualité du signal en tête d'analyse.

### J6 — Généralisation & ouverture · continu
- [ ] **Test externe** sur la 2ᵉ cohorte NSRR (MESA ou MrOS). À demander tôt, mais
      **à n'ouvrir qu'une seule fois, à la fin** (§5.3) — c'est la preuve de fiabilité
      qui compte, elle se périme dès qu'on regarde deux fois.
- [ ] `MontageMapper` : faire tourner le staging sur un EEG hors PSG.
- [ ] Boucle d'apprentissage actif à partir des corrections.
- [ ] Module recherche qEEG (voir §9).

---

## 9. Idées d'évolution — les six qui valent le coup

Classées par rapport valeur / risque.

**1. Le mode « second lecteur ».**
Le modèle score la nuit, on la compare au scoring du technicien, et on affiche la
**carte de désaccord**. Ça ne remplace personne, ça sert au contrôle qualité et à la
formation des techniciens. Zéro exposition réglementaire (aucune information
diagnostique nouvelle), valeur immédiate pour un labo. **C'est la porte d'entrée la
plus facile chez un clinicien** — et de loin la meilleure première conversation terrain.

**2. Re-scoring d'archives.**
Les labos et équipes de recherche ont des milliers de PSG anciennes, scorées avec des
critères hétérogènes sur vingt ans. Les re-scorer de façon homogène a une vraie valeur
scientifique, et c'est explicitement de la recherche. Ton pipeline SHHS *est* déjà ça.

**3. Le rapport de qualité du signal, avant toute analyse.**
Techniquement modeste, disproportionnellement crédibilisant. Un outil qui dit d'abord
ce qu'il ne peut pas lire inspire une confiance que les courbes ROC n'achètent pas.

**4. L'interopérabilité comme levier d'adoption.**
Exporter les annotations en EDF+ ou en XML compatible NSRR, pour que les résultats se
rouvrent dans les visualiseurs existants. C'est ce qui fait la différence entre « joli
prototype » et « outil qu'on peut essayer sans rien changer à son flux ».

**5. Le module recherche qEEG — ta direction neurosciences.**
Compartimenté, non prescriptif, purement descriptif : puissance d'ondes lentes,
densité de fuseaux, fréquence du pic alpha, pente aperiodic du spectre. SHHS est une
base de choix pour ça — grande cohorte, âges variés, suivi cardiovasculaire jusqu'en
2010. Le lien EEG de sommeil ↔ vieillissement cérébral est un champ de recherche actif
et publié (« brain age » estimé depuis l'EEG de sommeil). **C'est le prolongement
naturel de ta conversation sur l'EEG vieillissant, et ça reste hors du produit :
notebooks séparés, sortie descriptive, aucune recommandation.**

**6. La mise en relation de facteurs, descriptive.**
SHHS embarque des covariables riches (âge, IMC, tension, événements
cardiovasculaires). Montrer des associations — sans jamais énoncer de causalité ni de
conseil — nourrit ta curiosité sur le multifactoriel tout en restant du côté honnête
de la ligne.

**Ce que je ne recommande pas maintenant :** l'analyse temps réel, le support des
objets connectés grand public, et tout ce qui touche à la recommandation
d'amélioration du sommeil à partir de l'enregistrement d'un individu. Le troisième
n'est pas un problème de faisabilité — c'est la ligne réglementaire nette dont on a
déjà parlé.

---

## 10. Ce qu'on garde, ce qu'on archive

**Gardé :** FastAPI · Docker · CI GitHub Actions · MLflow · `feature_extractor.py`
(baseline RF) · le concept de `ValidationStore` (à passer en base) · le monitoring de
drift · l'intégration Claude · et surtout **la narration de rigueur** (fuite de
données corrigée, comparaison au désaccord humain).

**Archivé, non maintenu :** `streamlit_app.py` (1 010 lignes) reste tel quel comme
livrable Jedha et démo v2. On ne maintient pas deux interfaces.

**Supprimé du dépôt :** `venv/`, `mlruns/`, `htmlcov/`, `.coverage`, `.DS_Store`,
`fix_pipeline.py` et `rebuild_pipeline.py` (scripts de dépannage ponctuels), le `.pptx`.

---

## 11. Prochaine action, unique

**J0, aujourd'hui : sortir le projet d'iCloud et préparer un emplacement de données
hors iCloud.** Tout le reste en dépend, et c'est une demi-journée.

Ensuite seulement : le token NSRR et le premier lot de 50 sujets.
