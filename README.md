# 🧠 Somnia — Pipeline MLOps pour l'Analyse Polysomnographique

> Pipeline MLOps complet pour la classification automatique des stades de sommeil (EEG) et la détection d'apnées du sommeil (ECG).

**Jedha Bootcamp — AIA 2026 | Marine Deldicque**

---

## 🚀 Démos en ligne

| Composant | URL |
|-----------|-----|
| 🔬 API FastAPI | [marinedde-somnia-api.hf.space](https://marinedde-somnia-api.hf.space/docs) |
| 🧠 Dashboard Streamlit | [marinedde-somnia-dashboard.hf.space](https://marinedde-somnia-dashboard.hf.space) |

---

## 🏗️ Architecture

```
somnia/
├── app/                        # API FastAPI
│   └── main.py                 # Endpoints REST + monitoring drift
├── streamlit_app.py            # Dashboard (EEG + ECG + rapport IA + validation clinique)
├── models/                     # Modèles entraînés (.joblib)
├── notebooks/                  # Exploration et entraînement
├── tests/                      # Tests unitaires
├── mlruns/                     # Tracking MLflow
├── data/demo/                  # Signaux démo (.npy)
├── python_scripts/             # Scripts utilitaires
├── Dockerfile                  # Containerisation API
├── requirements.txt            # Dépendances API
├── requirements-dashboard.txt  # Dépendances dashboard
└── .github/workflows/ci.yml    # CI/CD GitHub Actions
```

---

## 🖥️ Fonctionnalités du Dashboard

**🧠 Analyse EEG — Classification des stades du sommeil**
- Visualisation signal EEG brut + densité spectrale (bandes δ, θ, α, β)
- Prédiction du stade (Wake / N1 / N2 / N3 / REM) avec probabilités
- Validation clinique médecin intégrée

**❤️ Analyse ECG — Détection des apnées**
- Visualisation signal ECG + SpO₂, AHI
- Prédiction binaire (Normal / Apnée) avec probabilités
- Données cliniques complémentaires (ronflement, somnolence diurne)
- Validation clinique médecin intégrée

**📊 Rapport complet**
- Synthèse EEG + ECG combinée
- Rapport clinique généré par IA (Claude — Anthropic) en < 5 secondes
- Export `.txt` et `.json`

**📈 Monitoring**
- Statistiques d'utilisation en temps réel
- Drift de confiance + drift des features (vs baseline d'entraînement)
- Historique des dernières prédictions
- Validations médecin persistées (`POST /validations`)

---

## 🤖 Modèles

### Modèle 1 — Classification des stades du sommeil (EEG)
- **Algorithme** : Random Forest (scikit-learn)
- **Dataset** : [Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/) (PhysioNet)
- **Features** : 16 features temporelles et spectrales extraites du signal EEG
- **Classes** : Wake, N1, N2, N3, REM
- **Endpoint** : `POST /predict/sleep-stage`

### Modèle 2 — Détection des apnées du sommeil (ECG)
- **Algorithme** : Random Forest (scikit-learn)
- **Dataset** : [Apnea-ECG Database](https://physionet.org/content/apnea-ecg/1.0.0/) (PhysioNet)
- **Features** : features spectrales et temporelles extraites du signal ECG
- **Classes** : Normal, Apnée
- **Endpoint** : `POST /predict/apnea`

**Tracking** : MLflow

---

## 🔄 Réentraînement automatisé

Après le preprocessing (`notebooks/02_preprocessing.ipynb`) :

```bash
python python_scripts/retrain.py              # EEG + ECG
python python_scripts/retrain.py --task eeg   # EEG seul
python python_scripts/retrain.py --dry-run    # Vérifier les données
```

Produit :
- `models/somnia_eeg_pipeline.joblib` / `models/somnia_ecg_pipeline.joblib`
- `models/baseline_stats.json` — référence pour le drift des features
- `models/training_metrics.json` — métriques val/test

Seuils par défaut : accuracy EEG ≥ 0.75, AUC ECG ≥ 0.65 (sinon le script refuse de déployer).

Workflow GitHub manuel : `.github/workflows/retrain.yml`

---

## ⚙️ Pipeline MLOps

```
Signal EEG brut                Signal ECG brut
      ↓                               ↓
Extraction features            Extraction features
(16 features EEG)              (features ECG)
      ↓                               ↓
StandardScaler → RF            StandardScaler → RF
(stades sommeil)               (détection apnée)
      ↓                               ↓
      └──────────────┬────────────────┘
                     ↓
         API FastAPI (Docker)
                     ↓
       Dashboard Streamlit + Claude AI
                     ↓
    CI/CD GitHub Actions → HuggingFace Spaces
```

---

## 🛠️ Stack technique

| Couche | Technologie |
|--------|-------------|
| Modèles | scikit-learn, Random Forest |
| Tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| IA générative | Claude (Anthropic API) |
| Containerisation | Docker |
| Déploiement | HuggingFace Spaces |
| CI/CD | GitHub Actions |

---

## 📂 Données

Les données ne sont pas incluses dans ce repo.

| Dataset | Source | Usage |
|---------|--------|-------|
| Sleep-EDF Database | [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) | Classification stades EEG |
| Apnea-ECG Database | [PhysioNet](https://physionet.org/content/apnea-ecg/1.0.0/) | Détection apnées ECG |

Placer les fichiers `.edf` dans `data/raw/` avant de lancer les notebooks.

---

## ▶️ Installation locale

```bash
git clone https://github.com/marinedde/somnia
cd somnia

# API
pip install -r requirements.txt
uvicorn app.main:app --reload

# Dashboard
pip install -r requirements-dashboard.txt
streamlit run streamlit_app.py
```

---

## Auteure

**Marine Deldicque** — Jedha Bootcamp CDSD/AIA 2026
