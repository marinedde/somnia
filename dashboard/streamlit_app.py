import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Somnia - Analyse Polysomnographique",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un look médical professionnel
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        font-weight: 700;
        padding: 1rem 0;
        border-bottom: 3px solid #3b82f6;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .clinical-note {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# URL de l'API
API_URL = "https://somnia-api.onrender.com"

# Titre principal avec design médical
st.markdown('<h1 class="main-header">🏥 Somnia - Système d\'Analyse Polysomnographique</h1>', unsafe_allow_html=True)
st.markdown("### *Classification Automatique des Stades de Sommeil par Intelligence Artificielle*")
st.markdown("---")

# Sidebar - Configuration et informations
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.markdown("## ⚙️ Configuration")
    
    # Configuration API
    api_url_input = st.text_input(
        "URL de l'API",
        value=API_URL,
        help="URL du serveur FastAPI"
    )
    
    # Mode de démonstration
    demo_mode = st.checkbox("Mode Présentation", value=True, help="Interface simplifiée pour démonstration")
    
    st.markdown("---")
    
    # Informations cliniques
    st.markdown("### 📋 Stades de Sommeil")
    st.markdown("""
    - **😴 Wake (W)** : Éveil
    - **💤 N1** : Sommeil léger (endormissement)
    - **😌 N2** : Sommeil léger consolidé
    - **🌙 N3** : Sommeil profond (lent)
    - **🔮 REM** : Sommeil paradoxal
    """)
    
    st.markdown("---")
    
    # Test de connexion
    st.markdown("### 🔌 État du Système")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Tester", use_container_width=True):
            with st.spinner("Test..."):
                try:
                    response = requests.get(f"{api_url_input}/health", timeout=10)
                    if response.status_code == 200:
                        st.success("✅ Connecté")
                    else:
                        st.error("❌ Erreur")
                except:
                    st.warning("⏳ Réveil en cours...")
    
    with col2:
        # Statut visuel
        try:
            response = requests.get(f"{api_url_input}/health", timeout=5)
            if response.status_code == 200:
                st.markdown("🟢 **En ligne**")
            else:
                st.markdown("🔴 **Hors ligne**")
        except:
            st.markdown("🟡 **En veille**")
    
    st.markdown("---")
    
    # Informations du projet
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    **Développé par :** Marine Deldicque  
    **Institution :** Jedha Bootcamp  
    **Version :** 1.0.0  
    **Date :** Octobre 2025
    """)

# Onglets principaux
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Analyse Clinique",
    "📊 Informations Système",
    "📚 Guide Médical",
    "🎓 Documentation Technique"
])

# ==================== ONGLET 1 : ANALYSE CLINIQUE ====================
with tab1:
    st.markdown("## 🔬 Interface d'Analyse Polysomnographique")
    
    # Informations patient (simulé pour démo)
    if demo_mode:
        with st.expander("👤 Informations Patient (Simulation)", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.text_input("ID Patient", value="DEMO-001", disabled=True)
            col2.text_input("Âge", value="45 ans", disabled=True)
            col3.text_input("Sexe", value="M", disabled=True)
            col4.text_input("Date", value=datetime.now().strftime("%d/%m/%Y"), disabled=True)
    
    st.markdown("---")
    
    # Layout principal en 2 colonnes
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### 📥 Acquisition du Signal")
        
        # Choix du mode d'entrée
        input_mode = st.radio(
            "Source du signal",
            ["🎯 Signal Synthétique (Démo)", "🎲 Signal Aléatoire", "📁 Upload Fichier (Prochainement)"],
            help="Sélectionnez la source du signal EEG à analyser"
        )
        
        if "Synthétique" in input_mode:
            st.markdown('<div class="clinical-note">🎯 <b>Mode Démonstration</b> : Signal EEG simulé pour validation du système</div>', unsafe_allow_html=True)
            
            col_param1, col_param2 = st.columns(2)
            
            with col_param1:
                stage_demo = st.selectbox(
                    "Stade à simuler",
                    ["Wake", "N1", "N2", "N3", "REM"],
                    help="Choisir le type de signal EEG à générer"
                )
            
            with col_param2:
                noise_level = st.slider("Niveau de bruit", 0.0, 0.3, 0.1, 0.05)
            
            # Générer un signal réaliste selon le stade
            sampling_rate = 100
            duration = 30
            t = np.linspace(0, duration, sampling_rate * duration)
            
            # Générer selon le stade choisi
            if stage_demo == "Wake":
                # Éveil : Alpha (8-13 Hz) + Beta (13-30 Hz)
                signal = (
                    0.5 * np.sin(2 * np.pi * 10 * t) +  # Alpha
                    0.3 * np.sin(2 * np.pi * 20 * t) +  # Beta
                    0.2 * np.sin(2 * np.pi * 15 * t) +  # Beta moyen
                    noise_level * np.random.randn(len(t))
                )
            
            elif stage_demo == "N1":
                # N1 : Theta (4-8 Hz) dominant
                signal = (
                    0.6 * np.sin(2 * np.pi * 6 * t) +   # Theta
                    0.2 * np.sin(2 * np.pi * 10 * t) +  # Alpha résiduel
                    0.15 * np.sin(2 * np.pi * 4 * t) +  # Theta lent
                    noise_level * np.random.randn(len(t))
                )
            
            elif stage_demo == "N2":
                # N2 : Theta + fuseaux de sommeil (12-14 Hz)
                base = 0.5 * np.sin(2 * np.pi * 5 * t)
                # Ajouter des fuseaux aléatoires
                for i in range(5):
                    start = np.random.randint(0, len(t)-500)
                    fuseau = np.zeros(len(t))
                    fuseau[start:start+500] = 0.8 * np.sin(2 * np.pi * 13 * t[start:start+500])
                    base += fuseau
                signal = base + noise_level * np.random.randn(len(t))
            
            elif stage_demo == "N3":
                # N3 : Delta (0.5-4 Hz) dominant
                signal = (
                    1.2 * np.sin(2 * np.pi * 2 * t) +    # Delta fort
                    0.3 * np.sin(2 * np.pi * 1 * t) +    # Delta lent
                    0.2 * np.sin(2 * np.pi * 3 * t) +    # Delta rapide
                    noise_level * np.random.randn(len(t))
                )
            
            elif stage_demo == "REM":
                # REM : Mixte rapide, ressemble à l'éveil
                signal = (
                    0.4 * np.sin(2 * np.pi * 8 * t) +    # Theta
                    0.3 * np.sin(2 * np.pi * 15 * t) +   # Beta
                    0.2 * np.sin(2 * np.pi * 25 * t) +   # Gamma
                    0.15 * np.sin(2 * np.pi * 30 * t) +  # Gamma rapide
                    noise_level * np.random.randn(len(t))
                )
            
            st.info(f"🎯 **Signal généré** : {stage_demo} - Le modèle devrait prédire ce stade")
            
        else:  # Signal aléatoire
            st.markdown('<div class="warning-box">🎲 <b>Signal Aléatoire</b> : Pour tests uniquement</div>', unsafe_allow_html=True)
            sampling_rate = 100
            duration = 30
            signal = np.random.randn(sampling_rate * duration) * 0.5
        
        # Visualisation du signal EEG
        st.markdown("### 📈 Tracé EEG (30 secondes)")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        time_axis = np.arange(len(signal)) / sampling_rate
        ax.plot(time_axis, signal, linewidth=0.8, color='#2563eb', alpha=0.8)
        ax.set_xlabel("Temps (secondes)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Amplitude (μV)", fontsize=11, fontweight='bold')
        ax.set_title("Électroencéphalogramme - Canal Unique", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 30])
        
        # Ajouter des marqueurs visuels
        for i in range(0, 31, 5):
            ax.axvline(x=i, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        
        st.pyplot(fig)
        
        # Statistiques du signal
        with st.expander("📊 Statistiques du Signal"):
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("Moyenne", f"{np.mean(signal):.3f} μV")
            stat_col2.metric("Écart-type", f"{np.std(signal):.3f} μV")
            stat_col3.metric("Minimum", f"{np.min(signal):.2f} μV")
            stat_col4.metric("Maximum", f"{np.max(signal):.2f} μV")
    
    with col_right:
        st.markdown("### 🎯 Résultat d'Analyse")
        
        # Bouton de prédiction
        predict_button = st.button(
            "🔬 LANCER L'ANALYSE",
            type="primary",
            use_container_width=True,
            help="Démarrer l'analyse automatique du signal EEG"
        )
        
        if predict_button:
            # Barre de progression
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("⏳ Préparation des données...")
            progress_bar.progress(20)
            time.sleep(0.3)
            
            status_text.text("🧠 Analyse par Intelligence Artificielle...")
            progress_bar.progress(50)
            
            try:
                # Préparer les données
                payload = {"signal": signal.tolist()}
                
                # Appeler l'API
                status_text.text("📡 Communication avec le serveur...")
                progress_bar.progress(70)
                
                response = requests.post(
                    f"{api_url_input}/predict",
                    json=payload,
                    timeout=30
                )
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Afficher le résultat
                    st.markdown('<div class="success-box">✅ <b>Analyse terminée avec succès</b></div>', unsafe_allow_html=True)
                    
                    # Stade prédit
                    predicted_stage = result["predicted_class"]
                    confidence = result["confidence"]
                    
                    # Emoji et description
                    stage_info = {
                        "Wake": ("😴", "Éveil", "#ef4444"),
                        "N1": ("💤", "Sommeil Léger (N1)", "#f59e0b"),
                        "N2": ("😌", "Sommeil Intermédiaire (N2)", "#3b82f6"),
                        "N3": ("🌙", "Sommeil Profond (N3)", "#8b5cf6"),
                        "REM": ("🔮", "Sommeil Paradoxal (REM)", "#ec4899")
                    }
                    
                    emoji, description, color = stage_info.get(predicted_stage, ("🧠", "Inconnu", "#6b7280"))
                    
                    # Grande carte de résultat
                    st.markdown(f"""
                    <div style="
                        text-align: center;
                        padding: 2.5rem;
                        background: linear-gradient(135deg, {color}15 0%, {color}30 100%);
                        border: 3px solid {color};
                        border-radius: 15px;
                        margin: 1.5rem 0;
                        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                    ">
                        <div style="font-size: 5em; margin: 0;">{emoji}</div>
                        <h2 style="color: {color}; margin: 1rem 0;">{description}</h2>
                        <p style="font-size: 1.8em; color: #4b5563; font-weight: 600;">
                            Confiance : {confidence*100:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Comparaison avec le signal attendu
                    if "Synthétique" in input_mode:
                        if predicted_stage == stage_demo:
                            st.success(f"✅ **Prédiction correcte !** Le modèle a bien identifié le stade {stage_demo}")
                        else:
                            st.warning(f"⚠️ **Prédiction incorrecte** : Attendu {stage_demo}, prédit {predicted_stage}")
                    
                    # Interprétation clinique
                    st.markdown("#### 📋 Interprétation Clinique")
                    
                    clinical_notes = {
                        "Wake": "Le patient est en état d'éveil. Activité cérébrale normale pour une personne éveillée.",
                        "N1": "Phase d'endormissement. Sommeil léger caractérisé par une transition entre l'éveil et le sommeil.",
                        "N2": "Sommeil léger consolidé. Représente environ 50% du temps de sommeil total chez l'adulte.",
                        "N3": "Sommeil profond (sommeil lent). Phase récupératrice essentielle pour la régénération physique.",
                        "REM": "Sommeil paradoxal. Phase associée aux rêves et à la consolidation de la mémoire."
                    }
                    
                    st.info(f"💡 **Note clinique :** {clinical_notes.get(predicted_stage, 'N/A')}")
                    
                    # Probabilités détaillées
                    st.markdown("#### 📊 Distribution des Probabilités")
                    
                    probs = result["probabilities"]
                    prob_df = pd.DataFrame({
                        "Stade": list(probs.keys()),
                        "Probabilité": [v * 100 for v in probs.values()]
                    }).sort_values("Probabilité", ascending=True)
                    
                    # Graphique horizontal
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    colors_bar = ['#10b981' if stage == predicted_stage else '#94a3b8' 
                                 for stage in prob_df['Stade']]
                    bars = ax2.barh(prob_df['Stade'], prob_df['Probabilité'], color=colors_bar, edgecolor='black', linewidth=1.5)
                    ax2.set_xlabel('Probabilité (%)', fontsize=11, fontweight='bold')
                    ax2.set_title('Analyse Multi-Classes', fontsize=12, fontweight='bold')
                    ax2.set_xlim([0, 100])
                    ax2.grid(axis='x', alpha=0.3, linestyle='--')
                    
                    # Ajouter les valeurs
                    for i, (bar, v) in enumerate(zip(bars, prob_df['Probabilité'])):
                        ax2.text(v + 2, i, f'{v:.1f}%', va='center', fontweight='bold')
                    
                    st.pyplot(fig2)
                    
                    # Métadonnées
                    with st.expander("🔍 Métadonnées de l'Analyse"):
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        meta_col1.metric("Index Classe", result["predicted_index"])
                        meta_col2.metric("Temps Traitement", "~50 ms")
                        meta_col3.metric("Modèle", "Random Forest v2")
                    
                else:
                    st.error(f"❌ **Erreur API {response.status_code}**")
                    st.code(response.text)
                    st.markdown('<div class="warning-box">⚠️ Le serveur a renvoyé une erreur. Vérifiez le format des données ou réessayez.</div>', unsafe_allow_html=True)
                    
            except requests.exceptions.Timeout:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ **Timeout : Le serveur ne répond pas**")
                st.markdown('<div class="warning-box">💡 Le serveur Render est peut-être en veille. Attendez 30 secondes et réessayez.</div>', unsafe_allow_html=True)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ **Erreur système :** {str(e)}")

# ==================== ONGLET 2 : INFOS SYSTÈME ====================
with tab2:
    st.markdown("## 📊 Informations du Système")
    
    if st.button("🔄 Actualiser les Informations"):
        with st.spinner("Chargement des informations système..."):
            try:
                response = requests.get(f"{api_url_input}/model-info", timeout=10)
                
                if response.status_code == 200:
                    model_info = response.json()
                    
                    # Métriques principales
                    st.markdown("### 🏆 Performances du Modèle")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric(
                        "Accuracy",
                        f"{model_info['accuracy']*100:.1f}%",
                        help="Taux de prédictions correctes"
                    )
                    col2.metric(
                        "F1-Score",
                        f"{model_info['f1_score']:.3f}",
                        help="Moyenne harmonique précision/recall"
                    )
                    col3.metric(
                        "Cohen's Kappa",
                        f"{model_info['cohens_kappa']:.3f}",
                        help="Accord inter-annotateurs"
                    )
                    
                    st.markdown("---")
                    
                    # Informations techniques
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Type de Modèle", model_info["model_type"])
                    col_b.metric("Nombre de Features", model_info["n_features"])
                    col_c.metric("Date d'Entraînement", model_info["training_date"])
                    
                    # Classes
                    st.markdown("### 🏷️ Classes Détectées")
                    st.write(", ".join(model_info["classes"]))
                    
                    # JSON complet
                    with st.expander("🔍 Données Brutes (JSON)"):
                        st.json(model_info)
                        
                else:
                    st.error(f"❌ Erreur {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")

# ==================== ONGLET 3 : GUIDE MÉDICAL ====================
with tab3:
    st.markdown("## 📚 Guide Médical d'Utilisation")
    
    st.markdown("""
    ### 🎯 Objectif Clinique
    
    Somnia est un système d'aide au diagnostic pour l'analyse automatisée des polysomnographies.
    Le système classifie les signaux EEG en 5 stades de sommeil selon la classification AASM.
    
    ### 🏥 Stades de Sommeil (AASM)
    
    #### 😴 Wake (W) - Éveil
    - **Caractéristiques** : Activité alpha (8-13 Hz), mouvements oculaires
    - **Signification** : Patient éveillé, yeux ouverts ou fermés
    
    #### 💤 N1 - Sommeil Léger
    - **Caractéristiques** : Diminution de l'activité alpha, ondes thêta (4-8 Hz)
    - **Durée** : 5-10% du temps de sommeil
    - **Signification** : Phase de transition, sommeil fragile
    
    #### 😌 N2 - Sommeil Intermédiaire
    - **Caractéristiques** : Fuseaux de sommeil, complexes K
    - **Durée** : 45-55% du temps de sommeil
    - **Signification** : Sommeil confirmé mais léger
    
    #### 🌙 N3 - Sommeil Profond
    - **Caractéristiques** : Ondes delta (0.5-4 Hz), >20% du tracé
    - **Durée** : 15-25% du temps de sommeil
    - **Signification** : Sommeil récupérateur, consolidation mnésique
    
    #### 🔮 REM - Sommeil Paradoxal
    - **Caractéristiques** : Activité rapide, mouvements oculaires rapides, atonie musculaire
    - **Durée** : 20-25% du temps de sommeil
    - **Signification** : Phase de rêves, consolidation mémoire émotionnelle
    
    ### ⚠️ Limitations et Précautions
    
    - ✅ **Outil d'aide** : Ne remplace pas l'expertise médicale
    - ✅ **Validation** : Performance actuelle ~65%, nécessite validation clinique
    - ✅ **Usage** : Démo technique, pas de certification médicale
    - ✅ **Contexte** : Entraîné sur dataset Sleep-EDF limité
    
    ### 📖 Références
    
    - AASM Manual for the Scoring of Sleep and Associated Events (2023)
    - Iber C. et al. "The AASM Manual for the Scoring of Sleep"
    - Berry RB. et al. "Rules for scoring respiratory events in sleep"
    """)

# ==================== ONGLET 4 : DOCUMENTATION TECHNIQUE ====================
with tab4:
    st.markdown("## 🎓 Documentation Technique")
    
    st.markdown("""
    ### 🏗️ Architecture du Système
```
    [Dashboard Streamlit] --REST API--> [FastAPI Server] --> [Random Forest Model]
```
    
    ### 🤖 Modèle de Machine Learning
    
    **Type** : Random Forest avec Feature Engineering
    
    **Features Extraites (16)** :
    - **Temporelles (8)** : Mean, Std, Min, Max, Q1, Q3, Skewness, Kurtosis
    - **Fréquentielles (5)** : Puissance Delta, Theta, Alpha, Beta, Gamma
    - **Ratios (3)** : Ratios de puissance normalisés
    
    **Performance** :
    - Accuracy : 65%
    - F1-Score : 0.64
    - Cohen's Kappa : 0.52
    
    ### 🔗 Endpoints API
    
    #### GET /health
    Vérifie l'état du serveur
    
    #### GET /model-info
    Retourne les informations du modèle
    
    #### POST /predict
    Prédit le stade de sommeil
    
    **Input** : `{"signal": [3000 valeurs]}`
    
    **Output** : `{"predicted_class": "N2", "confidence": 0.65, ...}`
    
    ### 📦 Technologies Utilisées
    
    - **Backend** : FastAPI, Uvicorn
    - **ML** : Scikit-learn, NumPy, SciPy
    - **Frontend** : Streamlit
    - **Déploiement** : Docker, Render
    
    ### 🔮 Perspectives d'Évolution
    
    #### Court terme
    - Amélioration de l'accuracy (objectif >90%)
    - Support de formats EDF
    - Export de rapports PDF
    
    #### Moyen terme
    - Détection d'apnées du sommeil
    - Interface multi-patients
    - Authentification sécurisée
    
    #### Long terme
    - Certification dispositif médical (CE)
    - Hébergement données de santé (HDS)
    - Intégration FHIR
    
    ### 👥 Crédits
    
    **Développeur** : Marine Deldicque  
    **Institution** : Jedha Bootcamp  
    **Dataset** : Sleep-EDF Database (Physionet)  
    **Date** : Octobre 2025
    
    ### 🔗 Liens Utiles
    
    - [GitHub Repository](https://github.com/marinedde/somnia)
    - [API Documentation](https://somnia-api.onrender.com/docs)
    - [Dataset Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem; background-color: #f9fafb; border-radius: 8px;">
    <p style="font-size: 0.9em; margin: 0;">
        🏥 <b>Somnia</b> - Système d'Analyse Polysomnographique par Intelligence Artificielle
    </p>
    <p style="font-size: 0.8em; margin: 0.5rem 0 0 0;">
        Développé par <b>Marine Deldicque</b> | Jedha Bootcamp 2025 | Version 1.0.0
    </p>
    <p style="font-size: 0.75em; color: #9ca3af; margin: 0.5rem 0 0 0;">
        ⚠️ Outil de démonstration - Ne pas utiliser à des fins diagnostiques
    </p>
</div>
""", unsafe_allow_html=True)