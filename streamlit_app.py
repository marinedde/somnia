"""
Somnia — Dashboard Streamlit
Fil rouge : 4h d'analyse manuelle → quelques secondes
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
import time
import json
from pathlib import Path
from scipy.signal import welch

# ─── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Somnia — Analyse du Sommeil et détection de l'apnéé du sommeil",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL  = "https://marinedde-somnia-api.hf.space"
DEMO_DIR = Path("data/demo")

STAGE_COLORS = {
    'Wake': '#E84855', 'N1': '#EF8354',
    'N2':  '#54C6EB',  'N3': '#048A81', 'REM': '#1B2A4A',
}

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #1B2A4A;
    margin: 0.4rem 0 0.6rem 0;
    letter-spacing: -1px;
}
.hero .tagline {
    font-style: italic;
    font-size: 1.15rem;
    color: #6C757D;
    margin-bottom: 1rem;
}
.hero .description {
    max-width: 720px;
    margin: 0 auto;
    font-size: 1rem;
    color: #495057;
    line-height: 1.7;
}

/* ── Feature cards (accueil) ── */
.feat-card {
    border-left: 4px solid;
    padding: 1.2rem 1.4rem;
    border-radius: 0 10px 10px 0;
    background: #FAFAFA;
    height: 100%;
}
.feat-card h4 {
    font-size: 1rem;
    font-weight: 600;
    color: #1B2A4A;
    margin: 0 0 0.5rem 0;
}
.feat-card p {
    font-size: 0.88rem;
    color: #6C757D;
    margin: 0;
    line-height: 1.6;
}

/* ── Section title ── */
.stitle {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #1B2A4A;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #E9ECEF;
}

/* ── Result card ── */
.rcard {
    border-radius: 12px;
    padding: 1.6rem;
    margin: 0.8rem 0;
    border-left: 5px solid;
}
.rcard-ok  { background:#E8F5F4; border-color:#048A81; }
.rcard-bad { background:#FDECEA; border-color:#E84855; }
.rcard-neu { background:#F0F4FF; border-color:#1B2A4A; }

/* ── Prob bars ── */
.pb-wrap  { margin: 0.35rem 0; }
.pb-head  { display:flex; justify-content:space-between; font-size:0.82rem; color:#495057; margin-bottom:2px; }
.pb-track { height:8px; border-radius:4px; background:#E9ECEF; overflow:hidden; }
.pb-fill  { height:100%; border-radius:4px; }

/* ── Time badge ── */
.tbadge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(239,131,84,0.12); border:1px solid #EF8354;
    color:#C8501A; padding:4px 14px; border-radius:50px;
    font-size:0.85rem; font-weight:600;
}

/* ── Medical alert ── */
.med-alert {
    background:#FFF8E1; border:1px solid #FFD54F;
    border-radius:10px; padding:0.9rem 1.1rem;
    font-size:0.84rem; color:#6D4C00; margin-top:1.5rem;
}

/* ── Rapport box ── */
.rapport-box {
    background:linear-gradient(135deg,#F0F7FF,#E8F5F4);
    border:1px solid #B3D9FF; border-radius:14px;
    padding:1.6rem; font-size:0.92rem;
    line-height:1.75; color:#1B2A4A;
}

/* ── Validation box ── */
.val-box {
    background:#F8F9FA; border:2px dashed #CED4DA;
    border-radius:12px; padding:1.2rem; margin-top:1rem;
}

/* Sidebar cleanup */
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
.stDeployButton{display:none;}
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except:
        return False, {}


def load_demo(name):
    """Charge un signal démo depuis data/demo/."""
    p = DEMO_DIR / f"{name}.npy"
    return np.load(p) if p.exists() else None


def call_api(endpoint, signal):
    try:
        r = requests.post(f"{API_URL}{endpoint}",
                          json={"signal": signal.tolist()}, timeout=30)
        return r.status_code == 200, r.json()
    except Exception as e:
        return False, {"error": str(e)}


def save_validation(task, result, verdict, comment, extra=None):
    """Persiste la validation clinique via l'API (audit trail)."""
    verdict_map = {
        "✅ Confirmé": "Confirmé",
        "❌ Incorrect": "Incorrect",
        "⚠️ Ambigu": "Ambigu",
    }
    payload = {
        "task": task,
        "model_prediction": result["predicted_class"],
        "model_confidence": result["confidence"],
        "clinician_verdict": verdict_map.get(verdict, verdict),
        "comment": comment or "",
        "extra": extra or {},
    }
    try:
        r = requests.post(f"{API_URL}/validations", json=payload, timeout=10)
        if r.status_code == 200:
            return True, r.json().get("id", "")
        return False, r.text
    except Exception as e:
        return False, str(e)


def pb_html(probs, cmap):
    html = ""
    for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        c = cmap.get(label, '#6C757D')
        pct = prob * 100
        html += f"""<div class="pb-wrap">
            <div class="pb-head"><span>{label}</span><span><b>{pct:.1f}%</b></span></div>
            <div class="pb-track"><div class="pb-fill" style="width:{pct}%;background:{c};"></div></div>
        </div>"""
    return html


def plot_eeg(signal, title="Signal EEG", fs=100):
    fig = plt.figure(figsize=(13, 6), facecolor='white')
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.3)
    t   = np.arange(len(signal)) / fs

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, signal * 1e6, color='#1B2A4A', lw=0.8, alpha=0.9)
    ax1.set(xlabel='Temps (s)', ylabel='Amplitude (µV)', title=title)
    ax1.title.set_fontsize(11); ax1.title.set_fontweight('bold')
    for s in ['top','right']: ax1.spines[s].set_visible(False)
    ax1.grid(axis='x', alpha=0.2)

    sig_uv = signal * 1e6
    freqs, psd = welch(sig_uv, fs=fs, nperseg=256)
    mask = freqs <= 35

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(freqs[mask], psd[mask], alpha=0.25, color='#048A81')
    ax2.plot(freqs[mask], psd[mask], color='#048A81', lw=1.5)
    band_defs = [('δ',(0.5,4),'#E84855'),('θ',(4,8),'#EF8354'),('α',(8,13),'#54C6EB'),('β',(13,30),'#1B2A4A')]
    for nm,(f1,f2),c in band_defs:
        m = (freqs>=f1)&(freqs<f2)&mask
        ax2.fill_between(freqs[m], psd[m], alpha=0.3, color=c, label=f'{nm} {f1}-{f2}Hz')
    ax2.set(xlabel='Fréquence (Hz)', ylabel='Puissance (µV²/Hz)')
    ax2.set_title('Densité spectrale', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7)
    for s in ['top','right']: ax2.spines[s].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1])
    bands = [('δ Delta\n0.5-4Hz',(0.5,4),'#E84855'),('θ Theta\n4-8Hz',(4,8),'#EF8354'),
             ('α Alpha\n8-13Hz',(8,13),'#54C6EB'),('β Beta\n13-30Hz',(13,30),'#1B2A4A'),
             ('γ Gamma\n30-35Hz',(30,35),'#048A81')]
    names = [b[0] for b in bands]
    powers = [np.mean(psd[(freqs>=b[1][0])&(freqs<b[1][1])]) if ((freqs>=b[1][0])&(freqs<b[1][1])).any() else 0 for b in bands]
    ax3.bar(names, powers, color=[b[2] for b in bands], alpha=0.85, edgecolor='white')
    ax3.set(ylabel='Puissance (µV²/Hz)')
    ax3.set_title('Puissance par bande', fontsize=10, fontweight='bold')
    ax3.tick_params(axis='x', labelsize=7)
    for s in ['top','right']: ax3.spines[s].set_visible(False)
    return fig


def plot_ecg(signal, prediction=None, fs=100):
    color = '#E84855' if prediction == 'Apnée' else '#048A81'
    fig   = plt.figure(figsize=(13, 5), facecolor='white')
    gs    = gridspec.GridSpec(1, 3, wspace=0.35)
    t     = np.arange(len(signal)) / fs

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t, signal, color=color, lw=0.7, alpha=0.9)
    ax1.set(xlabel='Temps (s)', ylabel='Amplitude (mV)',
            title=f"Signal ECG — {prediction or 'Analyse'}")
    ax1.title.set_fontsize(11); ax1.title.set_fontweight('bold')
    for s in ['top','right']: ax1.spines[s].set_visible(False)
    ax1.grid(axis='x', alpha=0.2)

    freqs, psd = welch(signal, fs=fs, nperseg=512)
    mask = freqs <= 40
    ax2  = fig.add_subplot(gs[0, 2])
    ax2.fill_between(freqs[mask], psd[mask], alpha=0.25, color=color)
    ax2.plot(freqs[mask], psd[mask], color=color, lw=1.5)
    ax2.set(xlabel='Fréquence (Hz)', ylabel='Puissance (mV²/Hz)')
    ax2.set_title('Spectre ECG', fontsize=10, fontweight='bold')
    for s in ['top','right']: ax2.spines[s].set_visible(False)
    return fig


def generate_report(eeg_result, ecg_result, patient_info):
    """Génère le rapport via Claude API ou en fallback structuré."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    parts = []
    if patient_info:
        parts.append(f"Patient : {patient_info.get('age')} ans, "
                     f"sexe {patient_info.get('sexe')}, IMC {patient_info.get('imc')}")
    if eeg_result:
        parts.append(f"EEG : stade {eeg_result['predicted_class']} "
                     f"({eeg_result['confidence']*100:.1f}% confiance). "
                     f"{eeg_result.get('interpretation','')}")
    if ecg_result:
        parts.append(f"ECG : {ecg_result['predicted_class']} "
                     f"({ecg_result['confidence']*100:.1f}% confiance), "
                     f"risque {ecg_result.get('risk_level','')}. "
                     f"{ecg_result.get('recommendation','')}")

    prompt = f"""Tu es un assistant de synthèse pour médecins du sommeil (aide à la décision uniquement).
Génère un brouillon structuré en français à partir des sorties algorithmiques :

{chr(10).join(parts)}

RÈGLES STRICTES :
- Ne pose PAS de diagnostic définitif.
- Ne prescris AUCUN médicament ni traitement.
- Formule des hypothèses et points à vérifier par le médecin.
- Rappelle que l'outil ne remplace pas la polysomnographie.

Structure :
1. **Résumé** (2-3 phrases, factuel)
2. **Analyse EEG** : stade prédit et limites
3. **Analyse ECG** : signal algorithmique (pas un AHI clinique)
4. **Points de vigilance** pour le spécialiste
5. **Limites de l'outil**

Maximum 350 mots."""

    if api_key:
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type":"application/json",
                         "x-api-key": api_key,
                         "anthropic-version":"2023-06-01"},
                json={"model":"claude-sonnet-4-20250514","max_tokens":1000,
                      "messages":[{"role":"user","content":prompt}]},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()['content'][0]['text'], 'claude'
        except:
            pass

    # Fallback structuré
    age  = patient_info.get('age','N/A')
    sexe = patient_info.get('sexe','N/A')
    imc  = patient_info.get('imc','N/A')
    lines = [
        "**RAPPORT D'ANALYSE SOMMEIL — Somnia**",
        f"*Patient : {age} ans, {sexe}, IMC {imc}*", "",
        "**1. Résumé exécutif**",
    ]
    if eeg_result and ecg_result:
        lines.append(f"L'analyse révèle un stade {eeg_result['predicted_class']} "
                     f"sur l'EEG et {ecg_result['predicted_class']} sur l'ECG.")
    elif eeg_result:
        lines.append(f"L'analyse EEG révèle un stade {eeg_result['predicted_class']}.")
    elif ecg_result:
        lines.append(f"L'analyse ECG détecte : {ecg_result['predicted_class']}.")

    if eeg_result:
        lines += ["", "**2. Analyse EEG**",
                  f"Stade {eeg_result['predicted_class']} détecté "
                  f"({eeg_result['confidence']*100:.1f}% de confiance). "
                  f"{eeg_result.get('interpretation','')}"]
    if ecg_result:
        lines += ["", "**3. Analyse ECG**",
                  f"{ecg_result['predicted_class']} — Risque {ecg_result.get('risk_level','')}. "
                  f"{ecg_result.get('recommendation','')}"]
    lines += ["", "**4. Recommandations**",
              "- Validation par un spécialiste du sommeil requise.",
              "- Corrélation avec les symptômes cliniques (somnolence, ronflement).",]
    if ecg_result and ecg_result.get('predicted_class') == 'Apnée':
        lines.append("- Polysomnographie complète recommandée pour évaluer l'AHI.")
    lines += ["", "**5. Limites**",
              "Ce rapport est généré automatiquement. "
              "Il ne constitue pas un diagnostic médical."]
    return "\n".join(lines), 'fallback'


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.5rem 0 1.2rem 0;'>
        <div style='font-size:2.2rem; margin-bottom:4px;'>🌙</div>
        <div style='font-family:"DM Serif Display",serif; font-size:1.6rem;
                    color:#1B2A4A; font-weight:400; line-height:1;'>Somnia</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Navigation**")
    page = st.radio("", [
        "🏠 Accueil",
        "🧠 Analyse EEG",
        "❤️ Analyse ECG",
        "📊 Rapport complet",
        "📈 Monitoring",
    ], label_visibility="collapsed")

    st.divider()
    api_ok, health = api_health()
    if api_ok:
        eeg_ok_srv = health.get('eeg_model_loaded', False)
        ecg_ok_srv = health.get('ecg_model_loaded', False)
        status_txt = "API connectée" if (eeg_ok_srv and ecg_ok_srv) else "API — modèles partiels"
        st.success(f"🟢 {status_txt}")
    else:
        st.error("🔴 API non disponible")
        st.caption("`uvicorn app.main:app --reload`")

    st.divider()
    st.markdown("**Patient**")
    p_age  = st.number_input("Âge", 18, 100, 55)
    p_sexe = st.selectbox("Sexe", ["M","F"])
    p_imc  = st.number_input("IMC", 15.0, 50.0, 27.5, step=0.1)
    patient_info = {"age":p_age, "sexe":p_sexe, "imc":p_imc}

    st.divider()
    st.caption("Marine Deldicque — AIA Jedha 2026")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE ACCUEIL
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":

    st.markdown("""
    <div class="hero">
        <div style='font-size:2.8rem;'>🌙</div>
        <h1>Somnia</h1>
        <div class="description">
            Somnia analyse automatiquement les signaux physiologiques nocturnes — EEG et ECG —
            pour classifier les stades de sommeil et détecter l'apnée du sommeil.
            L'objectif : rendre l'analyse polysomnographique accessible et instantanée,
            que vous soyez clinicien, chercheur ou en formation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # 4 feature cards
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("🧠 Analyse EEG", "#048A81",
         "Classifiez un signal EEG en 5 stades de sommeil (Wake, N1, N2, N3, REM) "
         "avec probabilités et interprétation clinique."),
        ("❤️ Analyse ECG", "#E84855",
         "Détectez l'apnée du sommeil minute par minute sur un signal ECG. "
         "Niveau de risque, recommandation clinique, validation médecin."),
        ("📊 Rapport complet", "#EF8354",
         "Synthèse EEG + ECG avec rapport clinique généré automatiquement par IA "
         "(Claude Anthropic). Export PDF et JSON."),
        ("⚡ Gain de temps", "#1B2A4A",
         "4 heures d'analyse manuelle réduites à quelques secondes. "
         "×8000 plus rapide qu'un technicien polysomnographiste."),
    ]
    for col, (title, color, desc) in zip([c1,c2,c3,c4], cards):
        col.markdown(f"""
        <div class="feat-card" style="border-color:{color};">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Performances
    st.markdown('<div class="stitle">Performances</div>', unsafe_allow_html=True)
    m1,m2,m3,m4,m5 = st.columns(5)
    for col, (label, val, delta) in zip([m1,m2,m3,m4,m5],[
        ("Précision EEG",     "83.4%",  "5 stades"),
        ("F1-score EEG",      "0.830",  "weighted"),
        ("AUC-ROC ECG",       "0.967",  "split aléatoire"),
        ("F1 Apnée",          "0.879",  "classe apnée"),
        ("Temps d'analyse",   "< 2s",   "vs 4h manuelles ⚡"),
    ]):
        col.metric(label, val, delta)

    # Contexte
    st.markdown('<div class="stitle">Contexte clinique</div>', unsafe_allow_html=True)
    cl, cr = st.columns([3,2])
    with cl:
        st.markdown("""
**L'apnée du sommeil touche 1 milliard de personnes** dans le monde et reste largement
sous-diagnostiquée. Une polysomnographie standard nécessite une nuit d'enregistrement
et **4 à 6 heures d'analyse manuelle** par un technicien qualifié.

Somnia combine deux analyses complémentaires :

- **EEG (Electroencéphalogramme)** : caractérise l'architecture du sommeil via les
  ondes cérébrales. Le modèle Random Forest atteint 83% de précision sur 5 stades
  (Sleep-EDF, 28 sujets, PhysioNet).

- **ECG (Electrocardiogramme)** : détecte les événements apnéiques via la variabilité
  cardiaque (HRV), les intervalles RR et la morphologie QRS. AUC-ROC 0.967
  (Apnea-ECG, 35 sujets, PhysioNet).

⚠️ Somnia est un outil d'aide à la décision. Le diagnostic reste de la responsabilité du médecin.
        """)
    with cr:
        fig_t, ax_t = plt.subplots(figsize=(5,3.5), facecolor='white')
        ax_t.barh(['Analyse\nmanuelle','Somnia'], [240, 0.033],
                   color=['#DEE2E6','#048A81'], height=0.45, edgecolor='none')
        ax_t.bar_label(ax_t.containers[0], labels=['4 heures','< 2 secondes'],
                       padding=6, fontsize=10, fontweight='bold', color='#495057')
        ax_t.set_xlim(0, 290)
        ax_t.set_title("Temps d'analyse comparé", fontsize=11,
                       fontweight='bold', color='#1B2A4A', pad=12)
        for s in ['top','right','bottom']: ax_t.spines[s].set_visible(False)
        ax_t.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_t.set_facecolor('white'); fig_t.patch.set_facecolor('white')
        st.pyplot(fig_t, use_container_width=True); plt.close()

        st.markdown("""
        <div style='background:#E8F5F4;border-radius:10px;padding:1rem;
                    text-align:center;margin-top:0.5rem;'>
            <div style='font-size:2.4rem;font-weight:700;color:#048A81;
                        font-family:"DM Serif Display",serif;'>×8 000</div>
            <div style='font-size:0.85rem;color:#1B2A4A;margin-top:2px;'>
                plus rapide qu'une analyse manuelle
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="med-alert">
        ⚠️ <b>Avertissement médical :</b> Somnia est un dispositif d'aide à la décision clinique.
        Il ne se substitue pas au diagnostic médical. Toute décision thérapeutique doit être
        validée par un médecin qualifié. Les performances sont issues de datasets de recherche
        et peuvent varier en conditions cliniques réelles.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE ANALYSE EEG
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧠 Analyse EEG":

    st.markdown("""
    <div style='padding:0.5rem 0 1rem 0;'>
        <h2 style='font-family:"DM Serif Display",serif;font-size:2rem;
                   color:#1B2A4A;margin:0 0 0.3rem 0;'>🧠 Analyse EEG</h2>
        <div style='color:#6C757D;font-size:0.95rem;'>
            Classification automatique des stades de sommeil
            &nbsp;·&nbsp; <span class="tbadge">⚡ &lt; 1 seconde vs 4h manuelles</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_out = st.columns([1,1], gap="large")

    with col_in:
        st.markdown('<div class="stitle">Signal EEG</div>', unsafe_allow_html=True)
        mode = st.radio("Source", ["🎲 Signal démo","📁 Importer .npy"], horizontal=True)

        eeg_signal = None
        if mode == "🎲 Signal démo":
            # Vrais signaux depuis data/demo/
            demo_options = {"N3 — Sommeil profond":"demo_eeg_n3",
                            "N2 — Sommeil léger": "demo_eeg_n2",
                            "REM — Sommeil paradoxal":"demo_eeg_rem",
                            "Wake — Éveil":"demo_eeg_wake"}
            demo_choice = st.selectbox("Exemple de signal", list(demo_options.keys()))
            sig = load_demo(demo_options[demo_choice])
            if sig is not None:
                eeg_signal = sig
                st.info(f"Signal EEG réel extrait de Sleep-EDF — {demo_choice} (3000 points, 100Hz)")
            else:
                st.warning("Fichier démo introuvable — lancez d'abord `scripts/extract_demo_signals.py`")
        else:
            up = st.file_uploader("Fichier .npy (3000 valeurs, 100Hz, 30s)", type=['npy'])
            if up:
                arr = np.load(up)
                if len(arr) == 3000:
                    eeg_signal = arr
                    st.success(f"Signal chargé — {len(arr)} points")
                else:
                    st.error(f"3000 points attendus, reçu {len(arr)}")

        if eeg_signal is not None:
            fig_p, ax_p = plt.subplots(figsize=(7,2.2), facecolor='white')
            ax_p.plot(np.arange(len(eeg_signal))/100, eeg_signal*1e6,
                      color='#1B2A4A', lw=0.8)
            ax_p.set(xlabel='Temps (s)', ylabel='µV')
            ax_p.set_title('Aperçu', fontsize=9)
            for s in ['top','right']: ax_p.spines[s].set_visible(False)
            st.pyplot(fig_p, use_container_width=True); plt.close()

            if st.button("🔍 Analyser", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    t0 = time.time()
                    ok, res = call_api("/predict/sleep-stage", eeg_signal)
                    ela = time.time() - t0
                if ok:
                    st.session_state.update(
                        eeg_result=res, eeg_signal=eeg_signal, eeg_elapsed=ela)
                    st.success(f"✅ Terminé en {ela:.3f}s")
                else:
                    st.error(f"Erreur : {res}")

    with col_out:
        st.markdown('<div class="stitle">Résultat</div>', unsafe_allow_html=True)

        if 'eeg_result' in st.session_state:
            r    = st.session_state['eeg_result']
            pred = r['predicted_class']
            conf = r['confidence']
            ela  = st.session_state.get('eeg_elapsed', 0)
            col  = STAGE_COLORS.get(pred, '#1B2A4A')

            st.markdown(f"""
            <div class="rcard rcard-ok" style="border-color:{col};">
                <div style='font-size:0.72rem;text-transform:uppercase;
                            letter-spacing:1.2px;color:#6C757D;'>STADE DÉTECTÉ</div>
                <div style='font-family:"DM Serif Display",serif;font-size:2.8rem;
                            color:{col};line-height:1.1;margin:4px 0;'>{pred}</div>
                <div style='font-size:0.9rem;color:#495057;'>{r.get('interpretation','')}</div>
                <div style='margin-top:0.8rem;font-size:0.82rem;color:#6C757D;'>
                    Confiance&nbsp;<b style='color:{col}'>{conf*100:.1f}%</b>
                    &nbsp;·&nbsp; Temps&nbsp;<b>{ela:.3f}s</b>
                    &nbsp;·&nbsp; <span style='color:#C8501A;font-weight:600;'>vs ~4h manuellement</span>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("**Probabilités par stade**")
            st.markdown(pb_html(r['probabilities'], STAGE_COLORS), unsafe_allow_html=True)

            with st.expander("📈 Analyse spectrale complète", expanded=True):
                fig_eeg = plot_eeg(st.session_state['eeg_signal'],
                                   title=f"EEG — Stade {pred} détecté")
                st.pyplot(fig_eeg, use_container_width=True); plt.close()

            st.markdown('<div class="stitle">Validation clinique</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="val-box">
                <b>🩺 Validation médecin requise</b>
                <p style='font-size:0.84rem;color:#6C757D;margin:0.4rem 0 0 0;'>
                Le résultat automatique doit être confirmé par un expert clinique
                avant toute décision thérapeutique.
                </p>
            </div>""", unsafe_allow_html=True)

            val = st.radio("Évaluation",
                           ["✅ Confirmé","❌ Incorrect","⚠️ Ambigu"],
                           horizontal=True, key="val_eeg")
            st.text_area("Commentaire", key="com_eeg", height=68,
                         placeholder="Qualité du signal, artéfacts observés...")
            if st.button("Enregistrer validation EEG"):
                ok_v, vid = save_validation(
                    "sleep_stage", r, val, st.session_state.get("com_eeg", "")
                )
                if ok_v:
                    st.success(f"Validation enregistrée ({val}) — audit #{vid[:8]}…")
                else:
                    st.warning(f"Validation locale seulement (API : {vid})")
        else:
            st.markdown("""
            <div style='background:#F8F9FA;border-radius:12px;padding:3rem;
                        text-align:center;color:#ADB5BD;'>
                <div style='font-size:3rem;'>🧠</div>
                <div style='margin-top:0.5rem;font-size:0.95rem;'>
                    Sélectionnez un signal et cliquez sur Analyser
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE ANALYSE ECG
# ═══════════════════════════════════════════════════════════════════════════
elif page == "❤️ Analyse ECG":

    st.markdown("""
    <div style='padding:0.5rem 0 1rem 0;'>
        <h2 style='font-family:"DM Serif Display",serif;font-size:2rem;
                   color:#1B2A4A;margin:0 0 0.3rem 0;'>❤️ Analyse ECG</h2>
        <div style='color:#6C757D;font-size:0.95rem;'>
            Détection automatique de l'apnée du sommeil
            &nbsp;·&nbsp; <span class="tbadge">⚡ &lt; 1 seconde vs 4h manuelles</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_out = st.columns([1,1], gap="large")

    with col_in:
        st.markdown('<div class="stitle">Signal ECG</div>', unsafe_allow_html=True)
        mode_ecg = st.radio("Source",
                            ["🎲 Démo — Signal normal","🎲 Démo — Apnée réelle","📁 Importer .npy"],
                            horizontal=False)

        ecg_signal = None
        if mode_ecg == "🎲 Démo — Signal normal":
            sig = load_demo("demo_ecg_normal")
            if sig is not None:
                ecg_signal = sig
                st.info("Signal ECG réel — sujet c01 (normal, 6000 points, 100Hz)")
            else:
                st.warning("Fichier démo introuvable — lancez `scripts/extract_demo_signals.py`")

        elif mode_ecg == "🎲 Démo — Apnée réelle":
            sig = load_demo("demo_ecg_apnea")
            if sig is not None:
                ecg_signal = sig
                st.warning("Signal ECG réel — sujet a01 (apnée sévère, 6000 points, 100Hz)")
            else:
                st.warning("Fichier démo introuvable — lancez `scripts/extract_demo_signals.py`")

        else:
            up_ecg = st.file_uploader("Fichier .npy (6000 valeurs, 100Hz, 60s)",
                                       type=['npy'], key="ecg_up")
            if up_ecg:
                arr = np.load(up_ecg)
                if len(arr) == 6000:
                    ecg_signal = arr
                    st.success(f"Signal chargé — {len(arr)} points")
                else:
                    st.error(f"6000 points attendus, reçu {len(arr)}")

        if ecg_signal is not None:
            fig_p2, ax_p2 = plt.subplots(figsize=(7,2.2), facecolor='white')
            ax_p2.plot(np.arange(len(ecg_signal))/100, ecg_signal,
                       color='#E84855', lw=0.6)
            ax_p2.set(xlabel='Temps (s)', ylabel='mV')
            ax_p2.set_title('Aperçu', fontsize=9)
            for s in ['top','right']: ax_p2.spines[s].set_visible(False)
            st.pyplot(fig_p2, use_container_width=True); plt.close()

            if st.button("🔍 Analyser", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    t0 = time.time()
                    ok, res = call_api("/predict/apnea", ecg_signal)
                    ela = time.time() - t0
                if ok:
                    st.session_state.update(
                        ecg_result=res, ecg_signal=ecg_signal, ecg_elapsed=ela)
                    st.success(f"✅ Terminé en {ela:.3f}s")
                else:
                    st.error(f"Erreur : {res}")

    with col_out:
        st.markdown('<div class="stitle">Résultat</div>', unsafe_allow_html=True)

        if 'ecg_result' in st.session_state:
            r    = st.session_state['ecg_result']
            pred = r['predicted_class']
            conf = r['confidence']
            risk = r.get('risk_level','')
            reco = r.get('recommendation','')
            ela  = st.session_state.get('ecg_elapsed',0)
            rc   = {'Faible':'#048A81','Modéré':'#EF8354','Élevé':'#E84855'}.get(risk,'#6C757D')
            pc   = '#E84855' if pred=='Apnée' else '#048A81'
            cc   = 'rcard-bad' if pred=='Apnée' else 'rcard-ok'

            st.markdown(f"""
            <div class="rcard {cc}" style="border-color:{pc};">
                <div style='font-size:0.72rem;text-transform:uppercase;
                            letter-spacing:1.2px;color:#6C757D;'>RÉSULTAT DÉTECTION</div>
                <div style='font-family:"DM Serif Display",serif;font-size:2.8rem;
                            color:{pc};line-height:1.1;margin:4px 0;'>{pred}</div>
                <div style='display:flex;align-items:center;gap:10px;margin:6px 0;'>
                    <span style='background:{rc}18;color:{rc};padding:3px 12px;
                                 border-radius:20px;font-size:0.82rem;font-weight:600;'>
                        Risque {risk}
                    </span>
                    <span style='font-size:0.82rem;color:#6C757D;'>
                        Confiance&nbsp;<b style='color:{pc}'>{conf*100:.1f}%</b>
                    </span>
                </div>
                <div style='font-size:0.9rem;color:#495057;'>{reco}</div>
                <div style='margin-top:0.8rem;font-size:0.82rem;color:#6C757D;'>
                    Temps&nbsp;<b>{ela:.3f}s</b>
                    &nbsp;·&nbsp;<span style='color:#C8501A;font-weight:600;'>vs ~4h manuellement</span>
                </div>
            </div>""", unsafe_allow_html=True)

            if pred == 'Apnée' and risk == 'Élevé':
                st.error("🚨 **Apnée sévère suspectée** — Consultation spécialisée et "
                         "polysomnographie recommandées pour évaluer l'AHI.")

            st.markdown("**Probabilités**")
            st.markdown(pb_html(r['probabilities'],
                                {'Normal':'#048A81','Apnée':'#E84855'}),
                        unsafe_allow_html=True)

            with st.expander("📈 Analyse du signal ECG", expanded=True):
                fig_ecg = plot_ecg(st.session_state['ecg_signal'], prediction=pred)
                st.pyplot(fig_ecg, use_container_width=True); plt.close()

            st.markdown('<div class="stitle">Validation clinique</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="val-box">
                <b>🩺 Validation médecin requise</b>
                <p style='font-size:0.84rem;color:#6C757D;margin:0.4rem 0 0 0;'>
                Confronter avec la clinique : ronflement, somnolence diurne,
                SpO₂ nocturne, IMC, antécédents cardiovasculaires.
                </p>
            </div>""", unsafe_allow_html=True)

            val_ecg = st.radio("Évaluation",
                               ["✅ Confirmé","❌ Incorrect","⚠️ Ambigu"],
                               horizontal=True, key="val_ecg")

            with st.expander("📋 Données cliniques complémentaires"):
                ca,cb = st.columns(2)
                spo2 = ca.number_input("SpO₂ min nocturne (%)", 60, 100, 88, key="spo2_clin")
                ahi  = ca.number_input("AHI PSG (si dispo)", 0.0, 150.0, 0.0, key="ahi_clin")
                snore = cb.checkbox("Ronflement signalé", key="snore_clin")
                somn = cb.selectbox("Somnolence diurne",
                             ["Absente","Légère","Modérée","Sévère"], key="somn_clin")

            st.text_area("Commentaire", key="com_ecg", height=68,
                         placeholder="IMC élevé, ronflement confirmé par conjoint...")
            if st.button("Enregistrer validation ECG"):
                extra = {
                    "spo2_min": st.session_state.get("spo2_clin"),
                    "ahi_psg": st.session_state.get("ahi_clin"),
                    "ronflement": st.session_state.get("snore_clin"),
                    "somnolence": st.session_state.get("somn_clin"),
                }
                ok_v, vid = save_validation(
                    "apnea", r, val_ecg, st.session_state.get("com_ecg", ""), extra
                )
                if ok_v:
                    st.success(f"Validation enregistrée ({val_ecg}) — audit #{vid[:8]}…")
                else:
                    st.warning(f"Validation locale seulement (API : {vid})")

        else:
            st.markdown("""
            <div style='background:#F8F9FA;border-radius:12px;padding:3rem;
                        text-align:center;color:#ADB5BD;'>
                <div style='font-size:3rem;'>❤️</div>
                <div style='margin-top:0.5rem;font-size:0.95rem;'>
                    Sélectionnez un signal et cliquez sur Analyser
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE RAPPORT COMPLET
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Rapport complet":

    st.markdown("""
    <div style='padding:0.5rem 0 1rem 0;'>
        <h2 style='font-family:"DM Serif Display",serif;font-size:2rem;
                   color:#1B2A4A;margin:0 0 0.3rem 0;'>📊 Rapport clinique complet</h2>
        <div style='color:#6C757D;font-size:0.95rem;'>
            Synthèse EEG + ECG avec rapport généré par IA
            &nbsp;·&nbsp; <span class="tbadge">⚡ Rapport complet en &lt; 5 secondes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    eeg_has = 'eeg_result' in st.session_state
    ecg_has = 'ecg_result' in st.session_state

    c1,c2 = st.columns(2)
    with c1:
        if eeg_has:
            r = st.session_state['eeg_result']
            st.success(f"✅ EEG — {r['predicted_class']} ({r['confidence']*100:.1f}%)")
        else:
            st.warning("⚠️ Analyse EEG manquante")
    with c2:
        if ecg_has:
            r = st.session_state['ecg_result']
            st.success(f"✅ ECG — {r['predicted_class']} ({r['confidence']*100:.1f}%)")
        else:
            st.warning("⚠️ Analyse ECG manquante")

    if eeg_has or ecg_has:
        st.markdown('<div class="stitle">Synthèse</div>', unsafe_allow_html=True)
        rows = []
        if eeg_has:
            r = st.session_state['eeg_result']
            rows.append({'Analyse':'🧠 EEG','Résultat':r['predicted_class'],
                         'Confiance':f"{r['confidence']*100:.1f}%",
                         'Détail':r.get('interpretation','')})
        if ecg_has:
            r = st.session_state['ecg_result']
            rows.append({'Analyse':'❤️ ECG','Résultat':r['predicted_class'],
                         'Confiance':f"{r['confidence']*100:.1f}%",
                         'Détail':r.get('recommendation','')})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="stitle">Rapport IA (Claude)</div>',
                    unsafe_allow_html=True)

        if not os.environ.get("ANTHROPIC_API_KEY",""):
            st.info("💡 `ANTHROPIC_API_KEY` non définie — rapport structuré généré sans IA. "
                    "Définissez la variable pour activer Claude.")

        cb1, cb2 = st.columns([1,2])
        with cb1:
            gen = st.button("🤖 Générer le rapport", type="primary",
                            use_container_width=True)
        with cb2:
            st.caption("Rapport généré par Claude (Anthropic) ou en mode structuré "
                       "si la clé API n'est pas disponible.")

        if gen:
            with st.spinner("Génération en cours..."):
                t0 = time.time()
                rapport, source = generate_report(
                    st.session_state.get('eeg_result'),
                    st.session_state.get('ecg_result'),
                    patient_info,
                )
                ela_r = time.time() - t0
            st.session_state['rapport'] = rapport
            if source == 'claude':
                st.success(f"✅ Rapport Claude généré en {ela_r:.2f}s")
            else:
                st.info(f"Rapport structuré généré en {ela_r:.2f}s (mode hors-ligne)")

        if 'rapport' in st.session_state:
            st.markdown(f"""<div class="rapport-box">
                {st.session_state['rapport']
                    .replace(chr(10),'<br>')
                    .replace('**','<b>',1)
                    .replace('**','</b>',1)}
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            d1,d2 = st.columns(2)
            with d1:
                st.download_button("📄 Télécharger (.txt)",
                    data=st.session_state['rapport'],
                    file_name=f"rapport_somnia_{p_age}ans.txt",
                    mime="text/plain", use_container_width=True)
            with d2:
                export = {"patient":patient_info,
                          "eeg":st.session_state.get('eeg_result',{}),
                          "ecg":st.session_state.get('ecg_result',{}),
                          "rapport":st.session_state.get('rapport','')}
                st.download_button("📊 Exporter (.json)",
                    data=json.dumps(export, indent=2, ensure_ascii=False),
                    file_name=f"somnia_data_{p_age}ans.json",
                    mime="application/json", use_container_width=True)

        st.markdown("""<div class="med-alert">
            ⚠️ <b>Rappel :</b> Ce rapport est un outil d'aide à la décision.
            Le diagnostic et la prise en charge restent de la responsabilité du médecin.
        </div>""", unsafe_allow_html=True)

    else:
        st.info("Effectuez au moins une analyse (EEG ou ECG) pour générer un rapport.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE MONITORING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Monitoring":

    st.markdown("""
    <div style='padding:0.5rem 0 1rem 0;'>
        <h2 style='font-family:"DM Serif Display",serif;font-size:2rem;
                   color:#1B2A4A;margin:0;'>📈 Monitoring</h2>
        <div style='color:#6C757D;font-size:0.95rem;'>
            Statistiques d'utilisation et surveillance du modèle
        </div>
    </div>
    """, unsafe_allow_html=True)

    if api_ok:
        try:
            stats  = requests.get(f"{API_URL}/monitoring/stats?last_n=100").json()
            drift  = requests.get(f"{API_URL}/monitoring/drift").json()
            recent = requests.get(f"{API_URL}/monitoring/recent?n=10").json()

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Total prédictions", stats.get('total_predictions',0))
            m2.metric("Analyses EEG",      stats.get('eeg_predictions',0))
            m3.metric("Analyses ECG",      stats.get('ecg_predictions',0))
            avg = stats.get('avg_confidence')
            m4.metric("Confiance moyenne", f"{avg*100:.1f}%" if avg else "—")

            st.markdown('<div class="stitle">Détection de drift</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Drift de confiance")
                if drift.get('drift_detected'):
                    st.error(f"⚠️ {drift.get('message')}")
                else:
                    st.success(f"✅ {drift.get('message','Stable')}")
            with c2:
                st.caption("Drift des features (vs entraînement)")
                try:
                    drift_f = requests.get(
                        f"{API_URL}/monitoring/drift/features?task=sleep_stage"
                    ).json()
                    if drift_f.get('drift_detected'):
                        st.error(f"⚠️ EEG — {drift_f.get('message')}")
                    else:
                        st.success(f"✅ EEG — {drift_f.get('message','Stable')}")
                except Exception:
                    st.info("Features drift — API indisponible")

            preds = recent.get('predictions',[])
            if preds:
                st.markdown('<div class="stitle">10 dernières prédictions</div>',
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{
                    'Timestamp' : p['timestamp'],
                    'Tâche'     : p['task'],
                    'Prédiction': p['prediction'],
                    'Confiance' : f"{p['confidence']*100:.1f}%",
                    'Temps (ms)': p['processing_time_ms'],
                } for p in preds]), use_container_width=True, hide_index=True)
            else:
                st.info("Aucune prédiction enregistrée pour le moment.")
        except Exception as e:
            st.error(f"Erreur monitoring : {e}")
    else:
        st.error("API non disponible — démarrez le serveur FastAPI")
