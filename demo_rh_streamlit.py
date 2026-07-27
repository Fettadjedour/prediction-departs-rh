============================================================
#  DÉMO CLIENT — PRÉDICTION DES DÉPARTS RH
#  Application Streamlit professionnelle
# ============================================================
#
#  COMMENT LANCER CETTE DÉMO :
#
#  Option 1 — En ligne (RECOMMANDÉ, gratuit, rien à installer) :
#    1. Va sur https://share.streamlit.io
#    2. Connecte ton compte GitHub
#    3. Upload ce fichier sur GitHub
#    4. Streamlit génère un lien public à partager au client
#
#  Option 2 — Sur ton ordinateur :
#    1. Ouvre un terminal
#    2. pip install streamlit scikit-learn pandas numpy plotly
#    3. streamlit run demo_rh_streamlit.py
#    4. S'ouvre automatiquement dans le navigateur
#
# ============================================================
 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
 
# ── Configuration de la page ──────────────────────────────
st.set_page_config(
    page_title="RH Predict — Anticipez les départs",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ── CSS personnalisé — Thème RH Clair & Professionnel ─────
st.markdown("""
<style>
    /* Fond général blanc cassé chaleureux */
    .main { background-color: #f5f7fa; }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f5f7fa;
    }
 
    /* Sidebar couleur RH — bleu marine doux */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b3a6b 0%, #2563a8 100%);
        color: white;
    }
    [data-testid="stSidebar"] label { color: white !important; }
    [data-testid="stSidebar"] p { color: white !important; }
    [data-testid="stSidebar"] span { color: white !important; }
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] { color: white !important; }
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: rgba(255,255,255,0.3) !important;
    }
 
    /* Titres en bleu RH */
    h1 { color: #1b3a6b !important; font-weight: 800 !important; }
    h2, h3 { color: #2563a8 !important; }
 
    /* Cartes blanches avec ombre légère */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(27,58,107,0.10);
        text-align: center;
        margin-bottom: 16px;
        border-top: 3px solid #2563a8;
    }
 
    /* Cartes de risque */
    .risk-high {
        background: linear-gradient(135deg, #fff0f0, #ffd6d6);
        border-left: 5px solid #dc2626;
        padding: 20px; border-radius: 12px;
        color: #7f1d1d !important;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        border-left: 5px solid #d97706;
        padding: 20px; border-radius: 12px;
        color: #78350f !important;
    }
    .risk-low {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-left: 5px solid #16a34a;
        padding: 20px; border-radius: 12px;
        color: #14532d !important;
    }
 
    /* Boutons RH bleu */
    .stButton > button {
        background: linear-gradient(135deg, #1b3a6b, #2563a8) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
    }
 
    /* Séparateur RH */
    hr { border-color: #dbeafe !important; }
 
    /* Métriques Streamlit */
    [data-testid="stMetricValue"] {
        color: #1b3a6b !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════
# ENTRAÎNEMENT DU MODÈLE (fait une seule fois au démarrage)
# ══════════════════════════════════════════════════════════
@st.cache_resource
def entrainer_modele():
    np.random.seed(42)
    n = 1470
    departements = ['Ventes', 'Technique', 'Support', 'IT', 'RH',
                    'Comptabilité', 'Marketing', 'R&D', 'Management', 'Produit']
    salaires_list = ['Faible', 'Moyen', 'Élevé']
 
    df = pd.DataFrame({
        'satisfaction':        np.random.uniform(0.1, 1.0, n).round(2),
        'derniere_evaluation': np.random.uniform(0.4, 1.0, n).round(2),
        'nb_projets':          np.random.randint(2, 8, n),
        'heures_mois':         np.random.randint(96, 310, n),
        'anciennete':          np.random.randint(1, 11, n),
        'accident_travail':    np.random.randint(0, 2, n),
        'promotion_5ans':      np.random.randint(0, 2, n),
        'departement':         np.random.choice(departements, n),
        'salaire':             np.random.choice(salaires_list, n),
    })
 
    prob_depart = (
        (1 - df['satisfaction']) * 0.5 +
        (df['heures_mois'] / 310) * 0.3 +
        (df['salaire'] == 'Faible').astype(int) * 0.2
    )
    df['depart'] = (prob_depart > np.random.uniform(0, 1, n)).astype(int)
 
    le_dept = LabelEncoder()
    le_sal  = LabelEncoder()
    df['departement_enc'] = le_dept.fit_transform(df['departement'])
    df['salaire_enc']     = le_sal.fit_transform(df['salaire'])
 
    features = ['satisfaction', 'derniere_evaluation', 'nb_projets',
                'heures_mois', 'anciennete', 'accident_travail',
                'promotion_5ans', 'departement_enc', 'salaire_enc']
 
    modele = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    modele.fit(df[features], df['depart'])
 
    return modele, le_dept, le_sal, features
 
modele, le_dept, le_sal, features = entrainer_modele()
 
 
# ══════════════════════════════════════════════════════════
# EN-TÊTE DE L'APPLICATION
# ══════════════════════════════════════════════════════════
col_logo, col_titre = st.columns([1, 5])
with col_logo:
    st.markdown("# 👥")
with col_titre:
    st.markdown("# RH Predict")
    st.markdown("*Anticipez les départs · Protégez vos talents · Réduisez vos coûts de recrutement*")
 
st.divider()
 
 
# ══════════════════════════════════════════════════════════
# BARRE LATÉRALE — FORMULAIRE EMPLOYÉ
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 👤 Profil de l'employé")
    st.markdown("*Renseignez les informations pour obtenir une analyse*")
    st.divider()
 
    satisfaction = st.slider(
        "😊 Satisfaction au travail",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0 = très insatisfait · 1 = très satisfait"
    )
 
    evaluation = st.slider(
        "⭐ Dernière évaluation",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="Score de la dernière évaluation de performance"
    )
 
    nb_projets = st.selectbox(
        "📁 Nombre de projets en cours",
        options=[2, 3, 4, 5, 6, 7],
        index=1
    )
 
    heures_mois = st.slider(
        "⏱ Heures travaillées / mois",
        min_value=96, max_value=310, value=160, step=5,
        help="Moyenne mensuelle · Norme = 151h"
    )
 
    anciennete = st.slider(
        "📅 Ancienneté (années)",
        min_value=1, max_value=10, value=3
    )
 
    departement = st.selectbox(
        "🏢 Département",
        options=['Ventes', 'Technique', 'Support', 'IT', 'RH',
                 'Comptabilité', 'Marketing', 'R&D', 'Management', 'Produit']
    )
 
    salaire = st.selectbox(
        "💰 Niveau de salaire",
        options=['Faible', 'Moyen', 'Élevé']
    )
 
    accident = st.radio(
        "🏥 Accident de travail",
        options=['Non', 'Oui'],
        horizontal=True
    )
 
    promotion = st.radio(
        "📈 Promotion dans les 5 dernières années",
        options=['Non', 'Oui'],
        horizontal=True
    )
 
    st.divider()
    analyser = st.button("🔍 Analyser cet employé", use_container_width=True, type="primary")
 
 
# ══════════════════════════════════════════════════════════
# ZONE PRINCIPALE — RÉSULTATS
# ══════════════════════════════════════════════════════════
 
# Préparer les données de l'employé
dept_enc = le_dept.transform([departement])[0]
sal_enc  = le_sal.transform([salaire])[0]
 
employe = pd.DataFrame([[
    satisfaction, evaluation, nb_projets, heures_mois, anciennete,
    1 if accident == 'Oui' else 0,
    1 if promotion == 'Oui' else 0,
    dept_enc, sal_enc
]], columns=features)
 
proba       = modele.predict_proba(employe)[0]
risque_pct  = proba[1] * 100
prediction  = modele.predict(employe)[0]
 
# ── Jauge de risque ───────────────────────────────────────
st.markdown("### 📊 Analyse du risque de départ")
 
col1, col2 = st.columns([2, 1])
 
with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risque_pct,
        number={'suffix': '%', 'font': {'size': 48}},
        title={'text': "Probabilité de départ", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#4f46e5"},
            'steps': [
                {'range': [0, 30],  'color': '#d1fae5'},
                {'range': [30, 60], 'color': '#fef3c7'},
                {'range': [60, 100],'color': '#fee2e2'},
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 4},
                'thickness': 0.75,
                'value': risque_pct
            }
        }
    ))
    fig_gauge.update_layout(
        height=280,
        margin=dict(t=40, b=0, l=20, r=20),
        paper_bgcolor='#f5f7fa',
        plot_bgcolor='#f5f7fa',
        font=dict(color='#1b3a6b')
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
 
with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
 
    if risque_pct >= 60:
        st.markdown(f"""
        <div class="risk-high">
            <h2>⚠️ RISQUE ÉLEVÉ</h2>
            <p><strong>{risque_pct:.0f}%</strong> de probabilité de départ</p>
            <p>Action immédiate recommandée</p>
        </div>
        """, unsafe_allow_html=True)
    elif risque_pct >= 30:
        st.markdown(f"""
        <div class="risk-medium">
            <h2>⚡ RISQUE MODÉRÉ</h2>
            <p><strong>{risque_pct:.0f}%</strong> de probabilité de départ</p>
            <p>Surveillance et actions préventives</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-low">
            <h2>✅ FAIBLE RISQUE</h2>
            <p><strong>{risque_pct:.0f}%</strong> de probabilité de départ</p>
            <p>Employé stable, continuer le suivi</p>
        </div>
        """, unsafe_allow_html=True)
 
st.divider()
 
# ── Facteurs de risque ────────────────────────────────────
st.markdown("### 🔍 Analyse des facteurs de risque")
 
col3, col4 = st.columns(2)
 
with col3:
    # Graphique radar des facteurs
    categories = ['Satisfaction', 'Charge travail', 'Ancienneté', 'Évaluation', 'Nb projets']
 
    # Normaliser pour le radar (0 à 1, 1 = risque élevé)
    val_satisfaction  = 1 - satisfaction
    val_charge        = (heures_mois - 96) / (310 - 96)
    val_anciennete    = anciennete / 10
    val_evaluation    = evaluation
    val_projets       = (nb_projets - 2) / 5
 
    valeurs = [val_satisfaction, val_charge, val_anciennete, val_evaluation, val_projets]
    valeurs += [valeurs[0]]  # fermer le radar
    categories += [categories[0]]
 
    fig_radar = go.Figure(go.Scatterpolar(
        r=valeurs,
        theta=categories,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='#6366f1', width=2),
        name='Profil employé'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='#f5f7fa'
        ),
        title="Profil de risque",
        height=320,
        margin=dict(t=50, b=20),
        paper_bgcolor='#f5f7fa',
        plot_bgcolor='#f5f7fa',
        font=dict(color='#1b3a6b')
    )
    st.plotly_chart(fig_radar, use_container_width=True)
 
with col4:
    st.markdown("#### 💡 Points d'attention")
    st.markdown("")
 
    alertes = []
    points_positifs = []
 
    if satisfaction < 0.4:
        alertes.append(f"😟 Satisfaction très basse ({satisfaction:.0%}) — entretien recommandé")
    elif satisfaction > 0.7:
        points_positifs.append(f"😊 Bonne satisfaction au travail ({satisfaction:.0%})")
 
    if heures_mois > 220:
        alertes.append(f"⏰ Surcharge de travail ({heures_mois}h/mois vs 151h normal)")
    elif heures_mois <= 170:
        points_positifs.append(f"⚖️ Charge de travail équilibrée ({heures_mois}h/mois)")
 
    if salaire == 'Faible':
        alertes.append("💸 Salaire faible — risque d'offres concurrentes")
    elif salaire == 'Élevé':
        points_positifs.append("💰 Rémunération élevée — facteur de rétention")
 
    if promotion == 'Non' and anciennete >= 4:
        alertes.append(f"📈 Aucune promotion en {anciennete} ans — risque de frustration")
    elif promotion == 'Oui':
        points_positifs.append("🎯 Promotion récente — signal positif")
 
    if nb_projets >= 6:
        alertes.append(f"📁 Trop de projets simultanés ({nb_projets}) — risque de burn-out")
 
    if alertes:
        st.markdown("**⚠️ Alertes :**")
        for a in alertes:
            st.markdown(f"- {a}")
 
    if points_positifs:
        st.markdown("**✅ Points positifs :**")
        for p in points_positifs:
            st.markdown(f"- {p}")
 
st.divider()
 
# ── Recommandations ───────────────────────────────────────
st.markdown("### 🎯 Recommandations RH")
 
col5, col6, col7 = st.columns(3)
 
with col5:
    st.markdown("""
    **📋 Actions immédiates**
    - Entretien individuel sous 2 semaines
    - Évaluer la charge de travail réelle
    - Identifier les sources d'insatisfaction
    """)
 
with col6:
    st.markdown("""
    **📅 Actions à 3 mois**
    - Plan de développement personnalisé
    - Révision salariale si justifiée
    - Réduction du nombre de projets
    """)
 
with col7:
    st.markdown("""
    **🔭 Actions à 6 mois**
    - Bilan de satisfaction formalisé
    - Opportunités d'évolution interne
    - Révision du plan de carrière
    """)
 
st.divider()
 
# ── Coût estimé du remplacement ──────────────────────────
st.markdown("### 💰 Impact financier estimé")
 
col8, col9, col10 = st.columns(3)
 
salaire_annuel_map = {'Faible': 28000, 'Moyen': 42000, 'Élevé': 65000}
salaire_annuel = salaire_annuel_map[salaire]
cout_remplacement = salaire_annuel * 0.5  # Convention : 50% du salaire annuel
 
with col8:
    st.metric(
        label="💶 Coût estimé du remplacement",
        value=f"{cout_remplacement:,.0f} €",
        help="Recrutement + formation + perte de productivité"
    )
with col9:
    st.metric(
        label="📉 Risque financier pondéré",
        value=f"{cout_remplacement * proba[1]:,.0f} €",
        help="Coût × probabilité de départ"
    )
with col10:
    st.metric(
        label="💡 Investissement rétention conseillé",
        value=f"{cout_remplacement * proba[1] * 0.3:,.0f} €",
        help="Budget actions RH recommandé (30% du risque financier)"
    )
 
st.divider()
 
# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div style='text-align: center; color: #9ca3af; font-size: 0.85rem; padding: 20px'>
    RH Predict · Outil d'aide à la décision RH · Les prédictions sont indicatives et doivent être combinées au jugement humain
</div>
""", unsafe_allow_html=True)
