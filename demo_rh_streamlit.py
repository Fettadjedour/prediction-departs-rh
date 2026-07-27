import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

st.set_page_config(
    page_title="RH Predict",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* === FOND GÉNÉRAL — Violet lavande doux === */
.stApp { background: linear-gradient(135deg, #f3f0ff 0%, #fdf4ff 50%, #fff7ed 100%); }
.main .block-container { padding: 2rem 2.5rem; }

/* === SIDEBAR — Violet foncé === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2d1b69 0%, #4c1d95 100%);
    border-right: none;
}
[data-testid="stSidebar"] > div { padding-top: 2rem; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #ddd6fe !important; font-size: 0.85rem; }
[data-testid="stSidebar"] hr { border-color: #5b21b6 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    width: 100%;
    padding: 0.65rem !important;
    font-size: 0.9rem !important;
    margin-top: 1rem;
    box-shadow: 0 4px 12px rgba(249,115,22,0.35) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.08) !important;
    border-color: #7c3aed !important;
    color: white !important;
}

/* === TEXTE PRINCIPAL === */
h1 { color: #2d1b69 !important; font-weight: 800 !important; font-size: 1.8rem !important; letter-spacing: -0.5px; }
h2 { color: #2d1b69 !important; font-weight: 700 !important; font-size: 1.2rem !important; }
h3 { color: #6d28d9 !important; font-weight: 600 !important; font-size: 1rem !important; }
h4 { color: #4c1d95 !important; font-weight: 700 !important; }
p, li { color: #374151 !important; font-size: 0.9rem !important; }
strong { color: #2d1b69 !important; }

/* === MÉTRIQUES — fond blanc avec top violet === */
[data-testid="stMetric"] {
    background: white;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(109,40,217,0.1);
    border-top: 3px solid #7c3aed;
}
[data-testid="stMetricValue"] { color: #2d1b69 !important; font-weight: 800 !important; font-size: 1.5rem !important; }
[data-testid="stMetricLabel"] { color: #7c3aed !important; font-size: 0.75rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; }

/* === SÉPARATEUR === */
hr { border-color: #ede9fe !important; margin: 1.5rem 0 !important; }

/* === BADGE RISQUE === */
.badge-high {
    display: inline-block;
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    color: #b91c1c;
    border: 1px solid #fca5a5;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    box-shadow: 0 2px 6px rgba(220,38,38,0.15);
}
.badge-medium {
    display: inline-block;
    background: linear-gradient(135deg, #fff7ed, #ffedd5);
    color: #c2410c;
    border: 1px solid #fdba74;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    box-shadow: 0 2px 6px rgba(249,115,22,0.15);
}
.badge-low {
    display: inline-block;
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    color: #15803d;
    border: 1px solid #86efac;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    box-shadow: 0 2px 6px rgba(16,185,129,0.15);
}

/* === CARTE RÉSULTAT === */
.result-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    box-shadow: 0 2px 12px rgba(109,40,217,0.08);
    margin-bottom: 1.5rem;
    border: 1px solid #ede9fe;
}

/* === ACTION CARDS === */
.action-card {
    background: white;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 8px rgba(109,40,217,0.07);
    border-left: 4px solid #7c3aed;
    height: 100%;
    border-top: 1px solid #ede9fe;
}
.action-card h4 {
    color: #4c1d95 !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.8rem;
}
.action-card p, .action-card li {
    color: #4b5563 !important;
    font-size: 0.85rem !important;
    line-height: 1.7;
}

/* === ALERTES === */
.alert-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid #f5f3ff;
    font-size: 0.85rem;
    color: #374151 !important;
}
.alert-item:last-child { border-bottom: none; }
</style>
""", unsafe_allow_html=True)


# ── Entraînement du modèle ──────────────────────────────────
@st.cache_resource
def entrainer_modele():
    np.random.seed(42)
    n = 1470
    depts = ['Ventes', 'Technique', 'Support', 'IT', 'RH',
             'Comptabilité', 'Marketing', 'R&D', 'Management', 'Produit']
    sals = ['Faible', 'Moyen', 'Élevé']
    df = pd.DataFrame({
        'satisfaction':        np.random.uniform(0.1, 1.0, n).round(2),
        'derniere_evaluation': np.random.uniform(0.4, 1.0, n).round(2),
        'nb_projets':          np.random.randint(2, 8, n),
        'heures_mois':         np.random.randint(96, 310, n),
        'anciennete':          np.random.randint(1, 11, n),
        'accident_travail':    np.random.randint(0, 2, n),
        'promotion_5ans':      np.random.randint(0, 2, n),
        'departement':         np.random.choice(depts, n),
        'salaire':             np.random.choice(sals, n),
    })
    prob = ((1 - df['satisfaction']) * 0.5 +
            (df['heures_mois'] / 310) * 0.3 +
            (df['salaire'] == 'Faible').astype(int) * 0.2)
    df['depart'] = (prob > np.random.uniform(0, 1, n)).astype(int)
    le_d = LabelEncoder(); le_s = LabelEncoder()
    df['dept_enc'] = le_d.fit_transform(df['departement'])
    df['sal_enc']  = le_s.fit_transform(df['salaire'])
    feats = ['satisfaction','derniere_evaluation','nb_projets','heures_mois',
             'anciennete','accident_travail','promotion_5ans','dept_enc','sal_enc']
    m = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    m.fit(df[feats], df['depart'])
    return m, le_d, le_s, feats

modele, le_dept, le_sal, features = entrainer_modele()


# ── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Profil de l'employé")
    st.markdown("---")

    satisfaction = st.slider("Satisfaction au travail", 0.0, 1.0, 0.5, 0.05,
                             help="0 = très insatisfait · 1 = très satisfait")
    evaluation   = st.slider("Dernière évaluation", 0.0, 1.0, 0.7, 0.05)
    nb_projets   = st.selectbox("Nombre de projets", [2,3,4,5,6,7], index=1)
    heures_mois  = st.slider("Heures / mois", 96, 310, 160, 5,
                              help="Norme = 151h/mois")
    anciennete   = st.slider("Ancienneté (années)", 1, 10, 3)
    departement  = st.selectbox("Département",
                   ['Ventes','Technique','Support','IT','RH',
                    'Comptabilité','Marketing','R&D','Management','Produit'])
    salaire      = st.selectbox("Niveau de salaire", ['Faible','Moyen','Élevé'])
    accident     = st.radio("Accident de travail", ['Non','Oui'], horizontal=True)
    promotion    = st.radio("Promotion (5 dernières années)", ['Non','Oui'], horizontal=True)
    st.markdown("---")
    analyser = st.button("Analyser cet employé", use_container_width=True)


# ── CALCUL ──────────────────────────────────────────────────
dept_enc = le_dept.transform([departement])[0]
sal_enc  = le_sal.transform([salaire])[0]
employe  = pd.DataFrame([[satisfaction, evaluation, nb_projets, heures_mois,
                          anciennete, 1 if accident=='Oui' else 0,
                          1 if promotion=='Oui' else 0, dept_enc, sal_enc]],
                        columns=features)
proba      = modele.predict_proba(employe)[0]
risque_pct = proba[1] * 100

if risque_pct >= 60:
    niveau, couleur_badge, couleur_barre = "RISQUE ÉLEVÉ", "badge-high", "#dc2626"
elif risque_pct >= 30:
    niveau, couleur_badge, couleur_barre = "RISQUE MODÉRÉ", "badge-medium", "#d97706"
else:
    niveau, couleur_badge, couleur_barre = "FAIBLE RISQUE", "badge-low", "#16a34a"


# ── EN-TÊTE ─────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:0.5rem;">
  <div style="background:linear-gradient(135deg,#7c3aed,#a855f7); color:white; width:48px; height:48px;
              border-radius:12px; display:flex; align-items:center;
              justify-content:center; font-size:1.4rem;
              box-shadow:0 4px 12px rgba(124,58,237,0.35);">👥</div>
  <div>
    <div style="font-size:1.6rem; font-weight:800; color:#2d1b69; line-height:1.2;">RH Predict</div>
    <div style="font-size:0.82rem; color:#7c3aed; margin-top:2px; font-weight:500;">
      Plateforme d'analyse prédictive des ressources humaines
    </div>
  </div>
  <div style="margin-left:auto;">
    <span class="{couleur_badge}">{niveau}</span>
  </div>
</div>
<hr>
""", unsafe_allow_html=True)


# ── KPIs ────────────────────────────────────────────────────
sal_map = {'Faible': 28000, 'Moyen': 42000, 'Élevé': 65000}
sal_ann = sal_map[salaire]
cout    = sal_ann * 0.5

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Probabilité de départ", f"{risque_pct:.1f}%")
with c2:
    st.metric("Coût de remplacement", f"{cout:,.0f} €")
with c3:
    st.metric("Risque financier", f"{cout * proba[1]:,.0f} €")
with c4:
    st.metric("Ancienneté", f"{anciennete} ans")

st.markdown("<br>", unsafe_allow_html=True)


# ── JAUGE + FACTEURS ────────────────────────────────────────
col_gauge, col_factors = st.columns([3, 2])

with col_gauge:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Score de risque de départ")


    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risque_pct,
        number={'suffix': '%', 'font': {'size': 52, 'color': '#0f1f3d', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1,
                     'tickcolor': '#9ca3af', 'tickfont': {'color': '#6b7280', 'size': 11}},
            'bar': {'color': couleur_barre, 'thickness': 0.25},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30],  'color': '#f0fdf4'},
                {'range': [30, 60], 'color': '#fffbeb'},
                {'range': [60, 100],'color': '#fef2f2'},
            ],
            'threshold': {
                'line': {'color': couleur_barre, 'width': 3},
                'thickness': 0.8, 'value': risque_pct
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=20, b=0, l=30, r=30),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Inter', color='#2d1b69')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_factors:
    st.markdown('<div class="result-card" style="height:100%">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Points d'attention")

    alertes, positifs = [], []
    if satisfaction < 0.4:
        alertes.append(f"Satisfaction critique ({satisfaction:.0%})")
    elif satisfaction > 0.7:
        positifs.append(f"Bonne satisfaction ({satisfaction:.0%})")
    if heures_mois > 220:
        alertes.append(f"Surcharge de travail ({heures_mois}h/mois)")
    elif heures_mois <= 170:
        positifs.append(f"Charge équilibrée ({heures_mois}h/mois)")
    if salaire == 'Faible':
        alertes.append("Rémunération sous le marché")
    elif salaire == 'Élevé':
        positifs.append("Rémunération attractive")
    if promotion == 'Non' and anciennete >= 4:
        alertes.append(f"Aucune promotion en {anciennete} ans")
    elif promotion == 'Oui':
        positifs.append("Promotion récente")
    if nb_projets >= 6:
        alertes.append(f"Surcharge projets ({nb_projets} simultanés)")

    if alertes:
        st.markdown("**Alertes**")
        for a in alertes:
            st.markdown(f'<div class="alert-item">🔴 {a}</div>', unsafe_allow_html=True)
    if positifs:
        st.markdown("**Points positifs**")
        for p in positifs:
            st.markdown(f'<div class="alert-item">🟢 {p}</div>', unsafe_allow_html=True)
    if not alertes and not positifs:
        st.markdown("Profil équilibré, continuer le suivi régulier.")
    st.markdown('</div>', unsafe_allow_html=True)


# ── GRAPHIQUE RADAR ─────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🕸️ Profil de risque détaillé")

col_radar, col_info = st.columns([3, 2])

with col_radar:
    categories = ['Insatisfaction', 'Surcharge', 'Ancienneté', 'Évaluation', 'Nb Projets']
    valeurs = [
        1 - satisfaction,
        (heures_mois - 96) / (310 - 96),
        anciennete / 10,
        evaluation,
        (nb_projets - 2) / 5
    ]
    valeurs_fermees = valeurs + [valeurs[0]]
    categories_fermees = categories + [categories[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=valeurs_fermees,
        theta=categories_fermees,
        fill='toself',
        fillcolor='rgba(124, 58, 237, 0.15)',
        line=dict(color='#7c3aed', width=2.5),
        name='Profil employé'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                           tickfont=dict(color='#7c3aed', size=10),
                           gridcolor='#ede9fe'),
            angularaxis=dict(tickfont=dict(color='#2d1b69', size=11)),
            bgcolor='#faf5ff'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=300,
        margin=dict(t=30, b=20, l=50, r=50),
        font=dict(family='Inter', color='#2d1b69'),
        showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_info:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#f5f3ff; border-radius:10px; padding:1.2rem; border:1px solid #ede9fe;">
    <p style="color:#4c1d95 !important; font-weight:700; font-size:0.85rem; margin-bottom:0.8rem;">
    📖 Comment lire ce graphique</p>
    <p style="color:#4b5563 !important; font-size:0.83rem; line-height:1.7;">
    Plus la surface colorée est grande, plus le risque est élevé.<br><br>
    <strong>Insatisfaction</strong> → Moral de l'employé<br>
    <strong>Surcharge</strong> → Volume de travail<br>
    <strong>Ancienneté</strong> → Années sans évolution<br>
    <strong>Évaluation</strong> → Performance récente<br>
    <strong>Nb Projets</strong> → Charge simultanée
    </p></div>""", unsafe_allow_html=True)


# ── PLAN D'ACTION ────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🎯 Plan d'action recommandé")

a1, a2, a3 = st.columns(3)
with a1:
    st.markdown("""
    <div class="action-card" style="border-left-color:#f97316; background:linear-gradient(135deg,#fff7ed,#fff);">
      <h4 style="color:#c2410c !important;">⚡ Actions immédiates</h4>
      <ul>
        <li>Entretien individuel sous 2 semaines</li>
        <li>Évaluer la charge de travail réelle</li>
        <li>Identifier les sources d'insatisfaction</li>
      </ul>
    </div>""", unsafe_allow_html=True)
with a2:
    st.markdown("""
    <div class="action-card" style="border-left-color:#7c3aed; background:linear-gradient(135deg,#f5f3ff,#fff);">
      <h4 style="color:#4c1d95 !important;">📅 À 3 mois</h4>
      <ul>
        <li>Plan de développement personnalisé</li>
        <li>Révision salariale si justifiée</li>
        <li>Réduction du nombre de projets</li>
      </ul>
    </div>""", unsafe_allow_html=True)
with a3:
    st.markdown("""
    <div class="action-card" style="border-left-color:#10b981; background:linear-gradient(135deg,#f0fdf4,#fff);">
      <h4 style="color:#065f46 !important;">🔭 À 6 mois</h4>
      <ul>
        <li>Bilan de satisfaction formalisé</li>
        <li>Opportunités d'évolution interne</li>
        <li>Révision du plan de carrière</li>
      </ul>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── IMPACT FINANCIER ────────────────────────────────────────
st.markdown("---")
st.markdown("#### 💶 Analyse financière")

f1, f2, f3 = st.columns(3)
with f1:
    st.metric("Coût estimé du remplacement", f"{cout:,.0f} €",
              help="Recrutement + formation + perte de productivité")
with f2:
    st.metric("Exposition financière", f"{cout * proba[1]:,.0f} €",
              help="Coût pondéré par la probabilité de départ")
with f3:
    st.metric("Budget rétention conseillé", f"{cout * proba[1] * 0.3:,.0f} €",
              help="30% de l'exposition — investissement recommandé")


# ── FOOTER ──────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="display:flex; justify-content:space-between; align-items:center;
            padding: 0.5rem 0; color: #9ca3af; font-size: 0.78rem;">
  <span>© 2026 RH Predict · Outil d'aide à la décision</span>
  <span>Les prédictions sont indicatives et doivent être combinées au jugement humain</span>
</div>
""", unsafe_allow_html=True)
