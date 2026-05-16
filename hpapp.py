import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)
 
# ─────────────────────────────────────────────
#  GLOBAL CSS  –  Dark luxury theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google Fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');
 
/* ---------- Root Variables ---------- */
:root {
    --bg-dark:    #0d0f14;
    --bg-card:    #161a23;
    --bg-hover:   #1e2330;
    --gold:       #c9a84c;
    --gold-light: #e4c97e;
    --text-main:  #f0eade;
    --text-muted: #8a8fa8;
    --accent:     #3a7bd5;
    --success:    #2ecc71;
    --border:     rgba(201, 168, 76, 0.18);
}
 
/* ---------- Base ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark) !important;
    color: var(--text-main) !important;
    font-family: 'DM Sans', sans-serif;
}
 
[data-testid="stHeader"] { background: transparent !important; }
 
/* hide default streamlit menu */
#MainMenu, footer { visibility: hidden; }
 
/* ---------- Tabs ---------- */
[data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    transition: all 0.25s ease !important;
}
[aria-selected="true"] {
    background: var(--gold) !important;
    color: #0d0f14 !important;
}
 
/* ---------- Sliders ---------- */
[data-testid="stSlider"] .rc-slider-track { background: var(--gold) !important; }
[data-testid="stSlider"] .rc-slider-handle {
    border-color: var(--gold) !important;
    background: var(--bg-dark) !important;
    box-shadow: 0 0 8px var(--gold) !important;
}
 
/* ---------- Predict Button ---------- */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, var(--gold), var(--gold-light)) !important;
    color: #0d0f14 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 17px !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    box-shadow: 0 4px 24px rgba(201, 168, 76, 0.35) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(201, 168, 76, 0.55) !important;
}
 
/* ---------- Metric Cards ---------- */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 13px !important; }
[data-testid="stMetricValue"] { color: var(--gold) !important; font-size: 26px !important; font-weight: 700 !important; }
 
/* ---------- Success alert ---------- */
[data-testid="stAlert"] {
    background: rgba(46, 204, 113, 0.08) !important;
    border: 1px solid rgba(46, 204, 113, 0.4) !important;
    border-radius: 14px !important;
    color: var(--success) !important;
}
 
/* ---------- Dataframes / bar chart ---------- */
[data-testid="stVegaLiteChart"] { border-radius: 12px !important; overflow: hidden !important; }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    padding: 28px 36px;
    margin-bottom: 24px;
    border-bottom: 1px solid rgba(201,168,76,0.2);
">
    <h1 style="
        font-family:'Playfair Display',serif;
        font-size: clamp(26px, 4vw, 42px);
        font-weight: 700;
        color: #f0eade;
        margin: 0;
        line-height: 1.2;
    ">🏠 House Price Prediction</h1>
</div>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("hpp.pkl", "rb") as f:
        return pickle.load(f)
 
model = load_model()
 
 
# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏡  Prediction", "📊  Analytics", "📘  About"])
 
 
# ════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ════════════════════════════════════════════
with tab1:
 
    # ── Section label ──
    st.markdown("""
    <p style="
        color:#c9a84c;
        font-size:12px;
        letter-spacing:2.5px;
        text-transform:uppercase;
        margin:8px 0 20px;
    ">Property Details</p>
    """, unsafe_allow_html=True)
 
    # ── Input Grid ──
    col1, col2 = st.columns(2, gap="large")
 
    with col1:
        # Card wrapper
        st.markdown('<div style="background:#161a23;border:1px solid rgba(201,168,76,0.15);border-radius:16px;padding:24px;">', unsafe_allow_html=True)
        st.markdown("##### 📐 Area  &  🛏 Bedrooms")
        size_sqft = st.slider("Area (sq ft)", min_value=100, max_value=10000, value=1200, step=50)
        bedrooms  = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
        st.markdown("</div>", unsafe_allow_html=True)
 
    with col2:
        st.markdown('<div style="background:#161a23;border:1px solid rgba(201,168,76,0.15);border-radius:16px;padding:24px;">', unsafe_allow_html=True)
        st.markdown("##### 🏚 Age  &  📍 Location")
        age      = st.slider("House Age (years)", min_value=0, max_value=50, value=5)
        distance = st.slider("Distance from City (km)", min_value=1, max_value=50, value=10)
        st.markdown("</div>", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Predict Button ──
    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_clicked = st.button("✦  Predict Price Now", use_container_width=True)
 
    # ── Result ──
    if predict_clicked:
        features = np.array([[size_sqft, bedrooms, age, distance]])
        result   = model.predict(features)[0]
 
        # Big price display
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #161a23, #1a2016);
            border: 1px solid rgba(201,168,76,0.35);
            border-radius: 18px;
            padding: 32px 36px;
            margin: 24px 0 20px;
            text-align: center;
        ">
            <p style="color:#8a8fa8;font-size:13px;letter-spacing:2px;text-transform:uppercase;margin:0 0 6px;">
                Estimated Market Value
            </p>
            <h2 style="
                font-family:'Playfair Display',serif;
                color:#c9a84c;
                font-size:clamp(32px,5vw,52px);
                margin:0;
            ">₹ {round(result, 2):,.2f}</h2>
            <p style="color:#2ecc71;font-size:13px;margin:8px 0 0;">
                ✔ Prediction generated successfully
            </p>
        </div>
        """, unsafe_allow_html=True)
 
        # Metric row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Area",      f"{size_sqft} sqft")
        m2.metric("Bedrooms",  f"{bedrooms} BHK")
        m3.metric("House Age", f"{age} yrs")
        m4.metric("Distance",  f"{distance} km")
 
 
# ════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ════════════════════════════════════════════
with tab2:
    st.markdown("""
    <p style="color:#c9a84c;font-size:12px;letter-spacing:2.5px;text-transform:uppercase;margin:8px 0 20px;">
    Feature Impact Analysis
    </p>
    """, unsafe_allow_html=True)
 
    # Bar chart of current input values
    chart_data = pd.DataFrame({
        "Feature": ["Area (×10 sqft)", "Bedrooms", "Age (yrs)", "Distance (km)"],
        "Value":   [size_sqft / 10, bedrooms, age, distance]
    })
    st.bar_chart(chart_data.set_index("Feature"), height=320, use_container_width=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Insight cards
    st.markdown("""
    <p style="color:#c9a84c;font-size:12px;letter-spacing:2.5px;text-transform:uppercase;margin:0 0 16px;">
    Key Insights
    </p>
    """, unsafe_allow_html=True)
 
    insights = [
        ("📐", "Area",     "Larger area directly increases the predicted price."),
        ("🛏", "Bedrooms", "More bedrooms signal higher property value."),
        ("🏚", "Age",      "Older properties tend to attract a lower valuation."),
        ("📍", "Distance", "Greater distance from the city centre reduces price."),
    ]
 
    cols = st.columns(4)
    for col, (icon, title, text) in zip(cols, insights):
        col.markdown(f"""
        <div style="
            background:#161a23;
            border:1px solid rgba(201,168,76,0.15);
            border-radius:14px;
            padding:20px 18px;
            height:100%;
        ">
            <div style="font-size:28px;margin-bottom:8px;">{icon}</div>
            <p style="color:#c9a84c;font-weight:600;margin:0 0 6px;font-size:14px;">{title}</p>
            <p style="color:#8a8fa8;font-size:13px;margin:0;line-height:1.5;">{text}</p>
        </div>
        """, unsafe_allow_html=True)
 
 
# ════════════════════════════════════════════
#  TAB 3 — ABOUT
# ════════════════════════════════════════════
with tab3:
    st.markdown("""
    <p style="color:#c9a84c;font-size:12px;letter-spacing:2.5px;text-transform:uppercase;margin:8px 0 24px;">
    Project Documentation
    </p>
    """, unsafe_allow_html=True)
 
    # Two-column layout
    left, right = st.columns(2, gap="large")
 
    with left:
        st.markdown("""
        <div style="background:#161a23;border:1px solid rgba(201,168,76,0.15);border-radius:16px;padding:28px;">
            <h4 style="color:#f0eade;margin:0 0 16px;">🎯 Objective</h4>
            <p style="color:#8a8fa8;font-size:14px;line-height:1.7;margin:0;">
                Predict residential house prices using a trained Machine Learning regression model, 
                giving buyers and sellers an instant, data-driven market estimate.
            </p>
        </div>
        <br>
        <div style="background:#161a23;border:1px solid rgba(201,168,76,0.15);border-radius:16px;padding:28px;">
            <h4 style="color:#f0eade;margin:0 0 16px;">⚙️ Tech Stack</h4>
            <ul style="color:#8a8fa8;font-size:14px;line-height:2;margin:0;padding-left:18px;">
                <li>Python 3.x</li>
                <li>Streamlit — UI framework</li>
                <li>Scikit-learn — ML model</li>
                <li>NumPy &amp; Pandas — data processing</li>
                <li>Pickle — model serialisation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
 
    with right:
        st.markdown("""
        <div style="background:#161a23;border:1px solid rgba(201,168,76,0.15);border-radius:16px;padding:28px;">
            <h4 style="color:#f0eade;margin:0 0 16px;">🔄 How It Works</h4>
            <ol style="color:#8a8fa8;font-size:14px;line-height:2.1;margin:0;padding-left:18px;">
                <li>User adjusts property sliders</li>
                <li>Input values are packaged into a feature vector</li>
                <li>Pre-trained ML model runs inference</li>
                <li>Predicted price is rendered in real time</li>
            </ol>
        </div>
        <br>
        <div style="background:#161a23;border:1px solid rgba(201,168,76,0.15);border-radius:16px;padding:28px;">
            <h4 style="color:#f0eade;margin:0 0 16px;">🚀 Future Scope</h4>
            <ul style="color:#8a8fa8;font-size:14px;line-height:2;margin:0;padding-left:18px;">
                <li>Live real-estate data integration</li>
                <li>GPS / map-based location input</li>
                <li>Neighbourhood amenity scoring</li>
                <li>Mobile-first PWA version</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top:48px;
    padding:24px 32px;
    background:#161a23;
    border:1px solid rgba(201,168,76,0.15);
    border-radius:16px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    flex-wrap:wrap;
    gap:12px;
">
    <div>
        <p style="color:#f0eade;font-weight:600;margin:0;font-size:14px;">
            👨‍💻 Akhilesh Sonkar , Anirudh Singh, Kushal Kumar Pandey 
        </p>
        <p style="color:#8a8fa8;margin:2px 0 0;font-size:12px;">
            B.Tech CSE · AKTU
        </p>
    </div>
    <p style="color:#8a8fa8;font-size:12px;margin:0;">
        House Price Predictor · Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)