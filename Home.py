import streamlit as st
from utils.ui_components import get_base_css, create_hero_section, create_feature_card

st.set_page_config(
    page_title="Vehicle Collision Engine | Home",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject Premium CSS
st.markdown(get_base_css(), unsafe_allow_html=True)

# Hero Section
st.markdown(create_hero_section(
    "Vehicle Collision Analysis Engine",
    "AI-Powered National Traffic Safety Intelligence Platform"
), unsafe_allow_html=True)

# About Section
st.markdown("""
<div style='background: rgba(255, 255, 255, 0.02); padding: 32px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 40px;'>
    <h2 style='text-align: center; margin-bottom: 24px;'>🎯 About This Platform</h2>
    <p style='font-size: 1.1rem; line-height: 1.8; color: #c0c0c0; text-align: center; max-width: 900px; margin: 0 auto;'>
        The <strong style='color: #ff4b4b;'>Vehicle Collision Analysis Engine</strong> is an advanced AI-powered system designed to 
        predict traffic risks and analyze accident patterns across <strong>36 Indian States & Union Territories</strong>. 
        Unlike traditional reactive systems, this platform <strong>proactively identifies high-risk zones</strong> using machine learning, 
        geospatial analytics, and explainable AI to help authorities prevent accidents before they occur.
    </p>
</div>
""", unsafe_allow_html=True)

# Feature Cards
st.markdown("## 🚀 Platform Features")
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(create_feature_card(
        "📊 Analytics Hub",
        "Comprehensive historical analysis with interactive dashboards covering 7,500+ accident records across India. Features include geospatial heat mapping, temporal trend analysis, vehicle-wise risk profiling, and weather impact assessment.",
        "📊",
        "Explore Dashboard"
    ), unsafe_allow_html=True)
    if st.button("Open Analytics", key="btn_analytics", use_container_width=True):
        st.switch_page("pages/1_Analytics.py")

with col2:
    st.markdown(create_feature_card(
        "🔮 Risk Prediction",
        "State-of-the-art Hybrid ML Model (XGBoost + Random Forest) with R² score of 0.89. Provides real-time risk scoring for any scenario with detailed Explainable AI insights showing exactly why a prediction was made.",
        "🔮",
        "Start Prediction"
    ), unsafe_allow_html=True)
    if st.button("Predict Risk", key="btn_predict", use_container_width=True):
        st.switch_page("pages/2_Risk_Prediction.py")

with col3:
    st.markdown(create_feature_card(
        "📡 Live Operations",
        "Simulated real-time control center mimicking professional traffic management systems. Monitor live incident feeds, CCTV surveillance, and automated alert systems designed for highway patrol authorities.",
        "📡",
        "Enter Control Room"
    ), unsafe_allow_html=True)
    if st.button("Live Feed", key="btn_live", use_container_width=True):
        st.switch_page("pages/3_Live_Ops.py")

st.markdown("<br>", unsafe_allow_html=True)

# Key Capabilities
st.markdown("## ✨ Key Capabilities")
cap1, cap2 = st.columns(2, gap="large")

with cap1:
    st.markdown("""
    <div class='glass-card'>
        <h3 style='margin-top: 0;'>🎯 What This System Does</h3>
        <ul style='color: #b0b0b0; line-height: 2; font-size: 1rem;'>
            <li><strong style='color: #ff4b4b;'>Accident Pattern Recognition</strong>: Identifies high-risk combinations of weather, vehicle type, time, and location</li>
            <li><strong style='color: #8b5cf6;'>Predictive Risk Scoring</strong>: Calculates danger levels for any given scenario in real-time</li>
            <li><strong style='color: #10b981;'>Geospatial Hotspot Mapping</strong>: Visualizes accident clusters across India to identify "Greyspots"</li>
            <li><strong style='color: #3b82f6;'>Explainable AI</strong>: Provides human-readable reasons behind every prediction</li>
            <li><strong style='color: #f59e0b;'>Safety Recommendations</strong>: Generates actionable advice for authorities and commuters</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with cap2:
    st.markdown("""
    <div class='glass-card'>
        <h3 style='margin-top: 0;'>🎓 Technical Highlights</h3>
        <ul style='color: #b0b0b0; line-height: 2; font-size: 1rem;'>
            <li><strong style='color: #ff4b4b;'>Hybrid Ensemble Model</strong>: Combines Random Forest and XGBoost for superior accuracy</li>
            <li><strong style='color: #8b5cf6;'>MoRTH-Standardized Data</strong>: Training data mirrors official Ministry statistics for realism</li>
            <li><strong style='color: #10b981;'>36 States/UTs Coverage</strong>: Complete national-level analysis capability</li>
            <li><strong style='color: #3b82f6;'>Interactive Visualizations</strong>: Plotly-based charts with drill-down capabilities</li>
            <li><strong style='color: #f59e0b;'>Modular Architecture</strong>: Clean separation of data, models, and UI components</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Technical Specs Section
st.markdown("## 🔧 System Architecture & Performance")
st.markdown("""
<div style='background: rgba(255, 255, 255, 0.02); padding: 40px; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.05);'>
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 32px;'>
        <div style='text-align: center; padding: 20px;'>
            <div style='font-size: 2.5rem; margin-bottom: 12px;'>⚡</div>
            <div style='font-weight: 700; font-size: 1.5rem; color: #ff4b4b;'>7,500+</div>
            <div style='color: #a0a0a0; margin-top: 8px; font-size: 0.95rem;'>Training Records</div>
        </div>
        <div style='text-align: center; padding: 20px;'>
            <div style='font-size: 2.5rem; margin-bottom: 12px;'>🗺️</div>
            <div style='font-weight: 700; font-size: 1.5rem; color: #8b5cf6;'>36</div>
            <div style='color: #a0a0a0; margin-top: 8px; font-size: 0.95rem;'>States & UTs</div>
        </div>
        <div style='text-align: center; padding: 20px;'>
            <div style='font-size: 2.5rem; margin-bottom: 12px;'>🤖</div>
            <div style='font-weight: 700; font-size: 1.5rem; color: #10b981;'>R² 0.89</div>
            <div style='color: #a0a0a0; margin-top: 8px; font-size: 0.95rem;'>Model Accuracy</div>
        </div>
        <div style='text-align: center; padding: 20px;'>
            <div style='font-size: 2.5rem; margin-bottom: 12px;'>🔍</div>
            <div style='font-weight: 700; font-size: 1.5rem; color: #3b82f6;'>XAI</div>
            <div style='color: #a0a0a0; margin-top: 8px; font-size: 0.95rem;'>Explainable AI</div>
        </div>
        <div style='text-align: center; padding: 20px;'>
            <div style='font-size: 2.5rem; margin-bottom: 12px;'>🚗</div>
            <div style='font-weight: 700; font-size: 1.5rem; color: #f59e0b;'>5</div>
            <div style='color: #a0a0a0; margin-top: 8px; font-size: 0.95rem;'>Vehicle Categories</div>
        </div>
        <div style='text-align: center; padding: 20px;'>
            <div style='font-size: 2.5rem; margin-bottom: 12px;'>🌦️</div>
            <div style='font-weight: 700; font-size: 1.5rem; color: #22c55e;'>4</div>
            <div style='color: #a0a0a0; margin-top: 8px; font-size: 0.95rem;'>Weather Conditions</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Use Cases
st.markdown("## 💼 Target Users & Applications")
use1, use2, use3 = st.columns(3, gap="large")

with use1:
    st.markdown("""
    <div class='glass-card'>
        <div style='font-size: 2.5rem; margin-bottom: 16px;'>🚓</div>
        <h3 style='margin: 0 0 12px 0;'>Traffic Authorities</h3>
        <p style='color: #b0b0b0; line-height: 1.6;'>
            Deploy preventive patrols in high-risk zones, optimize resource allocation, 
            and implement data-driven safety interventions.
        </p>
    </div>
    """, unsafe_allow_html=True)

with use2:
    st.markdown("""
    <div class='glass-card'>
        <div style='font-size: 2.5rem; margin-bottom: 16px;'>🏛️</div>
        <h3 style='margin: 0 0 12px 0;'>Policy Makers</h3>
        <p style='color: #b0b0b0; line-height: 1.6;'>
            Design evidence-based road safety policies, identify infrastructure gaps, 
            and track effectiveness of safety campaigns.
        </p>
    </div>
    """, unsafe_allow_html=True)

with use3:
    st.markdown("""
    <div class='glass-card'>
        <div style='font-size: 2.5rem; margin-bottom: 16px;'>📱</div>
        <h3 style='margin: 0 0 12px 0;'>Fleet Operators</h3>
        <p style='color: #b0b0b0; line-height: 1.6;'>
            Assess route risks before dispatch, optimize driver schedules based on 
            risk profiles, and reduce insurance claims.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p style='font-size: 1rem; margin: 0 0 8px 0;'>
        <strong style='color: #ff4b4b;'>Vehicle Collision Analysis Engine</strong> | Powered by Advanced AI & MoRTH Standards
    </p>
    <p style='font-size: 0.85rem; opacity: 0.7; margin: 0;'>
        Built with Streamlit, XGBoost, Scikit-Learn, Plotly & Pandas | Python 3.11
    </p>
</div>
""", unsafe_allow_html=True)