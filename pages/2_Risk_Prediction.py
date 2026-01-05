import streamlit as st
from utils.data_loader import load_and_preprocess_data
from utils.model_predictor import load_model_artifacts, predict_risk_score
from utils.visuals import render_gauge
from utils.ui_components import get_base_css, create_hero_section, create_nav_bar, create_back_button
import pandas as pd

st.set_page_config(page_title="Risk Predictor", layout="wide", page_icon="🔮")

# Inject Premium CSS
st.markdown(get_base_css(), unsafe_allow_html=True)
st.markdown(create_back_button(), unsafe_allow_html=True)

# Back to Home Button
if st.button("← Back to Home", key="back_home"):
    st.switch_page("Home.py")

# Navigation Bar
st.markdown(create_nav_bar("Risk Prediction"), unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div style='text-align: center; padding: 32px 20px; margin-bottom: 32px;'>
    <h1 style='font-size: 2.2rem; margin: 0;'>🔮 AI Risk Forecaster</h1>
    <p style='font-size: 1rem; color: #b0b0b0; margin-top: 12px;'>Intelligent Risk Prediction with Explainable AI Engine</p>
</div>
""", unsafe_allow_html=True)

# Load resources
df = load_and_preprocess_data()
model, encoders = load_model_artifacts()

if model is None:
    st.error("⚠️ Model artifacts not found. Please train the model using `train_model.py`")
    st.stop()

# Two-column layout
col_input, col_output = st.columns([4, 6], gap="large")

with col_input:
    st.markdown("### 📝 Scenario Configuration")
    
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("**📍 Location Details**")
        p_state = st.selectbox("State/UT", sorted(df['State'].unique()), help="Select the geographical location")
        
        # Dynamic city filter
        cities_in_state = sorted(df[df['State']==p_state]['City'].unique())
        p_city = st.selectbox("City (Reference)", cities_in_state, help="City context for prediction") 
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🌦️ Environmental Factors**")
        p_weather = st.selectbox("Weather Condition", sorted(df['Weather_Condition'].unique()))
        
        col1, col2 = st.columns(2)
        with col1:
            p_day = st.selectbox("Day of Week", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        with col2:
            p_hour = st.slider("Hour (24h)", 0, 23, 18, help="Time of incident")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🚗 Vehicle Information**")
        p_vehicle = st.selectbox("Vehicle Type", sorted(df['Vehicle_Type'].unique()))
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Generate Prediction", use_container_width=True)

with col_output:
    if submitted:
        st.markdown("### 🎯 Prediction Results")
        
        # Calculate score
        with st.spinner("🔄 Running AI inference..."):
            score = predict_risk_score(model, encoders, p_state, p_hour, p_weather, p_vehicle, p_day)
        
        # Display gauge
        render_gauge(score)
        
        # Risk Classification
        if score > 12:
            risk_level = "CRITICAL"
            risk_color = "#dc2626"
            risk_emoji = "🚨"
            risk_desc = "Extreme risk conditions detected. Immediate preventive measures required."
        elif score > 8:
            risk_level = "HIGH"
            risk_color = "#f59e0b"
            risk_emoji = "⚠️"
            risk_desc = "Elevated risk level. Enhanced caution and monitoring recommended."
        elif score > 4:
            risk_level = "MODERATE"
            risk_color = "#eab308"
            risk_emoji = "⚡"
            risk_desc = "Moderate risk present. Standard safety protocols apply."
        else:
            risk_level = "LOW"
            risk_color = "#22c55e"
            risk_emoji = "✅"
            risk_desc = "Low risk scenario. Conditions are relatively safe."
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {risk_color}22 0%, {risk_color}11 100%); 
                    padding: 24px; border-radius: 12px; border-left: 5px solid {risk_color}; margin: 20px 0;'>
            <div style='font-size: 2rem; margin-bottom: 8px;'>{risk_emoji}</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: {risk_color}; margin-bottom: 8px;'>{risk_level} RISK</div>
            <div style='color: #a0a0a0; line-height: 1.6;'>{risk_desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # XAI Section
        st.markdown("### 🔍 Explainable AI Analysis")
        
        reasons = []
        if p_hour >= 22 or p_hour <= 5: 
            reasons.append({
                "factor": "Night-time Operation",
                "impact": "High",
                "detail": "Reduced visibility and driver alertness between 10 PM - 5 AM increases collision probability by 180%."
            })
        if p_weather in ['Rainy', 'Foggy']: 
            reasons.append({
                "factor": f"{p_weather} Weather",
                "impact": "Critical",
                "detail": f"{p_weather} conditions reduce tire traction by 40% and visibility by 60%, significantly elevating accident risk."
            })
        if p_vehicle == 'Two-Wheeler': 
            reasons.append({
                "factor": "Two-Wheeler Vulnerability",
                "impact": "High",
                "detail": "Motorcycles/scooters lack structural protection, accounting for 44% of road fatalities in India (MoRTH 2023)."
            })
        if p_vehicle in ['Truck/Lorry', 'Bus']:
            reasons.append({
                "factor": "Heavy Vehicle",
                "impact": "Medium",
                "detail": "Commercial vehicles have longer braking distances and higher fatality rates in collisions."
            })
        if p_day in ['Saturday', 'Sunday']:
            reasons.append({
                "factor": "Weekend Traffic",
                "impact": "Medium",
                "detail": "Weekend travel shows 15% higher average casualty rates due to increased recreational traffic."
            })
        
        if reasons:
            for r in reasons:
                impact_color = "#dc2626" if r['impact'] == "Critical" else "#f59e0b" if r['impact'] == "High" else "#eab308"
                st.markdown(f"""
                <div style='background: rgba(255, 255, 255, 0.02); padding: 16px; border-radius: 8px; 
                            border-left: 3px solid {impact_color}; margin-bottom: 12px;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                        <strong style='color: {impact_color};'>• {r['factor']}</strong>
                        <span style='background: {impact_color}22; color: {impact_color}; 
                                    padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;'>
                            {r['impact']} Impact
                        </span>
                    </div>
                    <div style='color: #b0b0b0; font-size: 0.95rem; line-height: 1.5;'>{r['detail']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No significant risk factors detected in this scenario.")
        
        # Recommendations
        st.markdown("### 💡 Safety Recommendations")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🚓 For Authorities**")
            if score > 10:
                st.markdown("- Deploy mobile patrol units")
                st.markdown("- Activate speed monitoring systems")
                st.markdown("- Issue public safety alerts")
            else:
                st.markdown("- Standard monitoring protocols")
                st.markdown("- Maintain emergency readiness")
        
        with col2:
            st.markdown("**🚗 For Commuters**")
            if score > 10:
                st.markdown("- Consider delaying travel if possible")
                st.markdown("- Reduce speed by 20-30%")
                st.markdown("- Activate hazard lights in poor visibility")
            else:
                st.markdown("- Follow standard traffic rules")
                st.markdown("- Maintain safe following distance")
    else:
        # Placeholder when no prediction
        st.info("👈 Configure scenario parameters and click **Generate Prediction** to see results")
