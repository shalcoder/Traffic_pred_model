import streamlit as st
from utils.ui_components import get_base_css, create_hero_section, create_nav_bar, create_back_button
import random
import time

st.set_page_config(page_title="Live Operations", layout="wide", page_icon="📡")

# Inject Premium CSS
st.markdown(get_base_css(), unsafe_allow_html=True)
st.markdown(create_back_button(), unsafe_allow_html=True)

# Back to Home Button
if st.button("← Back to Home", key="back_home"):
    st.switch_page("Home.py")

# Navigation Bar
st.markdown(create_nav_bar("Live Operations Center"), unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div style='text-align: center; padding: 32px 20px; margin-bottom: 32px;'>
    <h1 style='font-size: 2.2rem; margin: 0;'>📡 Live Operations Center</h1>
    <p style='font-size: 1rem; color: #b0b0b0; margin-top: 12px;'>Real-Time Incident Monitoring & Command Control (Simulated)</p>
</div>
""", unsafe_allow_html=True)

# Connection Status
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%); 
            padding: 16px 24px; border-radius: 12px; border-left: 4px solid #10b981; margin-bottom: 32px;'>
    <div style='display: flex; align-items: center; gap: 12px;'>
        <div style='width: 12px; height: 12px; background: #10b981; border-radius: 50%; box-shadow: 0 0 12px #10b981;'></div>
        <div>
            <strong style='color: #10b981; font-size: 1.1rem;'>SYSTEM ACTIVE</strong>
            <span style='color: #a0a0a0; margin-left: 16px; font-size: 0.95rem;'>
                Connected to National Highway Authority Data Stream [SECURE]
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Live Metrics Dashboard
st.markdown("### 📊 System Status Metrics")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Active Sensors", "12,840", "+34 ↑", delta_color="normal")
with c2:
    st.metric("Events (Last Hr)", "45", "-2 ↓", delta_color="inverse")
with c3:
    st.metric("Avg Response Time", "8.2 min", "-0.5 min", delta_color="inverse")
with c4:
    st.metric("Network Uptime", "99.99%", "Stable")
with c5:
    st.metric("Critical Alerts", "3", "+1 ↑")

st.markdown("<br>", unsafe_allow_html=True)

# Main Content: Feed + Details
col_feed, col_map = st.columns([2, 1], gap="large")

with col_feed:
    st.markdown("### 🚨 Real-Time Incident Feed")
    
    refresh_btn = st.button("🔄 Refresh Live Stream", use_container_width=True)
    
    # Generate simulated alerts
    cities = ["Mumbai NH-48", "Delhi Eastern Peripheral", "Bangalore ORR", "Chennai ECR", 
              "Kolkata EM Bypass", "Hyderabad Outer Ring", "Pune Expressway", "Ahmedabad SG Highway"]
    types = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    
    msgs = {
        "CRITICAL": [
            "Multi-vehicle collision reported. Medical assistance dispatched.",
            "Major pile-up detected. Road closure in effect.",
            "Serious injury accident. Air ambulance requested."
        ],
        "HIGH": [
            "Vehicle breakdown causing 2km traffic backlog.",
            "Overturned truck blocking 2 lanes. Recovery in progress.",
            "Fire incident reported. Emergency services on scene."
        ],
        "MEDIUM": [
            "Visibility dropping below 50m due to fog. Caution advised.",
            "Minor collision cleared. Traffic resuming slowly.",
            "Debris on roadway. Cleanup crew dispatched."
        ],
        "LOW": [
            "Slow-moving traffic detected.",
            "Road maintenance in progress. Expect delays.",
            "Weather conditions improving."
        ]
    }
    
    # Display 6 recent alerts
    for i in range(6):
        t_type = random.choice(types)
        time_ago = random.randint(1, 45)
        time_unit = "min" if time_ago > 2 else "sec" if time_ago == 1 else "mins"
        
        alert = {
            "time": f"{time_ago} {time_unit} ago" if time_ago > 1 else "Just now",
            "loc": random.choice(cities),
            "type": t_type,
            "msg": random.choice(msgs[t_type])
        }
        
        if alert['type'] == 'CRITICAL':
            color = "#dc2626"
            bg_color = "rgba(220, 38, 38, 0.08)"
            icon = "🚨"
        elif alert['type'] == 'HIGH':
            color = "#f59e0b"
            bg_color = "rgba(245, 158, 11, 0.08)"
            icon = "⚠️"
        elif alert['type'] == 'MEDIUM':
            color = "#3b82f6"
            bg_color = "rgba(59, 130, 246, 0.08)"
            icon = "ℹ️"
        else:
            color = "#22c55e"
            bg_color = "rgba(34, 197, 94, 0.08)"
            icon = "✓"
        
        st.markdown(f"""
        <div style='padding: 16px; border-left: 4px solid {color}; background: {bg_color}; 
                    margin-bottom: 12px; border-radius: 8px; transition: all 0.3s ease;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <span style='font-size: 1.3rem;'>{icon}</span>
                    <strong style='font-size: 1.05rem;'>📍 {alert['loc']}</strong>
                </div>
                <small style='color: #888; font-size: 0.9rem;'>{alert['time']}</small>
            </div>
            <div style='display: flex; gap: 12px; align-items: center;'>
                <span style='background: {color}; color: white; padding: 4px 12px; 
                            border-radius: 20px; font-size: 0.85rem; font-weight: 700;'>
                    {alert['type']}
                </span>
                <span style='color: #d0d0d0; font-size: 0.95rem;'>{alert['msg']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_map:
    st.markdown("### 📹 Live CCTV Feeds")
    
    # Simulated CCTV thumbnails
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Traffic_jam_in_Delhi.jpg/640px-Traffic_jam_in_Delhi.jpg", 
             caption="📹 CAM-04: Delhi NH-48 (LIVE)", use_container_width=True)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/Traffic_at_Hebbal_Flyover.jpg", 
             caption="📹 CAM-12: Bangalore Hebbal (LIVE)", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Stats Card
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.02); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05);'>
        <h4 style='margin: 0 0 16px 0; font-size: 1.1rem;'>⚡ Quick Stats</h4>
        <div style='display: flex; justify-content: space-between; margin-bottom: 12px;'>
            <span style='color: #a0a0a0;'>Cameras Online</span>
            <strong style='color: #10b981;'>127/130</strong>
        </div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 12px;'>
            <span style='color: #a0a0a0;'>Coverage Area</span>
            <strong>~24,000 km</strong>
        </div>
        <div style='display: flex; justify-content: space-between;'>
            <span style='color: #a0a0a0;'>AI Detection Rate</span>
            <strong style='color: #3b82f6;'>96.4%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer: Emergency Contacts
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("### 📞 Emergency Contacts")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**🚓 Highway Patrol**<br>1800-266-1234", unsafe_allow_html=True)
with col2:
    st.markdown("**🚑 Emergency Medical**<br>102 / 108", unsafe_allow_html=True)
with col3:
    st.markdown("**🚒 Fire Services**<br>101", unsafe_allow_html=True)
with col4:
    st.markdown("**📞 Police Control**<br>100", unsafe_allow_html=True)
