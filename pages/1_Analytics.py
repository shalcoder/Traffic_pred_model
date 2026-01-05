import streamlit as st
import pandas as pd
from utils.data_loader import load_and_preprocess_data
from utils.visuals import render_kpi_cards, render_map, render_heatmap, render_advanced_charts
from utils.ui_components import get_base_css, create_hero_section, create_nav_bar, create_back_button
import plotly.express as px

st.set_page_config(page_title="Analytics Hub", layout="wide", page_icon="📊")

# Inject Premium CSS
st.markdown(get_base_css(), unsafe_allow_html=True)
st.markdown(create_back_button(), unsafe_allow_html=True)

# Back to Home Button
if st.button("← Back to Home", key="back_home"):
    st.switch_page("Home.py")

# Navigation Bar
st.markdown(create_nav_bar("Analytics Hub"), unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div style='text-align: center; padding: 32px 20px; margin-bottom: 32px;'>
    <h1 style='font-size: 2.2rem; margin: 0;'>📊 Analytics Dashboard</h1>
    <p style='font-size: 1rem; color: #b0b0b0; margin-top: 12px;'>Historical Traffic Insights & Pattern Recognition</p>
</div>
""", unsafe_allow_html=True)

# Load Data
df = load_and_preprocess_data()
if df.empty:
    st.error("⚠️ Data could not be loaded. Please ensure the dataset exists.")
    st.stop()

# Sidebar Filters
with st.sidebar:
    st.markdown("### 🔍 Intelligent Filters")
    st.markdown("---")
    
    # State Filter
    all_states = sorted(df['State'].unique())
    sel_states = st.multiselect(
        "📍 Select States/UTs", 
        all_states, 
        default=all_states[:3],
        help="Filter data by geographical region"
    )
    
    # City Filter (Dynamic based on selected states)
    if sel_states:
        available_cities = sorted(df[df['State'].isin(sel_states)]['City'].unique())
    else:
        available_cities = sorted(df['City'].unique())
    
    sel_cities = st.multiselect(
        "🏙️ Select Cities",
        available_cities,
        default=[],
        help="Filter by specific cities (leave empty for all)"
    )
    
    # Severity Filter
    severities = st.multiselect(
        "⚠️ Severity Levels",
        ["Low", "Medium", "High", "Critical"],
        default=["High", "Critical"],
        help="Focus on specific severity levels"
    )
    
    # Time Filter
    if 'Hour' in df.columns:
        time_range = st.slider(
            "🕐 Time of Day (24h)",
            0, 23, (0, 23),
            help="Filter accidents by hour"
        )

# Apply Filters
filtered_df = df.copy()
if sel_states:
    filtered_df = filtered_df[filtered_df['State'].isin(sel_states)]
if sel_cities:
    filtered_df = filtered_df[filtered_df['City'].isin(sel_cities)]
if severities:
    filtered_df = filtered_df[filtered_df['Severity'].isin(severities)]
if 'Hour' in df.columns:
    filtered_df = filtered_df[(filtered_df['Hour'] >= time_range[0]) & (filtered_df['Hour'] <= time_range[1])]

if filtered_df.empty:
    st.warning("⚠️ No data matches your filters. Try adjusting the selection.")
    st.stop()

# KPI Section
st.markdown("### 📈 Key Performance Indicators")
render_kpi_cards(filtered_df)

st.markdown("<br>", unsafe_allow_html=True)

# Main Analysis Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Geospatial Analysis", "📅 Temporal Patterns", "🎯 Risk Drivers", "📊 Advanced Metrics"])

with tab1:
    render_map(filtered_df)
    
    # Additional Stats below map
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("States Covered", filtered_df['State'].nunique(), "36 Total")
    with col2:
        st.metric("Cities Analyzed", filtered_df['City'].nunique())
    with col3:
        hottest_state = filtered_df.groupby('State').size().idxmax()
        st.metric("Highest Activity", hottest_state)

with tab2:
    render_heatmap(filtered_df)
    
    # Time series analysis
    st.subheader("📈 Monthly Trend Analysis")
    if 'Month' in filtered_df.columns:
        monthly = filtered_df.groupby('Month').size().reset_index(name='Count')
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        monthly['Month'] = pd.Categorical(monthly['Month'], categories=month_order, ordered=True)
        monthly = monthly.sort_values('Month')
        
        fig_line = px.line(
            monthly, x='Month', y='Count',
            markers=True, 
            title="Accident Frequency by Month",
            line_shape='spline'
        )
        fig_line.update_traces(line_color='#ff4b4b', line_width=3)
        st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚗 Vehicle-wise Distribution")
        v_counts = filtered_df['Vehicle_Type'].value_counts()
        fig_pie = px.pie(
            names=v_counts.index, 
            values=v_counts.values, 
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("🌦️ Weather Impact")
        w_counts = filtered_df['Weather_Condition'].value_counts()
        fig_bar = px.bar(
            x=w_counts.index, 
            y=w_counts.values,
            color=w_counts.values,
            color_continuous_scale='Reds',
            labels={'x': 'Weather', 'y': 'Incidents'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    # Render advanced analytics
    render_advanced_charts(filtered_df)

# Footer Stats
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Data Coverage", f"{len(df):,} Records")
with col2:
    st.metric("Filtered View", f"{len(filtered_df):,} Records")
with col3:
    fatality_rate = (filtered_df['Fatalities'].sum() / len(filtered_df) * 100)
    st.metric("Fatality Rate", f"{fatality_rate:.2f}%")
with col4:
    avg_injuries = filtered_df['Injuries'].mean()
    st.metric("Avg Injuries/Event", f"{avg_injuries:.1f}")
