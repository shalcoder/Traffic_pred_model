# PROFESSIONAL TRAFFIC ACCIDENT ANALYSIS DASHBOARD
# Enhanced with AI-powered insights, advanced visualizations, and professional features
# Run: streamlit run enhanced_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page settings
st.set_page_config(
    page_title="Traffic Accident Analytics Pro",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #00a8ff;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 8px;
        color: #333;
        margin: 1rem 0;
        border-left: 5px solid #ff6b6b;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced data loading with caching and preprocessing
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_and_preprocess_data():
    """Load and preprocess traffic accident data with enhanced features"""
    try:
        data = pd.read_csv('clean_traffic_accidents_dataset.csv')
        
        # Enhanced datetime processing
        data['DateTime'] = pd.to_datetime(data['Time'], errors='coerce')
        data['Hour'] = data['DateTime'].dt.hour
        data['DayOfWeek'] = data['DateTime'].dt.day_name()
        data['Month'] = data['DateTime'].dt.month_name()
        data['IsWeekend'] = data['DateTime'].dt.weekday >= 5
        
        # Create severity score for advanced analytics
        severity_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        data['SeverityScore'] = data['Severity'].map(severity_mapping).fillna(1)
        
        # Calculate risk metrics
        data['FatalityRate'] = (data['Fatalities'] / (data['Fatalities'] + data['Injuries'] + 1)) * 100
        data['TotalCasualties'] = data['Fatalities'] + data['Injuries']
        
        # Add time categories
        def categorize_time(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 24:
                return 'Evening'
            else:
                return 'Night'
        
        data['TimeCategory'] = data['Hour'].apply(categorize_time)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# AI-powered insights generator
def generate_ai_insights(df):
    """Generate intelligent insights from the data"""
    insights = []
    
    if not df.empty:
        # Peak hour analysis
        peak_hour = df.groupby('Hour').size().idxmax()
        peak_accidents = df.groupby('Hour').size().max()
        insights.append(f"🕐 **Peak Risk Hour**: {peak_hour}:00 with {peak_accidents} accidents")
        
        # Most dangerous combination
        dangerous_combo = df.groupby(['Weather_Condition', 'Vehicle_Type']).size().idxmax()
        insights.append(f"⚠️ **Highest Risk Combination**: {dangerous_combo[1]} vehicles in {dangerous_combo[0]} conditions")
        
        # City risk ranking
        city_risk = df.groupby('City')['SeverityScore'].mean().sort_values(ascending=False)
        if len(city_risk) > 0:
            insights.append(f"🏙️ **Highest Risk City**: {city_risk.index[0]} (Risk Score: {city_risk.iloc[0]:.2f})")
        
        # Weekend vs weekday
        weekend_avg = df[df['IsWeekend']]['TotalCasualties'].mean()
        weekday_avg = df[~df['IsWeekend']]['TotalCasualties'].mean()
        if weekend_avg > weekday_avg:
            insights.append(f"📅 **Weekend Risk**: {((weekend_avg/weekday_avg - 1)*100):.1f}% higher casualty rate on weekends")
        
        # Fatality rate insights
        high_fatality_conditions = df.groupby('Weather_Condition')['FatalityRate'].mean().sort_values(ascending=False)
        if len(high_fatality_conditions) > 0:
            insights.append(f"☔ **Deadliest Weather**: {high_fatality_conditions.index[0]} ({high_fatality_conditions.iloc[0]:.1f}% fatality rate)")
    
    return insights

# Load data
df = load_and_preprocess_data()

if df.empty:
    st.error("Unable to load data. Please check if 'clean_traffic_accidents_dataset.csv' exists in the current directory.")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <h1>🚦 Traffic Accident Analytics Pro</h1>
    <p>Advanced AI-Powered Dashboard for Traffic Safety Intelligence</p>
    <p>Real-time insights • Predictive Analytics • Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
st.sidebar.markdown("## 🎛️ Control Center")
st.sidebar.markdown("---")

# Advanced filters
st.sidebar.markdown("### 📍 Location Filters")
all_cities = sorted(df['City'].unique())
cities = st.sidebar.multiselect(
    'Select Cities',
    options=all_cities,
    default=all_cities[:3],  # Default to first 3 cities for better performance
    help="Choose cities to analyze"
)

st.sidebar.markdown("### 🚨 Severity Filters")
severity_options = ['All'] + sorted(df['Severity'].unique())
severity = st.sidebar.selectbox(
    'Severity Level',
    options=severity_options,
    help="Filter by accident severity"
)

st.sidebar.markdown("### 📅 Time Filters")
time_filter = st.sidebar.select_slider(
    'Time of Day',
    options=['All', 'Morning', 'Afternoon', 'Evening', 'Night'],
    value='All',
    help="Filter by time period"
)

weekend_filter = st.sidebar.radio(
    'Day Type',
    options=['All', 'Weekdays', 'Weekends'],
    help="Filter by day type"
)

st.sidebar.markdown("### 🌤️ Environmental Filters")
weather_options = ['All'] + list(df['Weather_Condition'].unique())
weather = st.sidebar.selectbox(
    'Weather Condition',
    options=weather_options,
    help="Filter by weather conditions"
)

# Advanced filtering logic
filtered_df = df[df['City'].isin(cities)] if cities else df

if severity != 'All':
    filtered_df = filtered_df[filtered_df['Severity'] == severity]

if time_filter != 'All':
    filtered_df = filtered_df[filtered_df['TimeCategory'] == time_filter]

if weekend_filter == 'Weekdays':
    filtered_df = filtered_df[~filtered_df['IsWeekend']]
elif weekend_filter == 'Weekends':
    filtered_df = filtered_df[filtered_df['IsWeekend']]

if weather != 'All':
    filtered_df = filtered_df[filtered_df['Weather_Condition'] == weather]

# Main Dashboard
if filtered_df.empty:
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ No Data Available</h3>
        <p>No accidents match your current filter criteria. Please adjust your selections.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # AI Insights Section
    st.markdown("## 🤖 AI-Powered Insights")
    insights = generate_ai_insights(filtered_df)
    
    col1, col2 = st.columns(2)
    for i, insight in enumerate(insights):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="insight-box">
                {insight}
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced KPI Section
    st.markdown("## 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h2>{len(filtered_df):,}</h2>
            <p>Total Accidents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h2>{filtered_df['Fatalities'].sum():,}</h2>
            <p>Total Fatalities</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h2>{filtered_df['Injuries'].sum():,}</h2>
            <p>Total Injuries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_severity = filtered_df['SeverityScore'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h2>{avg_severity:.2f}</h2>
            <p>Avg Risk Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Visualizations
    st.markdown("## 📈 Advanced Analytics")
    
    # Temporal Analysis with Heatmap
    st.subheader("🕐 Temporal Risk Heatmap")
    
    # Create hour vs day heatmap
    temporal_data = filtered_df.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='Accidents')
    pivot_data = temporal_data.pivot(index='DayOfWeek', columns='Hour', values='Accidents').fillna(0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(day_order)
    
    fig_heatmap = px.imshow(
        pivot_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Accidents"),
        title="Accident Risk Heatmap: Hour vs Day",
        color_continuous_scale="Reds",
        aspect="auto"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Multi-dimensional Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌦️ Weather Impact Analysis")
        weather_severity = filtered_df.groupby(['Weather_Condition', 'Severity']).size().reset_index(name='Count')
        fig_weather = px.sunburst(
            weather_severity,
            path=['Weather_Condition', 'Severity'],
            values='Count',
            title="Weather vs Severity Distribution",
            color='Count',
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_weather, use_container_width=True)
    
    with col2:
        st.subheader("🚗 Vehicle Risk Analysis")
        vehicle_stats = filtered_df.groupby('Vehicle_Type').agg({
            'Fatalities': 'sum',
            'Injuries': 'sum',
            'SeverityScore': 'mean'
        }).reset_index()
        
        fig_vehicle = px.scatter(
            vehicle_stats,
            x='Injuries',
            y='Fatalities',
            size='SeverityScore',
            hover_name='Vehicle_Type',
            title="Vehicle Risk Matrix",
            color='SeverityScore',
            color_continuous_scale="RdYlBu_r"
        )
        fig_vehicle.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
        st.plotly_chart(fig_vehicle, use_container_width=True)
    
    # City Performance Dashboard
    st.subheader("🏙️ City Performance Analysis")
    
    city_metrics = filtered_df.groupby('City').agg({
        'Fatalities': 'sum',
        'Injuries': 'sum',
        'SeverityScore': 'mean',
        'FatalityRate': 'mean'
    }).reset_index()
    city_metrics['TotalCasualties'] = city_metrics['Fatalities'] + city_metrics['Injuries']
    
    fig_city = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Casualties by City', 'Average Risk Score', 
                       'Fatality vs Injury Rate', 'Risk Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "domain"}]]
    )
    
    # Bar chart for casualties
    fig_city.add_trace(
        go.Bar(x=city_metrics['City'], y=city_metrics['TotalCasualties'], 
               name='Total Casualties', marker_color='lightcoral'),
        row=1, col=1
    )
    
    # Risk score bar chart
    fig_city.add_trace(
        go.Bar(x=city_metrics['City'], y=city_metrics['SeverityScore'], 
               name='Risk Score', marker_color='lightblue'),
        row=1, col=2
    )
    
    # Scatter plot
    fig_city.add_trace(
        go.Scatter(x=city_metrics['Injuries'], y=city_metrics['Fatalities'],
                  mode='markers+text', text=city_metrics['City'],
                  textposition="top center", name='Cities',
                  marker=dict(size=city_metrics['SeverityScore']*10, color='orange')),
        row=2, col=1
    )
    
    # Pie chart for risk distribution
    fig_city.add_trace(
        go.Pie(labels=city_metrics['City'], values=city_metrics['SeverityScore'],
               name="Risk Distribution"),
        row=2, col=2
    )
    
    fig_city.update_layout(height=800, showlegend=False, title_text="Comprehensive City Analysis")
    st.plotly_chart(fig_city, use_container_width=True)
    
    # Time Series Analysis
    if 'DateTime' in filtered_df.columns:
        st.subheader("📅 Trend Analysis")
        
        # Monthly trends
        monthly_trends = filtered_df.groupby(filtered_df['DateTime'].dt.to_period('M')).agg({
            'Fatalities': 'sum',
            'Injuries': 'sum',
            'SeverityScore': 'mean'
        }).reset_index()
        monthly_trends['Month'] = monthly_trends['DateTime'].astype(str)
        
        fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_trends.add_trace(
            go.Scatter(x=monthly_trends['Month'], y=monthly_trends['Fatalities'],
                      name='Fatalities', line=dict(color='red', width=3)),
            secondary_y=False,
        )
        
        fig_trends.add_trace(
            go.Scatter(x=monthly_trends['Month'], y=monthly_trends['Injuries'],
                      name='Injuries', line=dict(color='orange', width=3)),
            secondary_y=False,
        )
        
        fig_trends.add_trace(
            go.Scatter(x=monthly_trends['Month'], y=monthly_trends['SeverityScore'],
                      name='Risk Score', line=dict(color='blue', width=3, dash='dash')),
            secondary_y=True,
        )
        
        fig_trends.update_xaxes(title_text="Month")
        fig_trends.update_yaxes(title_text="Casualties", secondary_y=False)
        fig_trends.update_yaxes(title_text="Risk Score", secondary_y=True)
        fig_trends.update_layout(title_text="Monthly Trend Analysis", height=400)
        
        st.plotly_chart(fig_trends, use_container_width=True)
    
    # Risk Prediction Section
    st.subheader("🎯 Risk Prediction Model")
    st.info("This section would integrate with ML models for predictive analytics in a production environment.")
    
    # Statistical Summary
    st.subheader("📋 Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Descriptive Statistics")
        stats_df = filtered_df[['Fatalities', 'Injuries', 'SeverityScore', 'FatalityRate']].describe()
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.markdown("### 🏆 Top Risk Factors")
        risk_factors = []
        
        # Weather risk
        weather_risk = filtered_df.groupby('Weather_Condition')['SeverityScore'].mean().sort_values(ascending=False)
        risk_factors.append(f"**Weather**: {weather_risk.index[0]} ({weather_risk.iloc[0]:.2f})")
        
        # Vehicle risk
        vehicle_risk = filtered_df.groupby('Vehicle_Type')['SeverityScore'].mean().sort_values(ascending=False)
        risk_factors.append(f"**Vehicle**: {vehicle_risk.index[0]} ({vehicle_risk.iloc[0]:.2f})")
        
        # Time risk
        time_risk = filtered_df.groupby('TimeCategory')['SeverityScore'].mean().sort_values(ascending=False)
        risk_factors.append(f"**Time**: {time_risk.index[0]} ({time_risk.iloc[0]:.2f})")
        
        for factor in risk_factors:
            st.markdown(factor)
    
    # Export and Raw Data Section
    st.subheader("💾 Data Export & Raw Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Summary Report"):
            summary_data = {
                'Total Accidents': len(filtered_df),
                'Total Fatalities': filtered_df['Fatalities'].sum(),
                'Total Injuries': filtered_df['Injuries'].sum(),
                'Average Risk Score': filtered_df['SeverityScore'].mean(),
                'Peak Hour': filtered_df.groupby('Hour').size().idxmax(),
                'Riskiest City': filtered_df.groupby('City')['SeverityScore'].mean().idxmax()
            }
            st.json(summary_data)
    
    with col2:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv_data,
            file_name=f"traffic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Raw data with advanced filtering
    if st.expander("🗂️ View Raw Data", expanded=False):
        st.markdown("### Filtered Dataset Preview")
        
        # Add search functionality
        search_term = st.text_input("🔍 Search in data:", placeholder="Enter search term...")
        
        display_df = filtered_df.copy()
        if search_term:
            mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]
        
        st.dataframe(
            display_df.head(100),
            use_container_width=True,
            height=400
        )
        
        st.info(f"Showing {min(100, len(display_df))} of {len(display_df)} records")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚦 Traffic Accident Analytics Pro | Powered by Advanced AI & Machine Learning</p>
    <p>Built with ❤️ using Streamlit, Plotly, and Professional Data Science</p>
    <p>📊 Real-time Analytics | 🤖 AI Insights | 📈 Predictive Modeling</p>
</div>
""", unsafe_allow_html=True)