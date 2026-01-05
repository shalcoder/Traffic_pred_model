import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def render_kpi_cards(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Incidents", f"{len(df):,}")
    with col2:
        st.metric("Total Fatalities", f"{df['Fatalities'].sum():,}")
    with col3:
        st.metric("Injuries Reported", f"{df['Injuries'].sum():,}")
    with col4:
        avg_sev = df['SeverityScore'].mean() if 'SeverityScore' in df else 0
        st.metric("Avg Risk Index", f"{avg_sev:.2f}")

def render_map(df):
    st.subheader("🗺️ Geospatial Risk Hotspots")
    fig_map = px.scatter_mapbox(
        df, 
        lat="Latitude", 
        lon="Longitude", 
        color="Severity",
        size="Injuries", 
        hover_name="City",
        hover_data=["Weather_Condition", "Vehicle_Type", "State"],
        color_discrete_map={"Critical": "#dc2626", "High": "#f59e0b", "Medium": "#eab308", "Low": "#22c55e"},
        zoom=4.2, 
        center={"lat": 22.5, "lon": 78.9}, # India center
        height=450,
        mapbox_style="carto-positron" # Light, clean map
    )
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        mapbox=dict(
            bearing=0,
            pitch=0
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

def render_advanced_charts(df):
    st.subheader("📈 Advanced Impact Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Weather vs. Severity Correlation**")
        fig_sun = px.sunburst(
            df,
            path=['Weather_Condition', 'Severity'],
            values='Injuries',
            color='SeverityScore',
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig_sun, use_container_width=True)
        
    with col2:
        st.write("**Vehicle Risk Profile**")
        v_stats = df.groupby('Vehicle_Type').agg({'Fatalities':'sum', 'Injuries':'sum', 'SeverityScore':'mean'}).reset_index()
        fig_bub = px.scatter(
            v_stats, x="Injuries", y="Fatalities",
            size="SeverityScore", color="SeverityScore",
            hover_name="Vehicle_Type", size_max=60,
            title="Vehicle Type Risk Matrix"
        )
        st.plotly_chart(fig_bub, use_container_width=True)

    st.subheader("🏙️ City Safety Performance")
    city_group = df.groupby('City').agg({'SeverityScore':'mean', 'Accident_ID':'count'}).reset_index()
    city_group = city_group.sort_values('SeverityScore', ascending=False).head(10)
    
    fig_bar = px.bar(
        city_group, x='City', y='SeverityScore',
        color='Accident_ID', title="Top 10 Cities by Risk Score",
        color_continuous_scale='Magma'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

def render_heatmap(df):
    st.subheader("🕐 Temporal Risk Heatmap")
    temporal = df.groupby(['Day_of_Week', 'Hour']).size().reset_index(name='Count')
    pivot = temporal.pivot(index='Day_of_Week', columns='Hour', values='Count').fillna(0)
    
    # Sort Days
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(days)
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Hour", y="Day", color="Accidents"),
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_gauge(value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Severity Score"},
        gauge = {
            'axis': {'range': [None, 20]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 5], 'color': "lightgreen"},
                {'range': [5, 10], 'color': "yellow"},
                {'range': [10, 15], 'color': "orange"},
                {'range': [15, 20], 'color': "red"}],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': value}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
