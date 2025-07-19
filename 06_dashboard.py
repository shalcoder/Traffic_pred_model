
# STEP 6: INTERACTIVE DASHBOARD WITH STREAMLIT
# Save this as 'dashboard.py' and run with: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('clean_traffic_accidents_dataset.csv')

df = load_data()

# Dashboard title
st.title('🚗 Traffic Accident Analysis Dashboard')
st.sidebar.header('Filters')

# Sidebar filters
cities = st.sidebar.multiselect('Select Cities', df['City'].unique(), default=df['City'].unique()[:3])
severity = st.sidebar.selectbox('Severity Level', ['All'] + list(df['Severity'].unique()))

# Filter data
filtered_df = df[df['City'].isin(cities)]
if severity != 'All':
    filtered_df = filtered_df[filtered_df['Severity'] == severity]

# Main dashboard
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Total Accidents', len(filtered_df))
with col2:
    st.metric('Total Fatalities', filtered_df['Fatalities'].sum())
with col3:
    st.metric('Total Injuries', filtered_df['Injuries'].sum())

# Charts
st.subheader('Accidents by Hour')
hourly_data = filtered_df.groupby(pd.to_datetime(filtered_df['Time']).dt.hour).size()
fig1 = px.line(x=hourly_data.index, y=hourly_data.values, title='Hourly Accident Pattern')
st.plotly_chart(fig1, use_container_width=True)

st.subheader('Weather Impact')
weather_data = filtered_df.groupby('Weather_Condition').size()
fig2 = px.pie(values=weather_data.values, names=weather_data.index, title='Accidents by Weather')
st.plotly_chart(fig2, use_container_width=True)

st.subheader('Vehicle Type Distribution')
vehicle_data = filtered_df.groupby('Vehicle_Type').size()
fig3 = px.bar(x=vehicle_data.index, y=vehicle_data.values, title='Accidents by Vehicle Type')
st.plotly_chart(fig3, use_container_width=True)

# Raw data
if st.checkbox('Show Raw Data'):
    st.write(filtered_df)
