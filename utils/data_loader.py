import pandas as pd
import streamlit as st
import os

@st.cache_data(ttl=300)
def load_and_preprocess_data():
    """Load and preprocess traffic accident data"""
    # Fix path relative to the root execution
    file_path = os.path.join("data", "traffic_accidents_india_standardized.csv")
    
    try:
        data = pd.read_csv(file_path)
        
        # Enhanced datetime processing
        if 'Time' in data.columns:
            data['DateTime'] = pd.to_datetime(data['Time'], errors='coerce')
            data['Hour'] = data['DateTime'].dt.hour
            
        data['DayOfWeek'] = data.get('Day_of_Week', 'Unknown')
        
        # Severity Mapping
        severity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        if 'Severity' in data.columns:
            data['SeverityScore'] = data['Severity'].map(severity_map).fillna(1)
            
        # Time Categories
        def categorize_time(h):
            if 6 <= h < 12: return 'Morning'
            elif 12 <= h < 18: return 'Afternoon'
            elif 18 <= h < 24: return 'Evening'
            else: return 'Night'
            
        if 'Hour' in data.columns:
            data['TimeCategory'] = data['Hour'].apply(categorize_time)
            
        return data
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()
