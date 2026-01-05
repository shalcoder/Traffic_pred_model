import joblib
import pandas as pd
import os
import streamlit as st

def load_model_artifacts():
    try:
        model_path = os.path.join("models", "traffic_hybrid_model.pkl")
        encoder_path = os.path.join("models", "feature_encoders.pkl")
        
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        return model, encoders
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def predict_risk_score(model, encoders, state, hour, weather, vehicle, day):
    # Create input DF
    input_data = pd.DataFrame([{
        'State': state,
        'Hour': hour,
        'Weather_Condition': weather,
        'Vehicle_Type': vehicle,
        'Day_of_Week': day
    }])
    
    # Process Encoders
    for col, encoder in encoders.items():
        if col in input_data.columns:
            try:
                # Handle unknown labels
                input_data[col] = input_data[col].map(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                input_data[col] = encoder.transform(input_data[col])
            except Exception:
                input_data[col] = 0 # Fallback
                
    # Predict
    try:
        prediction = model.predict(input_data)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return 0
