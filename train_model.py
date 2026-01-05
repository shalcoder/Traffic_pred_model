import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_hybrid_model():
    print("Loading expanded dataset...")
    df = pd.read_csv('traffic_accidents_india_standardized.csv')

    # Convert Time to Hour
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour
    
    # Feature Engineering
    encoders = {}
    cat_cols = ['State', 'City', 'Weather_Condition', 'Vehicle_Type', 'Day_of_Week', 'Month']
    
    for col in cat_cols:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col].astype(str))
    
    # Define Target: Numerical Risk Score (Fatalities*3 + Injuries)
    df['Risk_Score'] = (df['Fatalities'] * 3) + df['Injuries']
    
    X = df[['State', 'Hour', 'Weather_Condition', 'Vehicle_Type', 'Day_of_Week']]
    y = df['Risk_Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Hybrid Model (Random Forest + XGBoost)...")
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    
    hybrid_model = VotingRegressor([('rf', rf), ('xgb', xgb)])
    hybrid_model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = hybrid_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Trained. RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
    
    # Save Model and Encoders
    joblib.dump(hybrid_model, 'traffic_hybrid_model.pkl')
    joblib.dump(encoders, 'feature_encoders.pkl')
    print("Model and Encoders saved successfully.")

if __name__ == "__main__":
    train_hybrid_model()
