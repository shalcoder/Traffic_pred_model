
# STEP 5: SIMPLE MACHINE LEARNING MODEL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv('clean_traffic_accidents_dataset.csv')

# Prepare features for modeling
# Ensure 'Time' is a datetime object to extract the hour
df['Hour'] = pd.to_datetime(df['Time']).dt.hour

# Select features and target
features_df = df[['Hour', 'Speed_Limit', 'Estimated_Speed', 'Weather_Condition', 'Vehicle_Type', 'Road_Type']]
target = df['Severity']

# Handle potential missing values
features_df.fillna(0, inplace=True)

# One-Hot Encode categorical variables. This is better than LabelEncoder for nominal features.
X = pd.get_dummies(features_df, columns=['Weather_Condition', 'Vehicle_Type', 'Road_Type'], drop_first=True)
y = target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

print("\nFeature Importance:")
print(importance_df)
