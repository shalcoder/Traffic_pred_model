
# STEP 2: TEMPORAL ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('clean_traffic_accidents_dataset.csv')

# Convert date and time columns
df['Date'] = pd.to_datetime(df['Date'])
df['Hour'] = pd.to_datetime(df['Time']).dt.hour

# Accidents by hour of day
hourly_accidents = df.groupby('Hour').size()
plt.figure(figsize=(12, 6))
hourly_accidents.plot(kind='bar')
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=0)
plt.show()

# Accidents by day of week
daily_accidents = df['Day_of_Week'].value_counts()
plt.figure(figsize=(10, 6))
daily_accidents.plot(kind='bar')
plt.title('Accidents by Day of Week')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

# Monthly trends
monthly_accidents = df.groupby('Month').size()
plt.figure(figsize=(10, 6))
monthly_accidents.plot(kind='bar')
plt.title('Accidents by Month')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()
