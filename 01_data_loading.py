
# STEP 1: DATA LOADING AND INITIAL EXPLORATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('clean_traffic_accidents_dataset.csv')

# Basic exploration
print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:", df.dtypes)
print("\nMissing Values:", df.isnull().sum())

# Display first few records
print("\nFirst 5 records:")
print(df.head())

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())
