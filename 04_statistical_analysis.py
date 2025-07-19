
# STEP 4: STATISTICAL ANALYSIS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2_contingency

df = pd.read_csv('clean_traffic_accidents_dataset.csv')

# Descriptive Statistics by Severity
severity_stats = df.groupby('Severity').agg({
    'Fatalities': ['mean', 'sum', 'std'],
    'Injuries': ['mean', 'sum', 'std'],
    'Total_Casualties': ['mean', 'sum', 'std']
})
print("Casualties by Accident Severity:")
print(severity_stats)

# Chi-square test: Is there a statistically significant association between Weather and Severity?
# A p-value < 0.05 suggests the association is not due to random chance.
contingency_table = pd.crosstab(df['Weather_Condition'], df['Severity'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square test (Weather vs Severity):")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Result: {'The association is statistically significant.' if p_value < 0.05 else 'The association is not statistically significant.'}")

# ANOVA test: Is there a statistically significant difference in the mean number of casualties among different vehicle types?
# A p-value < 0.05 suggests that at least one vehicle type has a different mean number of casualties.
vehicle_groups = [df[df['Vehicle_Type'] == vtype]['Total_Casualties'] 
                 for vtype in df['Vehicle_Type'].unique()]
f_stat, p_val = stats.f_oneway(*vehicle_groups)
print(f"\nANOVA (Vehicle Type vs Casualties):")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_val:.4f}")
print(f"Result: {'There is a significant difference in casualties among vehicle types.' if p_val < 0.05 else 'No significant difference in casualties among vehicle types.'}")

# Correlation Analysis
numeric_columns = ['Speed_Limit', 'Estimated_Speed', 'Fatalities', 'Injuries', 'Total_Casualties']
correlation_matrix = df[numeric_columns].corr()
print("\nCorrelation Matrix:")

# Visualize the correlation matrix with a heatmap for better readability
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()
