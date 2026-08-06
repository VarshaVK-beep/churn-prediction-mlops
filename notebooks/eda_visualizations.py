import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Churn class balance
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.savefig('notebooks/plots_churn_distribution.png')
plt.close()

# Tenure vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Churn', y='tenure')
plt.title('Tenure by Churn Status')
plt.savefig('notebooks/plots_tenure_vs_churn.png')
plt.close()

# Monthly charges vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges by Churn Status')
plt.savefig('notebooks/plots_charges_vs_churn.png')
plt.close()

# Churn rate by contract type
plt.figure(figsize=(8,4))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn by Contract Type')
plt.savefig('notebooks/plots_churn_by_contract.png')
plt.close()

print('Plots saved to notebooks/')
