import pandas as pd
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Average tenure by churn status
print('Avg tenure by churn:')
print(df.groupby('Churn')['tenure'].mean())

# Average monthly charges by churn status
print('\nAvg monthly charges by churn:')
print(df.groupby('Churn')['MonthlyCharges'].mean())

# Churn rate by contract type
print('\nChurn rate by contract type:')
print(df.groupby('Contract')['Churn'].value_counts(normalize=True))
