import pandas as pd
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Check missing values
print('Missing values per column:')
print(df.isnull().sum())

# Check churn class balance
print('\nChurn distribution:')
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True))
