import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append('src')
from preprocessing import load_and_clean_data, encode_categoricals

df = load_and_clean_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_encoded = encode_categoricals(df)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)
print('Saved to data/processed/')
