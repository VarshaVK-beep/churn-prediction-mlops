import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import encode_categoricals

model = joblib.load('src/best_model.pkl')
TRAIN_COLUMNS = pd.read_csv('data/processed/X_train.csv').columns.tolist()

customers = [
    {'name': 'High-risk (new, month-to-month)', 'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No', 'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check', 'MonthlyCharges': 95.0, 'TotalCharges': 190.0},
    {'name': 'Low-risk (long tenure, 2yr contract)', 'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes', 'tenure': 60, 'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'DSL', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes', 'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Two year', 'PaperlessBilling': 'No', 'PaymentMethod': 'Bank transfer (automatic)', 'MonthlyCharges': 45.0, 'TotalCharges': 2700.0},
    {'name': 'Medium-risk (1yr contract)', 'gender': 'Female', 'SeniorCitizen': 1, 'Partner': 'No', 'Dependents': 'No', 'tenure': 20, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes', 'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'No', 'Contract': 'One year', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Credit card (automatic)', 'MonthlyCharges': 65.0, 'TotalCharges': 1300.0}
]

names = []
probs = []
for c in customers:
    name = c.pop('name')
    df = pd.DataFrame([c])
    df_encoded = encode_categoricals(df)
    df_encoded = df_encoded.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    prob = model.predict_proba(df_encoded)[0][1]
    names.append(name)
    probs.append(prob)
    print(f'{name}: churn probability = {prob:.2f}')

plt.figure(figsize=(9, 5))
colors = ['crimson' if p > 0.5 else 'seagreen' for p in probs]
bars = plt.barh(names, probs, color=colors)
plt.xlabel('Churn Probability')
plt.title('Model Predictions on Sample Customers')
plt.xlim(0, 1)
for i, v in enumerate(probs):
    plt.text(v + 0.02, i, f'{v:.2f}', va='center')
plt.tight_layout()
plt.savefig('notebooks/plots_sample_customer_predictions.png', dpi=150)
plt.close()
print('Saved: notebooks/plots_sample_customer_predictions.png')
