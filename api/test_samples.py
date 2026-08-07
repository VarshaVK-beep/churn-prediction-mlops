import requests
import json

samples = [
    {
        "name": "High-risk customer (new, month-to-month, high charges)",
        "data": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0
        }
    },
    {
        "name": "Low-risk customer (long tenure, 2-year contract)",
        "data": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 45.0, "TotalCharges": 2700.0
        }
    },
    {
        "name": "Medium-risk customer (moderate tenure, one-year contract)",
        "data": {
            "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
            "tenure": 20, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.0, "TotalCharges": 1300.0
        }
    }
]

results = []
for sample in samples:
    response = requests.post('http://127.0.0.1:8000/predict', json=sample['data'])
    result = {'scenario': sample['name'], 'input': sample['data'], 'output': response.json()}
    results.append(result)
    print(sample['name'])
    print(response.json())
    print()

with open('docs/sample_predictions.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Saved to docs/sample_predictions.json')
