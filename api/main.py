import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.preprocessing import encode_categoricals

app = FastAPI(title='Churn Prediction API')

model = joblib.load('src/best_model.pkl')
TRAIN_COLUMNS = pd.read_csv('data/processed/X_train.csv').columns.tolist()

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get('/')
def root():
    return {'status': 'Churn Prediction API is running'}

@app.post('/predict')
def predict(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    df_encoded = encode_categoricals(df)
    df_encoded = df_encoded.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    prediction = model.predict(df_encoded)[0]
    probability = model.predict_proba(df_encoded)[0][1]
    return {
        'churn_prediction': bool(prediction),
        'churn_probability': round(float(probability), 3)
    }
