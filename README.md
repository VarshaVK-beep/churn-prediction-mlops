# Churn Prediction MLOps Pipeline
End-to-end ML project: data pipeline, model training, FastAPI serving, Docker deployment.

## API Demo

The trained model is served via a FastAPI endpoint (\/predict\). Below is a live test through the interactive Swagger UI, showing a customer's data in and the model's churn prediction + probability score out.

![API Demo](docs/api_demo_screenshot.png)

**Example request/response:**
- Input: customer attributes (tenure, contract type, monthly charges, etc.)
- Output: \churn_prediction\ (true/false) and \churn_probability\ (0-1 confidence score)

Run it locally:
\\\ash
python -m uvicorn api.main:app --reload
\\\
Then visit \http://127.0.0.1:8000/docs\ to test interactively.
