from fastapi import FastAPI, HTTPException
from schema import UserInput, prediction_output, BatchInput, BatchPredictionOutput
import numpy as np
import joblib

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Detects fraudulent transactions using XGBoost",
    version="1.0.0"
)

try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    raise RuntimeError(f'failed ot load model {e}')

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
THRESHOLD = 0.35

def run_prediction(data_dict: dict) -> dict:
    input_arr = np.array([data_dict[col] for col in FEATURE_ORDER]).reshape(1, -1)
 
    # Scale Time (index 0) and Amount (index 29)
    input_arr[:, [0, 29]] = scaler.transform(input_arr[:, [0, 29]])
 
    prob = float(model.predict_proba(input_arr)[0][1])
    pred = int(prob >= THRESHOLD)
 
    return {
        "prediction": pred,
        "probability": round(prob, 4),
        "label": "FRAUD" if pred == 1 else "NORMAL"
    }

@app.get('/')
def root():
    return {'message': 'Credit Card detection api is running'}

@app.get("/health")
def health():
    return {'status':'ok', 'model':'XGBoost', 'version':'1.0.0'}

@app.post("/predict", response_model=prediction_output)
def predict(data: UserInput):
    """Predict fraud for a single transaction."""
    try:
        return run_prediction(data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(data: BatchInput):
    """Predict fraud for multiple transactions at once."""
    try:
        results = [run_prediction(tx.model_dump()) for tx in data.transactions]
        fraud_count = sum(r["prediction"] for r in results)
        return {
            "total": len(results),
            "fraud_count": fraud_count,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))