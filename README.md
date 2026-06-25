# 💳 Credit Card Fraud Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-card-fraud-detector-ajzetz3dxgt7thj6mgrg8y.streamlit.app/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-FF6F00.svg?style=flat&logo=xgboost&logoColor=white)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **end-to-end Machine Learning system** that detects fraudulent credit card transactions using a fine-tuned **XGBoost** classifier. Built with a **FastAPI** REST backend and an interactive **Streamlit** frontend that communicates with the API in real time.

🔗 **Live App**: [credit-card-fraud-detector.streamlit.app](https://credit-card-fraud-detector-wlesjzfkfermsj82gwo9hu.streamlit.app/)
📡 **Live API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🏗️ Architecture
┌─────────────────────┐        HTTP Request        ┌──────────────────────────┐
│                     │  ─── POST /predict ──────► │                          │
│   Streamlit UI      │                             │   FastAPI Backend        │
│   (app.py)          │  ◄── JSON Response ───────  │   (main.py)              │
│                     │                             │                          │
└─────────────────────┘                             └────────────┬─────────────┘
                                                                 │
                                                                 ▼
                                                    ┌──────────────────────────┐
                                                    │  XGBoost Model           │
                                                    │  + StandardScaler        │
                                                    │  (model.pkl/scaler.pkl)  │
                                                    └──────────────────────────┘

Hosting:
  Streamlit UI  ──►  Streamlit Cloud (Free)
  FastAPI       ──►  Run locally with Uvicorn

  ---

## 📌 Key Features

- **🔍 Single Transaction Check** — Enter transaction details for instant fraud probability scoring via API
- **📁 Batch CSV Upload** — Score thousands of transactions at once using `/predict/batch` endpoint
- **🎚️ Adjustable Threshold Slider** — Fine-tune the probability threshold to balance false positives vs missed fraud
- **📊 Interactive Visualizations** — Probability bar charts, distribution histograms, and confusion matrices
- **📡 REST API with Swagger Docs** — Full OpenAPI documentation at `/docs`, testable in browser
- **⚡ Optimized Model** — RandomizedSearchCV + StratifiedKFold tuned XGBoost with custom threshold (0.35)

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | XGBoost (tuned), Scikit-learn, SMOTE |
| **REST API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit + Plotly |
| **Data** | Pandas, NumPy |
| **Serialization** | Joblib |
| **UI Hosting** | Streamlit Cloud |

---

## 📊 Dataset

**[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** — Kaggle

Transactions made by European cardholders in September 2013.

| Feature | Description |
|---|---|
| `Time` | Seconds elapsed since first transaction |
| `V1`–`V28` | PCA-transformed anonymized features |
| `Amount` | Transaction amount in EUR |
| `Class` | Target: 1 = Fraud, 0 = Legitimate |

> Dataset is highly imbalanced — only **0.17%** of 284,807 transactions are fraud.

---

## 🤖 ML Pipeline

Raw Data (284,807 rows)
        │
        ▼
   EDA + Visualization
   (class imbalance, correlation, distributions)
        │
        ▼
   Preprocessing
   (StandardScaler on Time & Amount, Stratified Split)
        │
        ▼
   Baseline Models
   (Logistic Regression, Random Forest, Naive Bayes, XGBoost)
        │
        ▼
   Class Imbalance Handling
   (SMOTE + scale_pos_weight comparison)
        │
        ▼
   Hyperparameter Tuning
   (RandomizedSearchCV + StratifiedKFold, 30 iterations)
        │
        ▼
   Threshold Tuning
   (Optimized at recall ≥ 0.85 on PR Curve → threshold = 0.35)
        │
        ▼
   Final Model: Tuned XGBoost

   ### 📈 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | **add yours** |
| PR-AUC | **add yours** |
| F1-Score (Fraud) | **add yours** |
| Recall (Fraud) | **add yours** |
| Precision (Fraud) | **add yours** |

---

## 📡 API Endpoints

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status check |
| `GET` | `/health` | Health check with model info |
| `POST` | `/predict` | Single transaction prediction |
| `POST` | `/predict/batch` | Batch prediction (multiple transactions) |
| `GET` | `/docs` | Interactive Swagger UI |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 1000, "Amount": 149.62,
    "V1": -1.35, "V2": -0.07, "V3": 2.53,
    "V4": 1.37, "V5": -0.33, "V6": 0.46,
    "V7": 0.23, "V8": 0.09, "V9": 0.36,
    "V10": 0.09, "V11": -0.55, "V12": -0.61,
    "V13": -0.99, "V14": -0.31, "V15": 1.46,
    "V16": -0.47, "V17": 0.20, "V18": 0.02,
    "V19": 0.40, "V20": 0.25, "V21": -0.01,
    "V22": 0.27, "V23": -0.11, "V24": 0.06,
    "V25": 0.12, "V26": -0.18, "V27": 0.13,
    "V28": -0.02
  }'
```

### Example Response

```json
{
  "prediction": 1,
  "probability": 0.9231,
  "label": "FRAUD"
}
```

---

## 📁 Project Structure
credit-card-fraud-detection/
│
├── 📓 notebooks/
│   └── fraud_detection.ipynb     # EDA, training, evaluation
│
├── main.py                        # FastAPI app (2 endpoints)
├── schema.py                      # Pydantic input/output schemas
├── app.py                         # Streamlit frontend
├── requirements.txt
├── .gitignore
└── README.md

> ⚠️ `model.pkl`, `scaler.pkl`, and `creditcard.csv` are not committed to this repo (too large / sensitive). Train your own using the notebook or download from Kaggle.

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add model files
Train the model using `notebooks/fraud_detection.ipynb` and place `model.pkl` and `scaler.pkl` in the root folder.

### 4. Start FastAPI backend
```bash
uvicorn main:app --reload --port 8000
```

### 5. Start Streamlit frontend (new terminal)
```bash
streamlit run app.py
```

Open `http://localhost:8501` for the UI and `http://localhost:8000/docs` for the API.

---

## 🗒️ Notebooks

The `notebooks/fraud_detection.ipynb` covers:

- **EDA** — Class imbalance visualization, correlation heatmap, feature distributions
- **Preprocessing** — Stratified split, StandardScaler on Time & Amount
- **Baseline Models** — 4 models compared with and without SMOTE
- **Hyperparameter Tuning** — RandomizedSearchCV (30 iterations) + StratifiedKFold (5-fold)
- **Threshold Optimization** — PR curve analysis, optimal threshold at recall ≥ 0.85
- **Evaluation** — ROC-AUC, PR-AUC, F1, Confusion Matrix

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
