import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ── API config ────────────────────────────────────────────────────────
# Change this to your EC2 public IP when deployed on AWS
# e.g. "http://13.233.100.200:8000"
API_URL = os.getenv("API_URL", "http://localhost:8000")

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ── API helpers ───────────────────────────────────────────────────────

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def predict_single(row: dict, threshold: float):
    """Call FastAPI /predict for one transaction."""
    try:
        r = requests.post(f"{API_URL}/predict", json=row, timeout=10)
        r.raise_for_status()
        result = r.json()
        # Apply user-chosen threshold on top of returned probability
        prob = result["probability"]
        pred = int(prob >= threshold)
        label = "FRAUD" if pred == 1 else "NORMAL"
        return pred, prob, label
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Cannot connect to API at {API_URL}. Is FastAPI running?")
    except Exception as e:
        raise RuntimeError(str(e))


def predict_batch(df: pd.DataFrame, threshold: float):
    """Call FastAPI /predict/batch for a CSV of transactions."""
    transactions = df[FEATURE_COLUMNS].to_dict(orient="records")
    payload = {"transactions": transactions}
    try:
        r = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        results = data["results"]
        probs = np.array([res["probability"] for res in results])
        # Apply user-chosen threshold
        preds = (probs >= threshold).astype(int)
        return preds, probs

    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Cannot connect to API at {API_URL}. Is FastAPI running?")
    except Exception as e:
        raise RuntimeError(str(e))


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💳 Fraud Detector")
    st.markdown(
        "This app uses a tuned **XGBoost** classifier trained on the "
        "Kaggle *Credit Card Fraud Detection* dataset to flag potentially "
        "fraudulent transactions."
    )

    # API health status
    st.markdown("---")
    st.markdown("**🔌 API Status**")
    if check_api_health():
        st.success(f"Connected ✅\n`{API_URL}`")
    else:
        st.error(f"Not reachable ❌\n`{API_URL}`")
        st.caption("Start FastAPI with:\n`uvicorn main:app --reload`")

    st.markdown("---")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.01,
        help=(
            "Transactions with predicted fraud probability ≥ this value are "
            "flagged as fraud. Lower it to catch more fraud at the cost of "
            "more false alarms."
        ),
    )
    st.markdown("---")
    with st.expander("ℹ️ About the features"):
        st.markdown(
            "`Time` and `Amount` are raw transaction fields. `V1`–`V28` are "
            "PCA-transformed features from the original dataset and are not "
            "individually interpretable — they come from anonymized "
            "transaction details."
        )

# ── Main UI ───────────────────────────────────────────────────────────
st.title("Credit Card Fraud Detection")

tab1, tab2 = st.tabs(["🔍 Single Transaction", "📂 Batch Upload (CSV)"])

# ── Tab 1: Single transaction ─────────────────────────────────────────
with tab1:
    st.subheader("Check a single transaction")
    st.caption(
        "Enter the transaction details below. If you don't have PCA values "
        "for V1–V28, leave them at 0."
    )

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input(
            "Time (seconds since first transaction)", min_value=0.0, value=0.0, step=1.0
        )
    with col2:
        amount_val = st.number_input("Amount ($)", min_value=0.0, value=0.0, step=0.01)

    with st.expander("Advanced: V1–V28 PCA features"):
        v_cols = st.columns(4)
        v_values = {}
        for i in range(1, 29):
            with v_cols[(i - 1) % 4]:
                v_values[f"V{i}"] = st.number_input(
                    f"V{i}", value=0.0, step=0.1, format="%.4f", key=f"v_{i}"
                )

    if st.button("🔎 Predict", type="primary"):
        row = {"Time": time_val, "Amount": amount_val, **v_values}

        with st.spinner("Calling Fraud Detection API..."):
            try:
                pred, prob, label = predict_single(row, threshold)
                is_fraud = pred == 1

                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    if is_fraud:
                        st.error("⚠️ Likely FRAUD")
                    else:
                        st.success("✅ Likely Legitimate")
                    st.metric("Fraud probability", f"{prob:.2%}")
                    st.metric("Threshold used", f"{threshold:.2f}")

                with res_col2:
                    fig = px.bar(
                        x=["Legitimate", "Fraud"],
                        y=[1 - prob, prob],
                        labels={"x": "", "y": "Probability"},
                        color=["Legitimate", "Fraud"],
                        color_discrete_map={"Legitimate": "#2ecc71", "Fraud": "#e74c3c"},
                        text_auto=".2%",
                    )
                    fig.update_layout(showlegend=False, yaxis_range=[0, 1], height=300)
                    st.plotly_chart(fig, use_container_width=True)

            except ConnectionError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ── Tab 2: Batch CSV upload ───────────────────────────────────────────
with tab2:
    st.subheader("Score a CSV file of transactions")
    st.caption(
        "Upload a CSV with columns `Time`, `V1`...`V28`, `Amount`. "
        "An optional `Class` column enables accuracy evaluation."
    )

    template_df = pd.DataFrame([{c: 0.0 for c in FEATURE_COLUMNS}])
    st.download_button(
        "⬇️ Download CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="transaction_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            has_labels = "Class" in data.columns

            # Validate columns
            missing_cols = [c for c in FEATURE_COLUMNS if c not in data.columns]
            if missing_cols:
                st.error(f"Missing columns in CSV: {missing_cols}")
                st.stop()

            with st.spinner(f"Scoring {len(data):,} transactions via API..."):
                pred, proba = predict_batch(data, threshold)

            result_df = data.copy()
            result_df["Fraud_Probability"] = proba
            result_df["Prediction"] = np.where(pred == 1, "Fraud", "Legitimate")

            n_total = len(result_df)
            n_fraud = int(pred.sum())

            # ── Metrics row ──
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Transactions scanned", f"{n_total:,}")
            m2.metric("Flagged as fraud", f"{n_fraud:,}")
            m3.metric("Fraud rate", f"{n_fraud / n_total:.2%}" if n_total else "0%")
            m4.metric("Threshold", f"{threshold:.2f}")

            # ── Results table ──
            st.markdown("#### Results")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "⬇️ Download results as CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

            # ── Probability distribution ──
            fig = px.histogram(
                result_df,
                x="Fraud_Probability",
                nbins=40,
                title="Distribution of predicted fraud probability",
                color_discrete_sequence=["#3498db"]
            )
            fig.add_vline(
                x=threshold, line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({threshold})",
                annotation_position="top right"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Evaluation (only if Class column present) ──
            if has_labels:
                from sklearn.metrics import classification_report, confusion_matrix

                st.markdown("#### Evaluation against provided `Class` labels")
                cm = confusion_matrix(data["Class"], pred)
                cm_fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Legitimate", "Fraud"],
                    y=["Legitimate", "Fraud"],
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(cm_fig, use_container_width=True)
                report = classification_report(data["Class"], pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        except ConnectionError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not process file: {e}")

st.markdown("---")
st.caption(
    "Built with Streamlit + FastAPI · Model: tuned XGBoost · "
    "Dataset: Kaggle Credit Card Fraud Detection"
)