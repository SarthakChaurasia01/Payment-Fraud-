from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PaymentShield AI | Fraud Risk",
    page_icon="🛡️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_detection_model.pkl"

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_OUT",
    "type_TRANSFER",
]

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Make sure models/fraud_detection_model.pkl is in the repository."
        )
    return joblib.load(MODEL_PATH)

st.title("🛡️ PaymentShield AI")
st.subheader("Payment transaction fraud-risk assessment")
st.write(
    "Enter the transaction details to estimate whether the transaction resembles "
    "fraudulent patterns learned by the model."
)

with st.sidebar:
    st.header("About this demo")
    st.write("Model: XGBoost classifier")
    st.write("Training data: PaySim simulated mobile-money transactions")
    threshold = st.slider(
        "Fraud decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Lower thresholds flag more transactions but may increase false positives.",
    )
    st.caption(
        "This threshold changes the app's decision rule, not the trained model."
    )

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

expected_features = getattr(model, "n_features_in_", None)
if expected_features != len(FEATURES):
    st.error(
        f"Model expects {expected_features} features, but the app is configured "
        f"for {len(FEATURES)}. Check the notebook's preprocessing and feature order."
    )
    st.stop()

with st.form("transaction_form"):
    st.markdown("### Transaction details")
    transaction_type = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT"])

    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input("Transaction amount", min_value=0.0, value=5000.0, step=100.0)
        oldbalance_org = st.number_input(
            "Sender balance before transaction", min_value=0.0, value=10000.0, step=100.0
        )
        newbalance_orig = st.number_input(
            "Sender balance after transaction", min_value=0.0, value=5000.0, step=100.0
        )
    with c2:
        oldbalance_dest = st.number_input(
            "Recipient balance before transaction", min_value=0.0, value=2000.0, step=100.0
        )
        newbalance_dest = st.number_input(
            "Recipient balance after transaction", min_value=0.0, value=7000.0, step=100.0
        )

    submitted = st.form_submit_button("Assess transaction risk", type="primary")

if submitted:
    if amount < 0 or min(oldbalance_org, newbalance_orig, oldbalance_dest, newbalance_dest) < 0:
        st.error("Amounts and balances cannot be negative.")
        st.stop()

    row = {
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
        "type_CASH_OUT": int(transaction_type == "CASH_OUT"),
        "type_TRANSFER": int(transaction_type == "TRANSFER"),
    }
    X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)

    try:
        probability = float(model.predict_proba(X)[0][1])
        predicted_class = int(probability >= threshold)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.markdown("---")
    st.markdown("### Assessment result")
    metric_col, result_col = st.columns([1, 2])
    with metric_col:
        st.metric("Estimated fraud probability", f"{probability:.2%}")
    with result_col:
        if predicted_class == 1:
            st.error("⚠️ **Flag for review** — the model score meets or exceeds the selected threshold.")
        else:
            st.success("✅ **Not flagged by this threshold** — this does not guarantee the transaction is legitimate.")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        f"Decision threshold: {threshold:.0%}. This is a model estimate, not a definitive fraud determination."
    )

st.markdown("---")
st.caption(
    "Educational prototype only. PaySim is simulated data; performance on it does not establish "
    "real-world UPI, banking, or payment-system effectiveness. Do not use this app to approve or reject "
    "real transactions without independent validation, monitoring, and human oversight."
)
