from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PaymentShield AI", page_icon="🛡️", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fraud_detection_model.pkl"
FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "type_CASH_OUT", "type_TRANSFER",
]
RAW_REQUIRED = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "type",
]

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px;}
.hero {padding: 1.25rem 1.4rem; border-radius: 16px; background: linear-gradient(115deg, #101b31, #172a42); border: 1px solid #2b4058; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2.1rem;}
.hero p {margin: .45rem 0 0 0; color: #c5d4e5;}
.small-note {font-size: .86rem; color: #9aa9bb;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def make_features(df):
    data = df.copy()
    data["type"] = data["type"].astype(str).str.upper().str.strip()
    invalid = sorted(set(data["type"].dropna()) - {"CASH_OUT", "TRANSFER"})
    if invalid:
        raise ValueError("Unsupported transaction type(s): " + ", ".join(invalid))
    for c in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    if data[RAW_REQUIRED].isna().any().any():
        raise ValueError("CSV contains missing or non-numeric required values.")
    if (data[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]] < 0).any().any():
        raise ValueError("Amounts and balances must not be negative.")
    X = pd.DataFrame(index=data.index)
    for c in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        X[c] = data[c]
    X["type_CASH_OUT"] = (data["type"] == "CASH_OUT").astype(int)
    X["type_TRANSFER"] = (data["type"] == "TRANSFER").astype(int)
    return X[FEATURES]

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

if getattr(model, "n_features_in_", None) != len(FEATURES):
    st.error(
        f"The loaded model expects {getattr(model, 'n_features_in_', 'an unknown number of')} "
        f"features, but this app supplies {len(FEATURES)}. Check training preprocessing."
    )
    st.stop()

st.markdown("""
<div class="hero">
  <h1>🛡️ PaymentShield AI</h1>
  <p>Payment fraud screening · Model-assisted triage · Batch review workspace</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Screening settings")
    threshold = st.slider(
        "Flag for review at probability",
        min_value=0.01, max_value=0.99, value=0.35, step=0.01,
        help="A demo decision threshold only. Calibrate on representative validation data before real use."
    )
    st.caption("Lower thresholds flag more cases and may increase false alarms.")
    st.divider()
    st.caption("Model: XGBoost classifier")
    st.caption("Training data: simulated PaySim transactions")

single_tab, batch_tab, guide_tab = st.tabs(
    ["🔎 Single transaction", "📂 Batch screening", "ℹ️ How to use"]
)

def score_df(raw):
    X = make_features(raw)
    probs = model.predict_proba(X)[:, 1]
    out = raw.copy().reset_index(drop=True)
    out["estimated_fraud_probability"] = probs
    out["risk_band"] = pd.cut(
        probs, bins=[-np.inf, 0.05, threshold, np.inf],
        labels=["Lower score", "Monitor", "Flag for review"], include_lowest=True
    ).astype(str)
    # Keep the flag decision explicitly tied to the user-selected threshold.
    out["decision"] = np.where(probs >= threshold, "Flag for review", "Not flagged")
    return out

with single_tab:
    st.subheader("Assess one transaction")
    st.write("Enter transaction details to estimate the model's fraud score.")
    with st.form("single_transaction"):
        transaction_type = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT"])
        left, right = st.columns(2)
        with left:
            amount = st.number_input("Transaction amount", min_value=0.0, value=5000.0, step=100.0)
            old_org = st.number_input("Sender balance before", min_value=0.0, value=10000.0, step=100.0)
            new_org = st.number_input("Sender balance after", min_value=0.0, value=5000.0, step=100.0)
        with right:
            old_dest = st.number_input("Recipient balance before", min_value=0.0, value=2000.0, step=100.0)
            new_dest = st.number_input("Recipient balance after", min_value=0.0, value=7000.0, step=100.0)
        submitted = st.form_submit_button("Calculate risk score", type="primary", use_container_width=True)

    if submitted:
        raw = pd.DataFrame([{
            "type": transaction_type, "amount": amount,
            "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        }])
        try:
            result = score_df(raw)
            p = float(result.loc[0, "estimated_fraud_probability"])
            a, b, c = st.columns(3)
            a.metric("Estimated fraud score", f"{p:.2%}")
            b.metric("Decision threshold", f"{threshold:.0%}")
            c.metric("Model decision", "Flag for review" if p >= threshold else "Not flagged")
            st.progress(float(np.clip(p, 0, 1)))
            if p >= threshold:
                st.warning("This transaction meets the selected review threshold. Review the transaction before making a decision.")
            else:
                st.success("This transaction is below the selected threshold. That does not prove it is legitimate.")
            with st.expander("Input summary"):
                st.dataframe(raw, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not score transaction: {e}")

with batch_tab:
    st.subheader("Screen a batch of transactions")
    st.write("Upload a CSV with one transaction per row. Required columns:")
    st.code(", ".join(RAW_REQUIRED), language="text")
    st.caption("Only CASH_OUT and TRANSFER rows are supported by this model.")
    template = pd.DataFrame([{
        "type": "TRANSFER", "amount": 5000,
        "oldbalanceOrg": 10000, "newbalanceOrig": 5000,
        "oldbalanceDest": 2000, "newbalanceDest": 7000,
    }, {
        "type": "CASH_OUT", "amount": 12000,
        "oldbalanceOrg": 15000, "newbalanceOrig": 3000,
        "oldbalanceDest": 500, "newbalanceDest": 12500,
    }])
    st.download_button("Download sample CSV template", template.to_csv(index=False).encode("utf-8"),
                       file_name="paymentshield_sample.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"], key="batch_csv")
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            missing = [c for c in RAW_REQUIRED if c not in raw.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            elif len(raw) == 0:
                st.warning("The uploaded CSV contains no rows.")
            else:
                st.write(f"Loaded **{len(raw):,}** transactions.")
                with st.expander("Preview uploaded data"):
                    st.dataframe(raw.head(20), use_container_width=True, hide_index=True)
                if st.button("Screen transactions", type="primary"):
                    try:
                        results = score_df(raw)
                        n_flagged = int((results["estimated_fraud_probability"] >= threshold).sum())
                        n_total = len(results)
                        avg_p = float(results["estimated_fraud_probability"].mean())
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Transactions screened", f"{n_total:,}")
                        k2.metric("Flagged for review", f"{n_flagged:,}", f"{n_flagged/n_total:.1%} of batch")
                        k3.metric("Mean estimated score", f"{avg_p:.2%}")
                        hist = pd.DataFrame({
                            "Score range": ["0–5%", "5–20%", "20–50%", "50–100%"],
                            "Transactions": [
                                int(((results["estimated_fraud_probability"] >= 0) & (results["estimated_fraud_probability"] < .05)).sum()),
                                int(((results["estimated_fraud_probability"] >= .05) & (results["estimated_fraud_probability"] < .20)).sum()),
                                int(((results["estimated_fraud_probability"] >= .20) & (results["estimated_fraud_probability"] < .50)).sum()),
                                int((results["estimated_fraud_probability"] >= .50).sum()),
                            ]
                        })
                        chart_col, text_col = st.columns([1, 1])
                        with chart_col:
                            st.markdown("#### Score distribution")
                            st.bar_chart(hist.set_index("Score range"))
                        with text_col:
                            st.markdown("#### Review queue")
                            flagged = results[results["estimated_fraud_probability"] >= threshold].sort_values(
                                "estimated_fraud_probability", ascending=False
                            )
                            st.write(f"{len(flagged):,} transactions meet the selected threshold.")
                            if len(flagged):
                                st.dataframe(flagged.head(15), use_container_width=True, hide_index=True)
                            else:
                                st.info("No rows met the current threshold. You can adjust the threshold in the sidebar.")
                        st.markdown("#### All scored transactions")
                        st.dataframe(
                            results.sort_values("estimated_fraud_probability", ascending=False),
                            use_container_width=True, hide_index=True
                        )
                        st.download_button(
                            "Download scored results",
                            results.to_csv(index=False).encode("utf-8"),
                            file_name="paymentshield_scored_transactions.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Batch screening failed: {e}")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

with guide_tab:
    st.subheader("What this app can and cannot tell you")
    st.markdown("""
**Workflow**
1. Enter one transaction or upload a batch CSV.
2. The app maps transaction fields to the seven features expected by the trained model.
3. The model returns an estimated fraud probability.
4. Scores at or above the selected threshold are placed in the review queue.

**Limitations**
- The model was trained on simulated PaySim transactions; that does not establish performance on real UPI or banking transactions.
- A low score does not guarantee a transaction is safe, and a high score does not prove fraud.
- The threshold is a configurable demo setting, not a validated operational policy.
- Do not use this prototype to approve or reject real transactions without independent evaluation, security review, monitoring, and human oversight.
- The app does not currently provide feature-attribution explanations such as SHAP; balance summaries are not proof of why the model made a prediction.
""")

st.divider()
st.caption("PaymentShield AI · Portfolio prototype for ML-assisted transaction triage · Not for real financial decisions")
