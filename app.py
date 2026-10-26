
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="PaymentShield AI",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "fraud_detection_model.pkl"
)

RAW_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_OUT",
    "type_TRANSFER",
]

# -------------------- STYLING --------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.4rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #102039, #1b3655);
        border: 1px solid #294565;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        color: white;
    }
    .hero p {
        margin: .45rem 0 0;
        color: #c4d5e8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- MODEL LOADING --------------------

@st.cache_resource
def load_model():
    """Load the saved model once per app session."""
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.info(
        "Check that models/fraud_detection_model.pkl exists "
        "and that your installed ML package versions are compatible."
    )
    st.stop()


if not hasattr(model, "predict_proba"):
    st.error("The saved model does not support predict_proba().")
    st.stop()

if getattr(model, "n_features_in_", None) != len(FEATURES):
    st.error(
        f"The model expects "
        f"{getattr(model, 'n_features_in_', 'an unknown number of')} "
        f"features, but this app supplies {len(FEATURES)}. "
        "Check the training feature order."
    )
    st.stop()


# -------------------- PREPROCESSING --------------------

def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw transactions into the model's expected features."""

    missing = [column for column in RAW_COLUMNS if column not in raw.columns]

    if missing:
        raise ValueError(
            "Missing required CSV columns: " + ", ".join(missing)
        )

    df = raw.copy()

    df["type"] = (
        df["type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    allowed_types = {"TRANSFER", "CASH_OUT"}
    invalid_types = sorted(set(df["type"]) - allowed_types)

    if invalid_types:
        raise ValueError(
            "This model supports TRANSFER and CASH_OUT only. "
            "Unsupported types: " + ", ".join(invalid_types)
        )

    numeric_columns = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[RAW_COLUMNS].isna().any().any():
        raise ValueError(
            "Some required values are missing or non-numeric. "
            "Please correct the CSV and upload it again."
        )

    if (df[numeric_columns] < 0).any().any():
        raise ValueError("Amounts and balances cannot be negative.")

    X = df[numeric_columns].copy()

    X["type_CASH_OUT"] = (
        df["type"] == "CASH_OUT"
    ).astype(int)

    X["type_TRANSFER"] = (
        df["type"] == "TRANSFER"
    ).astype(int)

    # Keep the exact feature order expected by the saved model.
    return X[FEATURES]


# -------------------- PREDICTION --------------------

def score_transactions(
    raw: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Return transaction data, scores, and review decisions."""

    X = prepare_features(raw)

    probabilities = model.predict_proba(X)[:, 1]

    results = raw.copy().reset_index(drop=True)

    results["fraud_probability"] = probabilities

    results["decision"] = np.where(
        probabilities >= threshold,
        "Flag for review",
        "Not flagged",
    )

    results["risk_band"] = pd.cut(
        probabilities,
        bins=[-np.inf, 0.05, 0.20, np.inf],
        labels=[
            "Lower score",
            "Elevated score",
            "Higher score",
        ],
    ).astype(str)

    return results


# -------------------- HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ PaymentShield AI</h1>
        <p>
            Machine-learning-assisted transaction fraud screening
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "PaymentShield AI is trained on simulated PaySim transactions."
)

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("Decision settings")

    threshold = st.slider(
        "Flag for review threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.35,
        step=0.01,
        help=(
            "Transactions with scores at or above this value "
            "will be flagged for review."
        ),
    )

    st.caption(
        "Changing this threshold changes the review decision, "
        "not the model's predicted probability."
    )

    st.divider()

    st.write("**Model:** XGBoost classifier")
    st.write("**Input features:** 7")


# -------------------- TABS --------------------

batch_tab, single_tab, about_tab = st.tabs(
    [
        "Batch screening",
        "Single transaction",
        "About the model",
    ]
)


# ==================================================
# BATCH SCREENING
# ==================================================

with batch_tab:
    st.subheader("Transaction screening")
    st.write(
        "Upload a CSV file to score multiple transactions "
        "and identify cases for further review."
    )

    st.markdown("**Required CSV columns**")

    st.code(
        "type,amount,oldbalanceOrg,newbalanceOrig,"
        "oldbalanceDest,newbalanceDest",
        language="text",
    )

    # Example file users can download.
    sample_data = pd.DataFrame(
        [
            {
                "type": "TRANSFER",
                "amount": 5000,
                "oldbalanceOrg": 10000,
                "newbalanceOrig": 5000,
                "oldbalanceDest": 2000,
                "newbalanceDest": 7000,
            },
            {
                "type": "CASH_OUT",
                "amount": 12000,
                "oldbalanceOrg": 15000,
                "newbalanceOrig": 3000,
                "oldbalanceDest": 500,
                "newbalanceDest": 12500,
            },
        ]
    )

    st.download_button(
        "Download sample CSV",
        data=sample_data.to_csv(index=False).encode("utf-8"),
        file_name="paymentshield_sample.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            raw_data = pd.read_csv(uploaded_file)

            if raw_data.empty:
                st.warning("The uploaded CSV contains no transactions.")
            else:
                st.write(
                    f"Transactions loaded: **{len(raw_data):,}**"
                )

                with st.expander("Preview uploaded transactions"):
                    st.dataframe(
                        raw_data.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                if st.button(
                    "Score transactions",
                    type="primary",
                    key="score_batch",
                ):
                    try:
                        results = score_transactions(
                            raw_data,
                            threshold,
                        )

                        total = len(results)

                        flagged = int(
                            (
                                results["fraud_probability"]
                                >= threshold
                            ).sum()
                        )

                        mean_score = float(
                            results["fraud_probability"].mean()
                        )

                        st.divider()
                        st.subheader("Screening summary")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Transactions scored",
                            f"{total:,}",
                        )

                        col2.metric(
                            "Flagged for review",
                            f"{flagged:,}",
                        )

                        col3.metric(
                            "Mean estimated score",
                            f"{mean_score:.2%}",
                        )

                        st.subheader("Score distribution")

                        score_ranges = pd.cut(
                            results["fraud_probability"],
                            bins=[
                                -0.001,
                                0.05,
                                0.20,
                                0.50,
                                1.0,
                            ],
                            labels=[
                                "0–5%",
                                "5–20%",
                                "20–50%",
                                "50–100%",
                            ],
                        )

                        distribution = (
                            score_ranges
                            .value_counts(sort=False)
                            .rename_axis("Score range")
                            .to_frame("Transactions")
                        )

                        st.bar_chart(distribution)

                        st.subheader("Review queue")

                        review_queue = (
                            results[
                                results["fraud_probability"]
                                >= threshold
                            ]
                            .sort_values(
                                "fraud_probability",
                                ascending=False,
                            )
                        )

                        if review_queue.empty:
                            st.info(
                                "No transactions met the selected "
                                "review threshold in this batch."
                            )
                        else:
                            st.dataframe(
                                review_queue,
                                use_container_width=True,
                                hide_index=True,
                            )

                        st.subheader("All scored transactions")

                        st.dataframe(
                            results.sort_values(
                                "fraud_probability",
                                ascending=False,
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            "Download scored results",
                            data=results.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name=(
                                "paymentshield_scored_results.csv"
                            ),
                            mime="text/csv",
                        )

                    except Exception as exc:
                        st.error(
                            f"Could not score the transactions: {exc}"
                        )

        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")


# ==================================================
# SINGLE TRANSACTION
# ==================================================

with single_tab:
    st.subheader("Assess one transaction")

    with st.form("single_transaction_form"):
        transaction_type = st.selectbox(
            "Transaction type",
            ["TRANSFER", "CASH_OUT"],
        )

        left, right = st.columns(2)

        with left:
            amount = st.number_input(
                "Transaction amount",
                min_value=0.0,
                value=5000.0,
                step=100.0,
            )

            sender_before = st.number_input(
                "Sender balance before",
                min_value=0.0,
                value=10000.0,
                step=100.0,
            )

            sender_after = st.number_input(
                "Sender balance after",
                min_value=0.0,
                value=5000.0,
                step=100.0,
            )

        with right:
            recipient_before = st.number_input(
                "Recipient balance before",
                min_value=0.0,
                value=2000.0,
                step=100.0,
            )

            recipient_after = st.number_input(
                "Recipient balance after",
                min_value=0.0,
                value=7000.0,
                step=100.0,
            )

        submitted = st.form_submit_button(
            "Calculate risk score",
            type="primary",
        )

    if submitted:
        transaction = pd.DataFrame(
            [
                {
                    "type": transaction_type,
                    "amount": amount,
                    "oldbalanceOrg": sender_before,
                    "newbalanceOrig": sender_after,
                    "oldbalanceDest": recipient_before,
                    "newbalanceDest": recipient_after,
                }
            ]
        )

        try:
            result = score_transactions(
                transaction,
                threshold,
            )

            probability = float(
                result.loc[0, "fraud_probability"]
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Estimated fraud probability",
                f"{probability:.2%}",
            )

            col2.metric(
                "Screening decision",
                (
                    "Flag for review"
                    if probability >= threshold
                    else "Not flagged"
                ),
            )

            st.progress(
                float(np.clip(probability, 0.0, 1.0))
            )

            if probability >= threshold:
                st.warning(
                    "The score meets the selected review threshold. "
                    "This is not proof that the transaction is fraudulent."
                )
            else:
                st.success(
                    "The score is below the selected threshold. "
                    "This does not guarantee that the transaction is safe."
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ==================================================
# ABOUT
# ==================================================

with about_tab:
    st.subheader("How PaymentShield AI works")

    st.markdown(
        """
        1. The user enters a transaction or uploads a CSV.
        2. The application validates the input fields.
        3. Transaction type is converted into the two encoded features
           expected by the model.
        4. The saved XGBoost model returns an estimated fraud probability.
        5. A configurable threshold determines which transactions
           are flagged for further review.

        ### Important limitations

        - PaySim is simulated data. Results do not establish effectiveness
          on real UPI or banking transactions.
        - A low score does not guarantee a transaction is safe.
        - A high score does not prove fraud.
        - The threshold is a demo setting, not a validated operational policy.
        - This app does not provide feature-attribution explanations.
        - Do not use this prototype to approve or reject real payments
          without rigorous validation, monitoring, and human oversight.
        """
    )

st.divider()

st.caption(
    "PaymentShield AI · Educational portfolio project · "
    "No real payments are processed."
)
