from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "fraud_detection_model.pkl"
FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_OUT",
    "type_TRANSFER",
]

def load_model():
    """Load the trusted, locally saved XGBoost model."""
    return joblib.load(MODEL_PATH)

def build_features(
    amount: float,
    oldbalance_org: float,
    newbalance_orig: float,
    oldbalance_dest: float,
    newbalance_dest: float,
    transaction_type: str,
) -> pd.DataFrame:
    if transaction_type not in {"CASH_OUT", "TRANSFER"}:
        raise ValueError("transaction_type must be CASH_OUT or TRANSFER")
    values = {
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "newbalanceOrig": newbalance_orig,
        "oldbalanceDest": oldbalance_dest,
        "newbalanceDest": newbalance_dest,
        "type_CASH_OUT": int(transaction_type == "CASH_OUT"),
        "type_TRANSFER": int(transaction_type == "TRANSFER"),
    }
    return pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)

def predict_fraud_probability(model, features: pd.DataFrame) -> float:
    if list(features.columns) != FEATURES:
        raise ValueError(f"Features must be in this exact order: {FEATURES}")
    return float(model.predict_proba(features)[0][1])
