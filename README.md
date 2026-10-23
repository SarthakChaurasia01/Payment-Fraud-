# PaymentShield AI

A Streamlit demo that uses an XGBoost classifier to estimate fraud risk for simulated PaySim mobile-money transactions.

## Features

- Interactive transaction form for `TRANSFER` and `CASH_OUT`
- Estimated fraud probability from `predict_proba`
- Adjustable decision threshold
- Input feature order aligned to the seven-feature model interface
- Streamlit Community Cloud deployment-ready layout

## Repository layout

```text
PaymentShield-AI/
├── app.py
├── models/
│   └── fraud_detection_model.pkl
├── notebooks/
│   └── fraud_detection_notebook.ipynb
├── src/
│   ├── __init__.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally

Python 3.10 or 3.11 is recommended for broad ML package compatibility.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Windows Command Prompt: `.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

Then install and run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository named `PaymentShield-AI`.
2. Upload the project files, including `models/fraud_detection_model.pkl`.
3. Open Streamlit Community Cloud and connect your GitHub account.
4. Create an app from the repository and set the main file path to `app.py`.
5. Deploy. If the model fails to load because of package-version incompatibility, use the versions from the training environment or re-export the model using a compatible XGBoost version.

## Model inputs

The app expects exactly these seven features, in this order:

1. `amount`
2. `oldbalanceOrg`
3. `newbalanceOrig`
4. `oldbalanceDest`
5. `newbalanceDest`
6. `type_CASH_OUT`
7. `type_TRANSFER`

The notebook filters the dataset to `CASH_OUT` and `TRANSFER`, drops `nameOrig`, `nameDest`, `isFlaggedFraud`, and `step`, and one-hot encodes `type`. The app creates the two one-hot columns itself.

## Important limitations

- PaySim contains simulated transactions. Results do not prove performance on real-world UPI or banking data.
- A model probability is not a guarantee that a transaction is fraudulent or legitimate.
- The threshold is a user-selected demo setting and should be calibrated on representative validation data before any real use.
- Never commit private transaction data, API keys, credentials, or personal information.
- Only load `.pkl` files from sources you trust; pickle-based files can execute code when loaded.
