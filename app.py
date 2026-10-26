                                    "maximum_fraud_probability": max_score,
                                    "unique_probability_count": int(
                                        results["fraud_probability"].nunique()
                                    ),
                                    "threshold": threshold,
                                }
                            )
                            if results["fraud_probability"].nunique() == 1:
                                st.warning(
                                    "Every transaction received exactly the same score. "
                                    "This may indicate a model/pipeline mismatch or a model "
                                    "that is not learning useful distinctions."
                                )
                            elif max_score < 0.05:
                                st.warning(
                                    "All scores are below 5%. Check the training target labels, "
                                    "feature order, preprocessing, and evaluation metrics on "
                                    "labelled holdout data. The app cannot fix a model that "
                                    "has learned or predicts the wrong pattern."
                                )

                    except Exception as exc:
                        st.error(f"Could not score the transactions: {exc}")

        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")


# =========================================================
# SINGLE TRANSACTION
# =========================================================

with single_tab:
    st.subheader("Assess one transaction")
    st.caption("A model score is a screening estimate, not proof of fraud or safety.")

    with st.form("single_transaction_form"):
        transaction_type = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT"])
        left, right = st.columns(2)

        with left:
            amount = st.number_input("Transaction amount", min_value=0.0, value=25000.0, step=100.0)
            sender_before = st.number_input("Sender balance before", min_value=0.0, value=30000.0, step=100.0)
            sender_after = st.number_input("Sender balance after", min_value=0.0, value=5000.0, step=100.0)

        with right:
            recipient_before = st.number_input("Recipient balance before", min_value=0.0, value=10000.0, step=100.0)
            recipient_after = st.number_input("Recipient balance after", min_value=0.0, value=35000.0, step=100.0)

        submitted = st.form_submit_button("Calculate risk score", type="primary")

    if submitted:
        transaction = pd.DataFrame(
            [{
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": sender_before,
                "newbalanceOrig": sender_after,
                "oldbalanceDest": recipient_before,
                "newbalanceDest": recipient_after,
            }],
            columns=RAW_COLUMNS,
        )

        try:
            result, prepared_features, probabilities, fraud_index = score_transactions(
                transaction, threshold
            )
            probability = float(result.loc[0, "fraud_probability"])
            col1, col2 = st.columns(2)
            col1.metric("Estimated fraud probability", f"{probability:.3%}")
            col2.metric("Screening decision", str(result.loc[0, "decision"]))
            st.progress(float(np.clip(probability, 0.0, 1.0)))
            st.write("Risk band:", str(result.loc[0, "risk_band"]))

            if probability >= threshold:
                st.warning("This score meets the review threshold. It is not proof of fraud.")
            else:
                st.info("This score is below the selected threshold. It does not guarantee safety.")

            with st.expander("Prediction details"):
                st.write("Model classes:", [str(x) for x in model.classes_])
                st.write("Fraud class index:", fraud_index)
                st.write("Prepared feature values:")
                st.dataframe(prepared_features, use_container_width=True)
                st.write("predict_proba output:")
                st.dataframe(
                    pd.DataFrame(probabilities, columns=[f"class_{x}" for x in model.classes_]),
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# =========================================================
# ABOUT & DIAGNOSTICS
# =========================================================

with about_tab:
    st.subheader("How PaymentShield AI works")
    st.markdown(
        """
        1. The user enters one transaction or uploads a CSV.
        2. The app validates the required fields and transaction type.
        3. It constructs the seven features expected by the saved model.
        4. The saved classifier returns class probabilities.
        5. The probability for the identified fraud class is compared with the selected threshold.

        **Important limitations**
        - PaySim is simulated data; this does not establish effectiveness on real UPI or banking transactions.
        - A low score does not guarantee safety; a high score does not prove fraud.
        - The threshold is a configurable demonstration setting, not a validated operational policy.
        - This app does not retrain the model and does not require the notebook at runtime.
        - Validate performance with labelled holdout data before relying on predictions.
        """
    )
    st.subheader("Saved model information")
    st.write("Model class:", type(model).__name__)
    st.write("Model classes:", [str(x) for x in model.classes_])
    st.write("Model feature count:", getattr(model, "n_features_in_", "Not reported"))
    st.write("Expected app feature order:", FEATURES)
    try:
        st.write("Identified fraud class index:", get_fraud_class_index(model))
    except Exception as exc:
        st.error(str(exc))

st.divider()
