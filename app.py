# app.py
import streamlit as st
import pandas as pd
from backend import train_model
import joblib

def main():
    st.title("Soccer Game Outcome Prediction")

    features = ["possession", "attacks", "shoots", "corners", "fouls", "substitutions"]
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(f"Enter {feature}", value=0)

    if st.button("Predict Outcome"):
        user_input_df = pd.DataFrame([user_input])

        # Load the trained model
        trained_model = joblib.load("trained_model.joblib")

        prediction = trained_model.predict(user_input_df)

        outcome = "Win" if prediction[0] == 1 else "Lose"
        st.success(f"Predicted Outcome: {outcome}")

if __name__ == "__main__":
    main()
