# app.py
import streamlit as st
import joblib
import pickle
import pandas as pd

def predict_outcome(opponent, referee, possession, goals):
    # Load the trained model
    #trained_model = joblib.load("trained_model.joblib")
    trained_model = pickle.load(open('trained_model.sav', 'rb'))
    
    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        "opponent": [opponent],
        "referee": [referee],
        "possession": [int(possession)],
        'total_goals': [int(goals)]
    })

    # Make a prediction
    prediction = trained_model.predict(user_input)

    return prediction[0]

def main():
    st.title("Football Match Outcome Predictor")

    # User input for opponent, referee, and possession
    opponent = st.text_input("Enter opponent team name:")
    referee = st.text_input("Enter referee name:")
    possession = st.text_input("Enter possession percentage:")
    goals = st.text_input("Enter total goals")

    if st.button("Predict Outcome"):
        if not opponent or not referee or not possession or not goals:
            st.warning("Please enter opponent, referee, and possession.")
        else:
            outcome = predict_outcome(opponent, referee, possession, goals)
            st.success(f"The match is likely to end in a {outcome} for Arsenal!")

if __name__ == "__main__":
    main()

