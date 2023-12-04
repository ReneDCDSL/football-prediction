# app.py
import os
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_football_data():
    url = "https://soccer-football-info.p.rapidapi.com/matches/by/basic/"
    querystring = {"s": "423f669ed2e3e1bf", "l": "en_US"}
    headers = {
        "X-RapidAPI-Key": "45ddb35fddmsh83c2b6071c04098p189a48jsn7f1c9dd36e16",
        "X-RapidAPI-Host": "soccer-football-info.p.rapidapi.com",
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()

def preprocess_data(data):
    features = ["possession", "opponent", "referee", "total_goals", "outcome"]
    processed_data = pd.DataFrame(columns=features)
    for match in data["result"]:
        match_info_teamA = match.get("teamA", {}).get("stats", {})
        match_info_teamB = match.get("teamB", {}).get("stats", {})

        # Skips match with missing data
        if match_info_teamA.get('possession') is None or (match['referee'] is None):
            continue

        possession_teamA = int(match_info_teamA.get("possession", 0))
        possession_teamB = int(match_info_teamB.get("possession", 0))

        possession_teamA = int(possession_teamA)
        possession_teamB = int(possession_teamB)

        total_goals = int(match["teamA"]["score"]["f"]) + int(match["teamB"]["score"]["f"])

        outcome = "win" if match["teamA"]["score"]["f"] > match["teamB"]["score"]["f"] else (
            "draw" if match["teamA"]["score"]["f"] == match["teamB"]["score"]["f"] else "lose"
        )

        opponent_name = match["teamB"]["name"]
        referee_name = match["referee"]["name"]

        processed_data = processed_data._append({
            "possession": possession_teamA,
            "opponent": opponent_name,
            "referee": referee_name,
            "total_goals": total_goals,
            "outcome": outcome
        }, ignore_index=True)

    return processed_data


# Function to load or train the model
def load_or_train_model():
    model_filename = "trained_model.joblib"
    data_filename = "preprocessed_data.pkl"

    if os.path.isfile(model_filename) and os.path.isfile(data_filename):
        # Load preprocessed data
        processed_data = pd.read_pickle(data_filename)
        # Load the trained model
        model = joblib.load(model_filename)
    else:
        # Assuming you have loaded the data somehow
        data = get_football_data()
        # Preprocess the data
        processed_data = preprocess_data(data)
        # Split the data into features and target
        X = processed_data.drop("outcome", axis=1)
        y = processed_data["outcome"]
        # Define the preprocessing steps for the model
        categorical_features = ["opponent", "referee"]
        numerical_features = ["possession", "total_goals"]
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='mean'), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        # Create the RandomForest model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        # Train the model
        model.fit(X, y)
        # Save the model and preprocessed data to disk
        joblib.dump(model, model_filename)
        processed_data.to_pickle(data_filename)

    return model, processed_data

def predict_outcome(model, opponent, referee, possession, goals):
    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        "opponent": [opponent],
        "referee": [referee],
        "possession": [float(possession)],
        'total_goals': [int(goals)]
    })

    # Make a prediction
    prediction = model.predict(user_input)

    return prediction[0]


def main():
    st.title("Football Match Outcome Predictor")

    # Load or train the model
    model, processed_data = load_or_train_model()
    
    # User input for opponent, referee, possession, and goals
    opponent = st.text_input("Enter opponent team name:")
    referee = st.text_input("Enter referee name:")
    possession = st.text_input("Enter possession percentage:")
    goals = st.text_input("Enter total goals")

    if st.button("Predict Outcome"):
        if not opponent or not referee or not possession or not goals:
            st.warning("Please enter opponent, referee, possession, and goals.")
        else:
            # Create a DataFrame with user input
            user_input = pd.DataFrame({
                "opponent": [opponent],
                "referee": [referee],
                "possession": [float(possession)],
                'total_goals': [int(goals)]
            })

            # Make a prediction
            prediction = model.predict(user_input)

            st.success(f"The match is likely to end in a {prediction[0]} for Arsenal!")

if __name__ == "__main__":
    main()
