# backend.py
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import pickle

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
        if match_info_teamA.get('possession')==None or (match['referee']==None):
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

        processed_data = processed_data.append({
            "possession": possession_teamA,
            "opponent": opponent_name,
            "referee": referee_name,
            "total_goals": total_goals,
            "outcome": outcome
        }, ignore_index=True)

    return processed_data

def train_model(processed_data):
    X = processed_data.drop("outcome", axis=1)
    y = processed_data["outcome"]

    # Define the categorical features and numerical features
    categorical_features = ["opponent", "referee"]
    numerical_features = ["possession", "total_goals"]

    # Create transformers for one-hot encoding of categorical features
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Create a preprocessor that applies transformers to different feature sets
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"
    )

    # Create a pipeline that applies the preprocessor and then fits the model
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier())
    ])

    # Fit the model
    model.fit(X, y)

    return model

# Train the model
data = get_football_data()
processed_data = preprocess_data(data)
trained_model = train_model(processed_data)

# Save the trained model
joblib.dump(trained_model, "trained_model.joblib", protocol=4)
pickle.dump(trained_model, open('trained_model.sav', 'wb'), protocol=4)