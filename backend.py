# backend.py
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

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
    features = ["possession", "attacks", "shoots", "corners", "fouls", "substitutions"]
    processed_data = pd.DataFrame(columns=features + ["outcome"])

    for match in data["result"]:
        match_info = match.get("teamA", {}).get("stats", {})
        match_features = [match_info.get(feature, None) for feature in features]
        outcome = 1 if match["teamA"]["score"]["f"] > match["teamB"]["score"]["f"] else 0

        processed_data = processed_data.append(
            pd.Series(match_features + [outcome], index=features + ["outcome"]),
            ignore_index=True,
        )

    return processed_data

def train_model(processed_data):
    X = processed_data.drop("outcome", axis=1)
    y = processed_data["outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    return model

if __name__ == "__main__":
    data = get_football_data()
    processed_data = preprocess_data(data)
    trained_model = train_model(processed_data)

    # Save the trained model
    joblib.dump(trained_model, "trained_model.joblib")
