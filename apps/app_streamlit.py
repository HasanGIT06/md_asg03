import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from feature_engineering.feature_engineering import feature_engineering
from preprocessing.preprocessing import preprocess_data

# Load preprocessor and model
encoders = joblib.load(Path(__file__).parent/"artifacts/preprocessor.pkl")
model  = joblib.load(Path(__file__).parent/"artifacts/model.pkl")

def main():
    st.title("ASG 04 MD - Leonardus Hasan - Spaceship Titanic Model Deployment")
    passenger_id    = st.text_input("Passenger ID", "0001_01")
    home_planet     = st.selectbox("Home Planet", ("Earth", "Europa", "Mars"), index=1)
    cryo_sleep      = st.selectbox("Cryo Sleep", (True, False), index=0)
    cabin           = st.text_input("Cabin", "B/0/P")
    destination     = st.selectbox("Destination", ("TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"), index = 2)
    vip             = st.selectbox("VIP", (True, False), index=0)
    age             = st.number_input("Age", min_value=0, value= 19)
    room_service    = st.slider("Room Service", min_value=0.0, value= 7000.0)
    food_court      = st.slider("Food Court", min_value=0.0, value= 15000.0)
    shopping_mall   = st.slider("Shopping Mall", min_value=0.0, value= 12000.0)
    spa             = st.slider("Spa", min_value=0.0, value= 10000.0)
    vr_deck         = st.slider("VR Deck", min_value=0.0, value= 12000.0)
    name            = st.text_input("Name", "Maham Ofracculy")

    if st.button("Make Prediction"):
        features = [[passenger_id, home_planet, cryo_sleep, cabin, destination, vip, age, room_service, food_court,
                     shopping_mall, spa, vr_deck, name]]
        features = pd.DataFrame(features, columns=["PassengerId", 'HomePlanet', 'CryoSleep', "Cabin", 'Destination', 'VIP',
                            'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', "Name"])
        result = make_prediction(features)
        if result == 1:
            st.success("✅️ Passenger will be transported")
        else:
            st.error("❌ Passenger will NOT be transpoted")

def make_prediction(features):
    features = feature_engineering(features)
    X = preprocess_data(features, is_train=False)
    if "Transported" in X.columns:
        X = X.drop(columns=["Transported"])
    pred = model.predict(X)
    return pred[0]

if __name__ == "__main__":
    main()