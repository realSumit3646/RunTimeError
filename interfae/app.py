import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Define IPL teams and their codes
team_codes = {
    "Chennai Super Kings": "CSK",
    "Kolkata Knight Riders": "KKR",

    "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH",
    "Mumbai Indians": "MI"
}

# Reverse the mapping for display purposes
code_to_team = {v: k for k, v in team_codes.items()}

# Load CSV data
df = pd.read_csv('match_data.csv')  # Ensure match_data.csv has the necessary columns

# Load the model
with open('best_model.pkl', 'rb') as model_file:
    try:
        prediction_model = pickle.load(model_file)
    except EOFError:
        st.error("Error: The saved model might be incompatible. Consider recreating it with the same sklearn version used for training.")
    except AttributeError:
        st.error("Error: Loaded object is not a model. Check the pickle file.")

def predict(team1_code, year1, team2_code, year2, team3_code, year3):
    try:
        # Filter the data from the CSV based on team and season
        team1_data = df[(df['bowling_team'] == team1_code) & (df['season'] == int(year1))].iloc[0]
        team2_data = df[(df['bowling_team'] == team2_code) & (df['season'] == int(year2))].iloc[0]
        team3_data = df[(df['bowling_team'] == team3_code) & (df['season'] == int(year3))].iloc[0]
    except IndexError:
        st.error("Data not found for one or more teams/years.")
        return

    # Extract relevant columns into arrays
    team1_array = team1_data[['batting_position', 'runs', 'balls', 'fours', 'sixes', 'strike_rate', 'Batting_FP', 'TBS', 'TBOS', 'OTBS', 'OTBOS']].values
    team2_array = team2_data[['batting_position', 'runs', 'balls', 'fours', 'sixes', 'strike_rate', 'Batting_FP', 'TBS', 'TBOS', 'OTBS', 'OTBOS']].values
    team3_array = team3_data[['TBS', 'TBOS', 'OTBS', 'OTBOS']].values

    # Prepare the input sequence for the model
    input_sequence = np.array([team1_array[:7], team2_array[:7]]).astype(np.float32).reshape(1, 2, 7)
    extra_team3_data = np.array(team3_array).reshape(1, -1).astype(np.float32)

    # Ensure the model is correctly loaded and callable
    if hasattr(prediction_model, 'predict'):
        # Use the model to predict the missing features for team 3
        new_match_pred = prediction_model.predict([input_sequence, extra_team3_data])

        # Return prediction in a dictionary format
        return {
            "team1": team1_array.tolist(),
            "team2": team2_array.tolist(),
            "predicted_team3": new_match_pred.tolist()
        }
    else:
        st.error("Loaded object is not a valid model. Check the pickle file.")

# Streamlit app interface
st.title("IPL Match Predictor")

# Dropdowns for team selection with full names but using codes
team1_code = st.selectbox(
    "Select Team 1:",
    options=list(team_codes.values()),
    format_func=lambda x: code_to_team[x]  # Display full team names
)
year1 = st.number_input("Select Year 1:", min_value=2008, max_value=2024)

team2_code = st.selectbox(
    "Select Team 2:",
    options=list(team_codes.values()),
    format_func=lambda x: code_to_team[x]  # Display full team names
)
year2 = st.number_input("Select Year 2:", min_value=2008, max_value=2024)

team3_code = st.selectbox(
    "Select Team 3:",
    options=list(team_codes.values()),
    format_func=lambda x: code_to_team[x]  # Display full team names
)
year3 = st.number_input("Select Year 3:", min_value=2008, max_value=2024)

# Store years and team codes in arrays
years = [year1, year2, year3]
team_codes_array = [team1_code, team2_code, team3_code]

# Prediction button
if st.button("Predict"):
    result = predict(team_codes_array[0], years[0], team_codes_array[1], years[1], team_codes_array[2], years[2])
    if result:
        st.write("Team 1 Data:", result["team1"])
        st.write("Team 2 Data:", result["team2"])
        st.write("Predicted Team 3 Data:", result["predicted_team3"])