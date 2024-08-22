import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = 'D:\\AI & ML\\Projects\\Projects for Internship\\EPL Match Winner Prediction 23-24\\matches.csv'
matches_df = pd.read_csv(file_path)

# print(matches_df.info())

# print(matches_df.head())

# Print columns to verify the names
print("Columns in the dataset:", matches_df.columns)

# Data Preprocessing
matches_df.fillna(0, inplace=True)

# Fit the LabelEncoder on all unique team names
all_teams = np.unique(np.concatenate((matches_df['Team'], matches_df['Opponent'])))
all_venues = np.unique(matches_df['Venue'])
all_results = np.unique(matches_df['Result'])

label_encoder_teams = LabelEncoder()
label_encoder_teams.fit(all_teams)

label_encoder_venues = LabelEncoder()
label_encoder_venues.fit(all_venues)

label_encoder_results = LabelEncoder()
label_encoder_results.fit(all_results)

# Encode categorical variables like Venue, Result, and Team Names
matches_df['Venue'] = label_encoder_venues.transform(matches_df['Venue'])
matches_df['Result'] = label_encoder_results.transform(matches_df['Result'])  # Win=2, Draw=1, Loss=0
matches_df['Team'] = label_encoder_teams.transform(matches_df['Team'])  # Encode team names
matches_df['Opponent'] = label_encoder_teams.transform(matches_df['Opponent'])

# Select relevant features for prediction
features = ['xG', 'xGA', 'Poss', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'Venue', 'Team', 'Opponent']
X = matches_df[features]
y = matches_df['Result']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling (for SVM and other models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to compare and choose the best model
def select_best_model(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
    }

    best_model_name = None
    best_score = 0
    best_model = None

    # Compare models using cross-validation
    for model_name, model in models.items():
        cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))
        print(f"{model_name} Accuracy: {cv_score:.4f}")

        if cv_score > best_score:
            best_score = cv_score
            best_model_name = model_name
            best_model = model

    print(f"\nBest Model: {best_model_name} with Accuracy: {best_score:.4f}")
    return best_model

# Select the best model
best_model = select_best_model(X_train, y_train)

# Train the best model on the full training data
best_model.fit(X_train, y_train)

# Function to predict win probabilities for two teams
def predict_match_outcome(team, opponent, venue):
    # Remove spaces from the input team names
    team = team.strip()
    opponent = opponent.strip()
    venue = venue.strip()

    # Encode the input team names using the label encoder
    encoded_team = label_encoder_teams.transform([team])[0]
    encoded_opponent = label_encoder_teams.transform([opponent])[0]
    encoded_venue = label_encoder_venues.transform([venue])[0]

    # Filter the dataset for the given team, opponent, and venue
    match_data = matches_df[(matches_df['Team'] == encoded_team) & 
                            (matches_df['Opponent'] == encoded_opponent) & 
                            (matches_df['Venue'] == encoded_venue)]
    
    if match_data.empty:
        return "No match data available for the specified input."

    # Calculate the average statistics for the selected team in the given venue
    avg_xG = match_data['xG'].mean()
    avg_xGA = match_data['xGA'].mean()
    avg_Poss = match_data['Poss'].mean()
    avg_Sh = match_data['Sh'].mean()
    avg_SoT = match_data['SoT'].mean()
    avg_Dist = match_data['Dist'].mean()
    avg_FK = match_data['FK'].mean()
    avg_PK = match_data['PK'].mean()

    # Prepare input data for prediction
    input_data = np.array([[avg_xG, avg_xGA, avg_Poss, avg_Sh, avg_SoT, avg_Dist, avg_FK, avg_PK, encoded_venue, encoded_team, encoded_opponent]])
    input_data = scaler.transform(input_data) 
    
    # Get the prediction probabilities
    probabilities = best_model.predict_proba(input_data)[0]
    
    # Return the probabilities
    return {
        team: f"{probabilities[2]:.5f}",
        "Draw": f"{probabilities[1]:.5f}",
        opponent: f"{probabilities[0]:.5f}"
    }

team = input("Enter the name of the Team: ")
opponent = input("Enter the name of the Opponent: ")
venue = input("Enter the venue (e.g., 'Home' or 'Away'): ")

result_probabilities = predict_match_outcome(team, opponent, venue)
print(f"Winning Probabilities: {result_probabilities}")
