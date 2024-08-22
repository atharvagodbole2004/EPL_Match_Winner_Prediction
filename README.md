# EPL_Match_Winner_Prediction
Using Kaggle Data for EPL 23/24 Season to predict winning probabilities for matches.

Overview: This project aims to predict the outcome of English Premier League (EPL) football matches using machine learning techniques. The prediction model is built using historical match data and includes feature engineering, model training, and evaluation. The project employs three machine learning models: Random Forest, Support Vector Machine (SVM), and XGBoost, to determine the best-performing model for predicting match outcomes.

Dataset: https://www.kaggle.com/datasets/mertbayraktar/english-premier-league-matches-20232024-season/data
The dataset used for this project is sourced from the English Premier League matches. It includes various features related to team performance, match statistics, and results. The key columns in the dataset are:

Team: Name of the team playing the match. Opponent: Name of the opposing team. Venue: Venue of the match (e.g., Home, Away). Result: Outcome of the match (Win, Draw, Loss). xG: Expected Goals for the team. xGA: Expected Goals Against the team. Poss: Possession percentage. Sh: Total Shots. SoT: Shots on Target. Dist: Distance of shots. FK: Free Kicks. PK: Penalty Kicks.

Installation: To run this project, you need to have Python installed along with the following libraries: 
pandas 
numpy 
scikit-learn 
xgboost
