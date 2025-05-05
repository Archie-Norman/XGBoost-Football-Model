import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv("C:/Users/archi/Desktop/sports betting/Saves data/cleaned-data.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

if not (data['timestamp'] < pd.Timestamp('2023-01-01')).any():
    raise ValueError("Historical data missing: Please add data from before 2023.")

data = data.drop_duplicates()


# Filter rows with sufficient games played
data = data[data['games_played_home_total'] > 5]
data = data[data['games_played_away_total'] > 5]



# Handle missing values
data['venue_home'] = data['venue_home'].fillna('Unknown')
data['venue_away'] = data['venue_away'].fillna('Unknown')

# Ensure timestamp column is in datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')


# Split past and future data
future_data = data[data['timestamp'] > datetime.now()]
data = data[data['timestamp'] < pd.Timestamp.now()]

print(data)
print(future_data)

features = [
    'venue_home', 'team_home', 'against_home', 'competition_id_home',
    'points_before_game_home', 'conceded_before_game_home', 'goals_before_game_home',
    'games_to_points_ratio_home_home', 'games_to_points_ratio_away_home',
    'last_games_elo_home', 'against_last_elo_home', 'elo_diff_home',
    'against_points_before_game_home', 'against_conceded_before_game_home',
    'against_goals_before_game_home', 'against_games_to_points_ratio_home_home',
    'against_games_to_points_ratio_away_home', 'points_before_game_away',
    'conceded_before_game_away', 'goals_before_game_away',
    'last_games_elo_away', 'against_points_before_game_away',
    'against_conceded_before_game_away', 'against_goals_before_game_away',
    'max_elo_so_far_away', 'min_elo_so_far_away', 'median_elo_so_far_away',
    'mean_elo_so_far_away', 'std_elo_so_far_away'
]

X = data[features]
y = data['win_or_loss_home']



# One-hot encoding for categorical variables
X_train_encoded = pd.get_dummies(X, columns=['venue_home', 'team_home', 'against_home', 'competition_id_home'], drop_first=True)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)


# Train the base XGBoost model
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',
    max_depth=3,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    alpha = 5,
    gamma=0.2,
    reg_lambda=5,
    n_estimators=200,
    learning_rate=0.3,
)

xgb_model.fit(X_train_scaled, y)

import matplotlib.pyplot as plt
# Set feature names as a list for the model's booster
xgb_model.get_booster().feature_names = list(X_train_encoded.columns)

# Plot feature importance
xgb.plot_importance(xgb_model, importance_type='weight')
plt.show()

# Prepare future data
X_future = future_data[features]


# One-hot encoding for future data
X_future_encoded = pd.get_dummies(X_future, columns=['venue_home', 'team_home', 'against_home', 'competition_id_home'], drop_first=True)
X_future_encoded = X_future_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
X_future_scaled = scaler.transform(X_future_encoded)

# Make predictions on future data
future_predictions = xgb_model.predict(X_future_scaled)
future_probabilities = xgb_model.predict_proba(X_future_scaled)

# Map predictions and probabilities
result_mapping = {2: 'Win', 1: 'Draw', 0: 'Loss'}
future_data['predicted_win_or_loss_home'] = future_predictions
future_data['predicted_win_or_loss_home'] = future_data['predicted_win_or_loss_home'].map(result_mapping)
future_data['prob_win'] = (future_probabilities[:, 2] * 100).round(0).astype(int)
future_data['prob_draw'] = (future_probabilities[:, 1] * 100).round(0).astype(int)
future_data['prob_loss'] = (future_probabilities[:, 0] * 100).round(0).astype(int)
print('bootstrap')
# Bootstrapping for confidence intervals
# Convert to NumPy arrays (if they aren't already)
X_train_np = np.asarray(X_train_scaled)
y_np = np.asarray(y)

n_bootstrap = 100
boot_probabilities = np.zeros((len(X_future_scaled), n_bootstrap, 3))  # samples x iterations x classes

for i in range(n_bootstrap):
    print(f"Bootstrap iteration {i+1}/{n_bootstrap}")
    
    # Faster resampling
    idx = np.random.randint(0, len(X_train_np), len(X_train_np))
    X_resampled = X_train_np[idx]
    y_resampled = y_np[idx]

    # Train new XGBoost model
    xgb_model_boot = xgb.XGBClassifier(
        eval_metric='logloss',
        max_depth=3,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        alpha=5,
        gamma=0.2,
        reg_lambda=5,
        n_estimators=200,
        learning_rate=0.3,
        
    )
    xgb_model_boot.fit(X_resampled, y_resampled)

    # Predict probabilities for future data
    boot_probabilities[:, i, :] = xgb_model_boot.predict_proba(X_future_scaled)

# Compute confidence intervals (5th and 95th percentiles for 95% CI)
lower_bounds = np.percentile(boot_probabilities, 5, axis=1)
upper_bounds = np.percentile(boot_probabilities, 95, axis=1)

# Store confidence intervals in dataframe
future_data['prob_win_lower'] = (lower_bounds[:, 2] * 100).round(0).astype(int)
future_data['prob_win_upper'] = (upper_bounds[:, 2] * 100).round(0).astype(int)
future_data['prob_draw_lower'] = (lower_bounds[:, 1] * 100).round(0).astype(int)
future_data['prob_draw_upper'] = (upper_bounds[:, 1] * 100).round(0).astype(int)
future_data['prob_loss_lower'] = (lower_bounds[:, 0] * 100).round(0).astype(int)
future_data['prob_loss_upper'] = (upper_bounds[:, 0] * 100).round(0).astype(int)
# Mean probabilities across bootstrap samples
mean_probs = np.mean(boot_probabilities, axis=1)
future_data['prob_win_mean'] = (mean_probs[:, 2] * 100).round(0).astype(int)
future_data['prob_draw_mean'] = (mean_probs[:, 1] * 100).round(0).astype(int)
future_data['prob_loss_mean'] = (mean_probs[:, 0] * 100).round(0).astype(int)

# Median probabilities across bootstrap samples
median_probs = np.median(boot_probabilities, axis=1)
future_data['prob_win_median'] = (median_probs[:, 2] * 100).round(0).astype(int)
future_data['prob_draw_median'] = (median_probs[:, 1] * 100).round(0).astype(int)
future_data['prob_loss_median'] = (median_probs[:, 0] * 100).round(0).astype(int)


# Select final columns for output
output_data = future_data[['timestamp', 'team_home', 'team_away', 'predicted_win_or_loss_home',
                           'prob_win', 'prob_win_lower', 'prob_win_upper',
                           'prob_draw', 'prob_draw_lower', 'prob_draw_upper',
                           'prob_loss', 'prob_loss_lower', 'prob_loss_upper',
                           'win_or_loss_home', 
                           'prob_win_mean', 'prob_draw_mean', 'prob_loss_mean',
                           'prob_win_median', 'prob_draw_median', 'prob_loss_median']]


# Initialize the new columns with placeholder values (None or NaN)
output_data['win_odds'] = None   # You can replace None with a default value like NaN if needed
output_data['draw_odds'] = None  # Or any other default value
output_data['loss_odds'] = None  # Same here

# Save the updated future_data to a CSV file
output_data.to_csv("ci.csv", index=False)

print("Predictions and probabilities added and saved to 'match_predictions.csv'")