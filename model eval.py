import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV


from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.calibration import calibration_curve

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from xgboost import plot_importance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import log_loss
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import shapiro


# Load your dataset
data = pd.read_csv("C:/Users/archi/Desktop/sports betting/Saves data/cleaned-data.csv")




data = data.drop_duplicates()


# Filter rows with sufficient games played
data = data[data['games_played_home_total'] > 5]
data = data[data['games_played_away_total'] > 5]

print("1")

# Handle missing values
data['venue_home'] = data['venue_home'].fillna('Unknown')
data['venue_away'] = data['venue_away'].fillna('Unknown')

# Ensure timestamp column is in datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')

cutoff = '2024-01-01'
# Split past and future data
data = data[data['timestamp'] < pd.Timestamp.now()]
data = data[data['timestamp'] > cutoff]

data['win_or_loss_home'] = pd.to_numeric(data['win_or_loss_home'], errors='coerce')
data['is_draw_home'] = data['win_or_loss_home'].apply(lambda x: 1 if x == 2 else 0)


features = [
    'venue_home', 
    'team_home', 
    'against_home',
    'competition_id_home',
    'points_before_game_home',
    'conceded_before_game_home', 
    'goals_before_game_home',
    'games_to_points_ratio_home_home', 
    'games_to_points_ratio_away_home',
    'last_games_elo_home', 
    'against_last_elo_home', 
    'elo_diff_home',
    'against_points_before_game_home', 
    'against_conceded_before_game_home',
    'against_goals_before_game_home',
    'against_games_to_points_ratio_home_home',
    'against_games_to_points_ratio_away_home', 
    'points_before_game_away',
    'conceded_before_game_away', 
    'goals_before_game_away',
    'last_games_elo_away', 
    'against_points_before_game_away',
    'against_conceded_before_game_away', 
    'against_goals_before_game_away',
    'max_elo_so_far_away',
    'min_elo_so_far_away', 
    'median_elo_so_far_away',
    'mean_elo_so_far_away', 
    'std_elo_so_far_away',
    'max_elo_so_far_home',
    'min_elo_so_far_home', 
    'median_elo_so_far_home',
    'mean_elo_so_far_home', 
    'std_elo_so_far_home',
    'season'
]

X = data[features]
y = data['win_or_loss_home']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode again to ensure consistency across train, test, and the full dataset
X_train_encoded = pd.get_dummies(X_train, columns=['venue_home', 'team_home', 'against_home', 'competition_id_home','season'], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=['venue_home', 'team_home', 'against_home', 'competition_id_home','season'], drop_first=True)


X_train_encoded = X_train_encoded.loc[:, ~X_train_encoded.columns.duplicated()]
X_test_encoded = X_test_encoded.loc[:, ~X_test_encoded.columns.duplicated()]

X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

X_train_scaled = X_train_encoded
X_test_scaled = X_test_encoded

# Train the base XGBoost model
import xgboost as xgb
print("1")
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',  # Keeps the same metric for multi-class classification
    max_depth=4,  # Reduce depth to prevent overfitting
    min_child_weight=10,  # Increase to require more samples to split
    subsample=0.7,  # Lower subsample to introduce randomness
    colsample_bytree=0.6,  # Reduce column sampling for additional regularization
    alpha=5,  # Increase L1 regularization to penalize large coefficients
    gamma=1,  # Add a minimum split loss to prevent overfitting small splits
    reg_lambda=10,  # Increase L2 regularization for smoother weights
    n_estimators=200,  # Reduce number of trees to avoid overfitting
    learning_rate=0.05  # Lower learning rate for more gradual learning
)



# Fit the model
xgb_model.fit(X_train_scaled, y_train)

print("1")
# try platt and cross validation
# Predictions on test data
y_pred = xgb_model.predict(X_test_scaled)
y_proba = xgb_model.predict_proba(X_test_scaled)
y_pred_train = xgb_model.predict(X_train_scaled)
y_proba_train = xgb_model.predict_proba(X_train_scaled)


print(f"test Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"train Accuracy: {accuracy_score(y_train, y_pred_train)}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

#print(f"ROC AUC: {roc_auc_score(y_test, y_proba, multi_class='ovr')}")
print(f"Log Loss: {log_loss(y_test, y_proba)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")


#scores = cross_val_score(xgb_model, X_train_encoded, y, cv=5, scoring='accuracy')  # Example for classification
#print(f"Cross-validated accuracy: {scores.mean()} ± {scores.std()}")


from sklearn.metrics import brier_score_loss
for i in range(3):
    brier = brier_score_loss(y_test == i, y_proba[:, i])
    print(f"Brier Score for test class {i}: {brier}")
    brier = brier_score_loss(y_train == i, y_proba_train[:, i])
    print(f"Brier Score for train class {i}: {brier}")



pd.Series(y_pred).value_counts(normalize=True)


from sklearn.metrics import precision_recall_curve
for i in range(3):
    precision, recall, _ = precision_recall_curve(y_test == i, y_proba[:, i])
    plt.plot(recall, precision, label=f"Class {i}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()



# Predictions on train data
y_train_pred = xgb_model.predict(X_train_scaled)
y_train_proba = xgb_model.predict_proba(X_train_scaled)

# Combine predictions and probabilities from train and test
combined_y_true = np.concatenate([y_train, y_test])
combined_y_proba = np.vstack([y_train_proba, y_proba])

# Plot Calibration Curve for each class
plt.figure(figsize=(10, 8))

# Loop through each class (0, 1, 2)
for i in range(3):  # Assuming you have 3 classes: Loss (0), Draw (1), Win (2)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        combined_y_true == i,  # Treat class `i` as the positive class
        combined_y_proba[:, i],  # Get the probabilities for class `i`
        n_bins=100
    )
    
    # Plot each class's calibration curve
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f"Class {i} Calibration Curve")

# Plot a perfectly calibrated line (diagonal)
plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly calibrated", color='gray')

plt.title('Calibration Curve for Platt Calibrated Model Prediction (Train and Test)')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.legend()
plt.grid(True)
plt.show()

