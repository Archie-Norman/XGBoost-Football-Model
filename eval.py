import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from xgboost import plot_importance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import log_loss
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import shapiro

# Load your dataset
data = pd.read_csv("C:/Users/archi/Desktop/sports betting/sorted/delete.csv")
data = data.drop_duplicates()

# Filter rows with sufficient games played
data = data[data['games_played_home_total'] > 5]
data = data[data['games_played_away_total'] > 5]

# Handle missing values
data['venue_home'] = data['venue_home'].fillna('Unknown')
data['venue_away'] = data['venue_away'].fillna('Unknown')

# Ensure timestamp column is in datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')

data = data[data['timestamp'] < pd.Timestamp.now()]




#        taking the log
##########################################################################################################################

# Function to apply transformations and check normality
def find_best_transformation(data):
    results = {}
    transformations = {
        'log': lambda x: np.log1p(x),  # Log transformation
        'sqrt': lambda x: np.sqrt(x),  # Square root transformation
        'reciprocal': lambda x: 1 / (x + 1e-9),  # Reciprocal (avoid zero division)
        'original': lambda x: x  # No transformation
    }
    
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        best_p_value = 0
        best_transform = 'original'
        
        for name, func in transformations.items():
            try:
                transformed = func(data[column].clip(lower=0))  # Clip to handle negative values
                stat, p_value = shapiro(transformed)  # Shapiro-Wilk normality test
                
                if p_value > best_p_value:  # Choose transformation with best p-value
                    best_p_value = p_value
                    best_transform = name
            except Exception as e:
                print(f"Skipping transformation {name} on {column}: {e}")
        
        results[column] = (best_transform, best_p_value)
    
    return results

# Run the function to find the best transformations
best_transforms = find_best_transformation(data)

# Apply the best transformations to the data
def apply_best_transformations(data, best_transforms):
    transformed_df = data.copy()  # Copy original data to avoid modifying it
    transformations = {
        'log': lambda x: np.log1p(x),
        'sqrt': lambda x: np.sqrt(x),
        'reciprocal': lambda x: 1 / (x + 1e-9),
        'original': lambda x: x
    }
    
    for column, (transform, _) in best_transforms.items():
        transformed_df[column] = transformations[transform](transformed_df[column].clip(lower=0))
    
    return transformed_df

# Apply the transformations
transformed_df = apply_best_transformations(data, best_transforms)

data = transformed_df



##########################################################################################################################





#       balancing
##########################################################################################################################

# Extract the win_or_loss_home column
win_or_loss_home = data['win_or_loss_home']

# Count occurrences of 0, 1, and 2
count_0 = (win_or_loss_home == 0).sum()
count_1 = (win_or_loss_home == 1).sum()
count_2 = (win_or_loss_home == 2).sum()

# Print the results
print(f"Count of 0: {count_0}")
print(f"Count of 1: {count_1}")
print(f"Count of 2: {count_2}")

# Optional: Total sum of counts
total_sum = count_1 / ((count_0 + count_2) * 2)
print(f"Total sum: {total_sum}")

##############################################################################

# Determine which class has more entries and how many to delete
if count_0 > count_2:
    excess_count = count_0 - count_2
    # Get indices of rows where win_or_loss_home == 0
    count_0_indices = data[data['win_or_loss_home'] == 0].index.tolist()
    # Shuffle the indices to delete randomly
    np.random.shuffle(count_0_indices)
    # Drop the excess count 0 entries randomly
    data = data.drop(count_0_indices[:excess_count])  # Ensure data is updated
elif count_2 > count_0:
    excess_count = count_2 - count_0
    # Get indices of rows where win_or_loss_home == 2
    count_2_indices = data[data['win_or_loss_home'] == 2].index.tolist()
    # Shuffle the indices to delete randomly
    np.random.shuffle(count_2_indices)
    # Drop the excess count 2 entries randomly
    data = data.drop(count_2_indices[:excess_count])  # Ensure data is updated

# Verify the new count of 0 and 2
new_count_0 = (data['win_or_loss_home'] == 0).sum()
new_count_2 = (data['win_or_loss_home'] == 2).sum()

print(f"New count of 0: {new_count_0}")
print(f"New count of 2: {new_count_2}")

##############################################################################
# Total data size
total_count = len(data)

targ_num = total_count * total_sum

#total_sum = 0.05

print(f"Target percentage for count 1: {total_sum * 100:.2f}%")
print(f"Total data points: {total_count}")
print(f"Target number of draws: {targ_num}")

desired_count_1 = int(total_sum * total_count)

# If you want random deletion, shuffle the indices
if count_1 > desired_count_1:
    excess_count_1 = count_1 - desired_count_1
    # Get indices of rows where win_or_loss_home == 1
    count_1_indices = data[data['win_or_loss_home'] == 1].index.tolist()  # Convert to list
    # Shuffle the list of indices to delete randomly
    np.random.shuffle(count_1_indices)
    # Drop the excess count 1 entries randomly
    data = data.drop(count_1_indices[:excess_count_1])  # Ensure data is updated

# Verify the new count of 1 and its percentage
new_count_1 = (data['win_or_loss_home'] == 1).sum()
new_total_count = len(data)
new_percentage_1 = new_count_1 / new_total_count * 100

print(f"New count of 1: {new_count_1}")
print(f"New percentage of count 1: {new_percentage_1:.2f}%")

# Extract the win_or_loss_home column again to verify changes
win_or_loss_home = data['win_or_loss_home']

# Count occurrences of 0, 1, and 2 after deletion
count_0 = (win_or_loss_home == 0).sum()
count_1 = (win_or_loss_home == 1).sum()
count_2 = (win_or_loss_home == 2).sum()

# Print the final results
print(f"Final Count of 0: {count_0}")
print(f"Final Count of 1: {count_1}")
print(f"Final Count of 2: {count_2}")


##########################################################################################################################

# Select features and target variable
X = data[['venue_home', 
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
          'season',
          'year']]

y = data['win_or_loss_home']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode again to ensure consistency across train, test, and the full dataset
X_train_encoded = pd.get_dummies(X_train, columns=['venue_home', 'team_home', 'against_home', 'competition_id_home','season'], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=['venue_home', 'team_home', 'against_home', 'competition_id_home','season'], drop_first=True)
X_encoded = pd.get_dummies(X, columns=['venue_home', 'team_home', 'against_home', 'competition_id_home','season'], drop_first=True)


# Ensure that X_test_encoded matches X_train_encoded in column names (important for scaling)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
X_encoded = X_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Now scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)
X_scaled = scaler.transform(X_encoded)


# Train the base XGBoost model
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',
    max_depth=3,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_lambda=5,
    n_estimators=200,
    learning_rate=0.3
)

# Fit the model using sample weights for class balancing
xgb_model.fit(X_train_scaled, y_train)

# Calibrate the model using isotonic regression
calibrated_model = CalibratedClassifierCV(estimator=xgb_model, method='sigmoid', cv='prefit')

# try platt and cross validation

calibrated_model.fit(X_train_scaled, y_train)

# Predictions on test data
y_pred = calibrated_model.predict(X_test_scaled)
y_proba = calibrated_model.predict_proba(X_test_scaled)


print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba, multi_class='ovr')}")
print(f"Log Loss: {log_loss(y_test, y_proba)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")

scores = cross_val_score(xgb_model, X_scaled, y, cv=5, scoring='accuracy')  # Example for classification
print(f"Cross-validated accuracy: {scores.mean()} ± {scores.std()}")

#ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Loss', 'Draw', 'Win'])
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Display the confusion matrix with labels
print("Confusion Matrix:")
print(f"              Predicted Loss   Predicted Draw   Predicted Win")
print(f"Actual Loss     {cm[0, 0]:>5}              {cm[0, 1]:>5}              {cm[0, 2]:>5}")
print(f"Actual Draw     {cm[1, 0]:>5}              {cm[1, 1]:>5}              {cm[1, 2]:>5}")
print(f"Actual Win      {cm[2, 0]:>5}              {cm[2, 1]:>5}              {cm[2, 2]:>5}")


from sklearn.metrics import brier_score_loss
for i in range(3):
    brier = brier_score_loss(y_test == i, y_proba[:, i])
    print(f"Brier Score for class {i}: {brier}")


# Set feature names as a list for the model's booster
xgb_model.get_booster().feature_names = list(X_train_encoded.columns)

# Plot feature importance
xgb.plot_importance(xgb_model, importance_type='weight')
plt.show()


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
y_train_pred = calibrated_model.predict(X_train_scaled)
y_train_proba = calibrated_model.predict_proba(X_train_scaled)

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
        n_bins=50
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
        n_bins=50
    )
    
    # Plot each class's calibration curve
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f"Class {i} Calibration Curve")

# Plot a perfectly calibrated line (diagonal)
plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly calibrated", color='gray')

plt.title('Calibration Curve for Raw XGBoost Model Prediction (Train and Test)')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.legend()
plt.grid(True)
plt.show()

# over fitting ? and should i use the log funtion of anything 