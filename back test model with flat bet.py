import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from datetime import datetime

# Load your dataset
data = pd.read_csv("historicl data")
data = data.drop_duplicates()

na_rows = data[data.isna().any(axis=1)]
#print(na_rows.head())



# Filter rows with sufficient games played
data = data[data['games_played_home_total'] > 5]
data = data[data['games_played_away_total'] > 5]

# Handle missing values
data['venue_home'] = data['venue_home'].fillna('Unknown')
data['venue_away'] = data['venue_away'].fillna('Unknown')

# Ensure timestamp column is in datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')

# Sort data by timestamp in ascending order
data = data.sort_values(by='timestamp')

# Select features and target variable
X = data[[
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
          'season',
          'year',
          'against_goals_before_game_away'
          ]]

y = data['win_or_loss_home']

# Store odds data separately
odds_columns = ['B365H_home', 'B365D_home', 'B365A_home','competition_id_home','season','timestamp']
odds_data = data[odds_columns]


# Calculate the split index (70% for training, 30% for testing)
split_index = int(len(data) * 0.7)

# Split the data based on the calculated index
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
odds_train, odds_test = odds_data.iloc[:split_index], odds_data.iloc[split_index:]

# One-hot encoding for categorical variables
X_train_encoded = pd.get_dummies(X_train, columns=['team_home', 'against_home', 'competition_id_home','season'], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=['team_home', 'against_home', 'competition_id_home','season'], drop_first=True)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

xgb_model = xgb.XGBClassifier(
    eval_metric='logloss',  # Keeps the same metric for multi-class classification
    max_depth=4,  # Reduce depth to prevent overfitting
    min_child_weight=10,  # Increase to require more samples to split
    subsample=0.7,  # Lower subsample to introduce randomness
    colsample_bytree=0.6,  # Reduce column sampling for additional regularization
    alpha=5,  # Increase L1 regularization to penalize large coefficients
    gamma=1,  # Add a minimum split loss to prevent overfitting small splits
    reg_lambda=10,  # Increase L2 regularization for smoother weights
    n_estimators=150,  # Reduce number of trees to avoid overfitting
    learning_rate=0.05  # Lower learning rate for more gradual learning
)

xgb_model.fit(X_train_scaled, y_train)

# Calibrate the model using isotonic regression
calibrated_model = CalibratedClassifierCV(estimator=xgb_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_train_scaled, y_train)

# Predictions on test data
y_pred = xgb_model.predict(X_test_scaled)
y_proba = xgb_model.predict_proba(X_test_scaled)

print(classification_report(y_test, y_pred))

# Get the predicted probabilities (for all three classes: loss, draw, win)
y_proba_df = pd.DataFrame(y_proba, columns=['probability_loss', 'probability_draw', 'probability_win'])
print(y_proba_df)

# Combine the odds data with the predicted probabilities
test_with_probabilities = X_test.copy()

# Reset indices to avoid NaN values due to misalignment
test_with_probabilities.reset_index(drop=True, inplace=True)
y_proba_df.reset_index(drop=True, inplace=True)
odds_data.reset_index(drop=True, inplace=True)

# Add the predicted probabilities
test_with_probabilities['probability_loss'] = y_proba_df['probability_loss']
test_with_probabilities['probability_draw'] = y_proba_df['probability_draw']
test_with_probabilities['probability_win'] = y_proba_df['probability_win']

# Add the actual labels from the test set for comparison
test_with_probabilities['actual_label'] = y_test.reset_index(drop=True)

# Add the odds data to the DataFrame
test_with_probabilities['B365H_home'] = odds_data['B365H_home']
test_with_probabilities['B365D_home'] = odds_data['B365D_home']
test_with_probabilities['B365A_home'] = odds_data['B365A_home']
test_with_probabilities['competition_id_home'] = odds_data['competition_id_home']
test_with_probabilities['timestamp'] = odds_data['timestamp']

# Select the specific columns you want to save
columns_to_save = ['team_home', 
                   'against_home', 
                   'probability_loss', 
                   'probability_draw', 
                   'probability_win', 
                   'actual_label', 
                   'B365H_home', 
                   'B365D_home', 
                   'B365A_home',
                   'competition_id_home',
                   'season',
                   'timestamp']

# Filter the DataFrame to include only the selected columns
test_with_probabilities_filtered = test_with_probabilities[columns_to_save]

# bet selection
#############################################################################################################

df = test_with_probabilities_filtered
# Ensure odds columns are numeric
df[['B365H_home', 'B365D_home', 'B365A_home']] = df[['B365H_home', 'B365D_home', 'B365A_home']].apply(pd.to_numeric, errors='coerce')

print(df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


# Assuming your data is loaded in a pandas DataFrame called 'df'
df['EV_win'] = (df['probability_win'] * df['B365H_home']) - (1-df['probability_win'])*1
df['EV_draw'] = (df['probability_draw'] * df['B365D_home']) - (1-df['probability_draw'])*1
df['EV_loss'] = (df['probability_loss'] * df['B365A_home']) - (1-df['probability_loss'])*1

# Create a column 'best_bet' that holds the outcome with the highest EV
df['best_bet'] = df[['EV_win', 'EV_draw', 'EV_loss']].idxmax(axis=1)

# Create a new column 'best_bet_ev' that contains the EV of the best bet
df['best_bet_ev'] = df.apply(lambda row: row['EV_win'] if row['best_bet'] == 'EV_win' else
                             (row['EV_draw'] if row['best_bet'] == 'EV_draw' else
                              row['EV_loss']), axis=1)

def calculate_edge(row):
    if row['best_bet'] == 'EV_win':
        return row['probability_win'] - (1 / row['B365H_home'])
    elif row['best_bet'] == 'EV_draw':
        return row['probability_draw'] - (1 / row['B365D_home'])
    elif row['best_bet'] == 'EV_loss':
        return row['probability_loss'] - (1 / row['B365A_home'])
    else:  # Assuming 'away' is the only other option
        return 99999999


def best_bet(row):
    if row['best_bet'] == 'EV_win':
        return row['B365H_home']
    elif row['best_bet'] == 'EV_draw':
        return ['B365D_home']
    else:  # Assuming 'away' is the only other option
        return ['B365A_home']

df['selected_odds'] = df.apply(best_bet, axis=1)


df['percent_diff'] = df.apply(calculate_edge, axis=1)
#df = df[df['percent_diff'] > 0]
# Filter rows where 'best_bet_ev' is positive
df = df[df['best_bet_ev'] > 0]

test = df[df['percent_diff'] < 0]

print(df)
def calculate_winnings(row):
    if row['best_bet'] == 'EV_win':
        if row['actual_label'] == 2:
            return 1 * row['B365H_home']
        else:
            return -1
    elif row['best_bet'] == 'EV_draw':
        if row['actual_label'] == 1:
            return 1 * row['B365D_home']
        else:
            return -1
    elif row['best_bet'] == 'EV_loss':
        if row['actual_label'] == 0:
            return 1 * row['B365A_home']
        else:
            return -1
    else:
        return 0  # In case no bet is made (shouldn't happen if all rows have a best_bet)


# Apply the function to create the 'winnings' column
df['winnings'] = df.apply(calculate_winnings, axis=1)

total_winnings = df['winnings'].sum()
# Display the total winnings
print("Total Winnings:", total_winnings)
print("Total Winnings per bet:", total_winnings/10709)

df = df.sort_values(by='timestamp')  # Sort by timestamp
df['sum_win'] = df['winnings'].cumsum()

# Group by 'competition_id_home' and calculate cumulative sum of winnings for each competition (league)
df['com_sum'] = df.groupby('competition_id_home')['winnings'].cumsum()

# Plotting total winnings for each competition over timestamp
plt.figure(figsize=(10, 6))  # Optional: Customize the figure size

# Loop through each unique competition and plot its cumulative winnings over timestamp
for competition in df['competition_id_home'].unique():
    competition_data = df[df['competition_id_home'] == competition]  # Filter data for each competition
    plt.plot(competition_data['timestamp'], competition_data['com_sum'], label=f"Competition {competition}")

# Add labels and title
plt.xlabel('Timestamp')
plt.ylabel('Total Winnings')
plt.title('Total Winnings Over Time by Competition')
plt.legend()  # Show legend for each competition
# Rotate timestamp labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()



# Initialize the figure for plotting
plt.figure(figsize=(10, 8))

# For each class (loss = 0, draw = 1, win = 2), plot a calibration curve
for class_label, prob_column in zip([0, 1, 2], ['probability_loss', 'probability_draw', 'probability_win']):
    # Get the true labels and predicted probabilities for the current class
    y_true = (df['actual_label'] == class_label).astype(int)  # Create binary labels for the class
    y_probs = df[prob_column]  # Get the predicted probabilities for the class
    
    # Only proceed if the probability is non-zero
    if np.any(y_probs > 0):  # Check if there are any non-zero probabilities
        # Compute the calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_probs, n_bins=10)
        
        # Plot the calibration curve for this class
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'Class {class_label} Calibration Curve')
    
# Plot a reference line (Perfect Calibration)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

# Customize the plot
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curves for Each Class')
plt.legend(loc='best')
plt.grid(True)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Create the figure and axis
fig, ax1 = plt.subplots()

# Scatter plot on the first axis
sns.scatterplot(x=df['percent_diff'], y=df['winnings'], alpha=0.7, ax=ax1)

# Create a second y-axis
ax2 = ax1.twinx()

# Filter rows where winnings is -1
filtered_df = df[df['winnings'] == -1]

# Proportional distribution plot for percent_diff where winnings = -1 on the second axis
sns.histplot(filtered_df['percent_diff'], kde=True, bins=30, stat='probability', ax=ax2, color='orange')

# Customize labels
ax1.set_xlabel('Percentage Difference')
ax1.set_ylabel('Winnings')
ax2.set_ylabel('Proportion of losses')

# Set title
plt.title('Scatter Plot with Proportional Distribution and Secondary Y-axis')
plt.show()

# Create the table as a DataFrame
table = pd.DataFrame({
    'odds': df['selected_odds'],
    'winnings': df['winnings'],
    'percent_diff': df['percent_diff']
})

import pandas as pd
import numpy as np

# Define the bins for percent_diff groups
bins = [0.1, 0.11, 0.2, 0.21, 0.3, 0.31, 0.4, 0.41, np.inf]
labels = ['0.1-0.11', '0.11-0.2', '0.2-0.21', '0.21-0.3', '0.3-0.31', '0.31-0.4', '0.4-0.41', '0.41-plus']

# Create the groups for percent_diff
table['percent_diff_group'] = pd.cut(table['percent_diff'], bins=bins, labels=labels, right=False)
# Count the number of rows where winnings = -1 for each group
grouped_counts = table[table['winnings'] == -1].groupby('percent_diff_group').size()

# Get the total number of rows in each group
total_counts = table.groupby('percent_diff_group').size()

# Normalize the counts by dividing the number of -1 winnings by the total rows in the group
normalized_counts = grouped_counts / total_counts
import matplotlib.pyplot as plt

# Plot the normalized distribution
plt.bar(normalized_counts.index, normalized_counts, alpha=0.7, color='blue')
plt.xlabel('Percentage Difference Group')
plt.ylabel('Normalised Count of losses')
plt.title('Normalised Distribution of Percentage Difference (for losses)')
plt.xticks(rotation=45)
plt.show()


# Create the table as a DataFrame
table = pd.DataFrame({
    'odds': df['selected_odds'],
    'winnings': df['winnings'],
    'percent_diff': df['percent_diff']
})

# Ensure odds column is numeric
table['odds'] = pd.to_numeric(table['odds'], errors='coerce')

# Remove rows with NaN values after conversion (if any)
table = table.dropna(subset=['odds'])

# Define bins and labels for odds
odds_bins = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, float('inf')]
odds_labels = ['1-1.25', '1.26-1.5', '1.51-1.75', '1.76-2', '2.1-2.25', '2.26-2.5', '2.52+']

# Create groups for odds
table['odds_group'] = pd.cut(table['odds'], bins=odds_bins, labels=odds_labels, right=False)


# Count the number of rows where winnings = -1 for each group
grouped_counts = table[table['winnings'] == -1].groupby('odds_group').size()

# Get the total number of rows in each group
total_counts = table.groupby('odds_group').size()

# Normalize the counts by dividing the number of -1 winnings by the total rows in the group
normalized_counts = grouped_counts / total_counts

# Plot the normalized distribution
plt.bar(normalized_counts.index, normalized_counts, alpha=0.7, color='blue')
plt.xlabel('Odds Group')
plt.ylabel('Normalised Count of losses')
plt.title('Normalised Distribution of Odds (for losses)')
plt.xticks(rotation=45)
plt.show()