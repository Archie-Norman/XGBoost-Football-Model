import requests
import pandas as pd
import time
import os

# Define your RapidAPI key and headers
api_key = '51e0fd3f97mshe4c041f9b0ad8e6p1c7e6cjsn25f270470d84'
comp_ids = range(1, 161)  # Competition IDs from 1 to 160
headers = {
    'X-RapidAPI-Key': api_key,
    'X-RapidAPI-Host': 'football-web-pages1.p.rapidapi.com'
}

# Define the base URL for the Fixtures/Results endpoint
base_url = 'https://football-web-pages1.p.rapidapi.com/fixtures-results.json'

# Check if the CSV file exists and load existing data
csv_filename = "api-data.csv"
if os.path.exists(csv_filename):
    existing_data = pd.read_csv(csv_filename)
else:
    existing_data = pd.DataFrame()

# Initialize an empty list to store new results
new_results = []

# Loop through each competition ID
for comp_id in comp_ids:
    params = {'comp': comp_id}

    while True:  # Handle rate limits
        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code == 429:
            print("Rate limit reached. Waiting for 60 seconds before retrying...")
            time.sleep(70)
            continue

        if response.status_code == 200:
            data = response.json()
            break
        else:
            print(f"Failed to fetch fixtures/results for competition ID {comp_id}. Status code: {response.status_code}")
            print("Response Content:", response.text)
            break

    fixtures_results = data.get('fixtures-results', {}).get('matches', [])

    if fixtures_results:
        df = pd.DataFrame(fixtures_results)
        home_teams = df['home-team'].apply(pd.Series)
        away_teams = df['away-team'].apply(pd.Series)

        columns_to_include = ['date', 'time']
        if 'venue' in df.columns:
            columns_to_include.append('venue')

        results = pd.concat([
            df[columns_to_include], 
            home_teams[['name', 'score']].rename(columns={'name': 'home_team', 'score': 'home_score'}), 
            away_teams[['name', 'score']].rename(columns={'name': 'away_team', 'score': 'away_score'})
        ], axis=1)
        results['competition_id'] = comp_id

        new_results.append(results)
        print(f"Fetched fixtures/results for competition ID {comp_id}")

# Convert the list of new results into a DataFrame
if new_results:
    new_data = pd.concat(new_results, ignore_index=True)

    # Combine old and new data
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Remove duplicates, keeping the most recent entry
    combined_data = combined_data.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')

    # Save back to CSV
    combined_data.to_csv(csv_filename, index=False)

    print("Updated CSV with new data, removing duplicates.")
else:
    print("No new data to update.")
