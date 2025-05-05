max_exposure = 10
max_bet = 0.05 * max_exposure  # 2% of bankroll

import pandas as pd
import os

# Load the data from a CSV file
df = pd.read_csv("C:/Users/archi/Desktop/sports betting/Saves data/ci.csv")


# Convert odds columns to numeric and check for errors
odds_columns = ["win_odds", "draw_odds", "loss_odds"]
invalid_rows = df[~df[odds_columns].applymap(lambda x: str(x).replace('.', '', 1).isdigit()).all(axis=1)]

# Display or save invalid rows
print(invalid_rows)

df1 = df



def kelly_fraction(p,o):
    p = p/100
    kelly_fraction = (p * (o - 1) - (1 -p)) / (o-1)
    return kelly_fraction

# cal kelly 
df['kelly_win'] = df.apply(lambda row: kelly_fraction(row['prob_win_lower'], row['win_odds']), axis=1)
df['kelly_draw'] = df.apply(lambda row: kelly_fraction(row['prob_draw_lower'], row['draw_odds']), axis=1)
df['kelly_loss'] = df.apply(lambda row: kelly_fraction(row['prob_loss_lower'], row['loss_odds']), axis=1)

# Add columns to identify the MAX Kelly fraction and its type (Win/Draw/Loss)
df['max_kelly'] = df[['kelly_win', 'kelly_draw', 'kelly_loss']].max(axis=1)
df['best_bet'] = df[['kelly_win', 'kelly_draw', 'kelly_loss']].idxmax(axis=1).str.replace('kelly_', '')

# filter  out anything with negatice kelly
df = df[df['max_kelly'] > 0]


def cal_ev(p,o):
    p = p / 100
    ev = p * o - 1
    return ev

df['ev_win'] = df.apply(lambda row: cal_ev(row['prob_win_lower'], row['win_odds']), axis=1)
df['ev_draw'] = df.apply(lambda row: cal_ev(row['prob_draw_lower'], row['draw_odds']), axis=1)
df['ev_loss'] = df.apply(lambda row: cal_ev(row['prob_loss_lower'], row['loss_odds']), axis=1)

df['max_ev'] = df[['ev_win', 'ev_draw', 'ev_loss']].max(axis=1)
df = df[df['max_ev'] > 0]

total_kelly = df['max_kelly'].sum()
df['bet_amount'] = (df['max_kelly'] / total_kelly) * max_exposure
df['bet_amount'] = df['bet_amount'].clip(upper=max_bet)
df['bet_amount'] = df['bet_amount'].clip(lower=0.1)
df['date'] = pd.to_datetime(df['timestamp']).dt.date

df['bet_amount'] = df['bet_amount'].round(2)

df = df[['date', 'team_home', 'team_away', 'best_bet','bet_amount']]
print(df)
total_bet = df['bet_amount'].sum()
print(total_bet)


df.to_csv("stake_amount.csv", index=False)


#df.to_csv("stake_history.csv", mode="a", index=False, header=not os.path.exists("stake_history.csv"))
df1.to_csv("full_odds.csv", mode="a", index=False, header=not os.path.exists("full_odds.csv"))
