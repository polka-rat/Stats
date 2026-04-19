import pandas as pd
import sys
from io import StringIO

# Load the main script to get access to the predict function
exec(open('main.py').read())

# Load the cleaned data
print("Loading cleaned data...")
df = pd.read_csv('ipl_combined_cleaned.csv')

# Get the first match's first innings, first 18 balls (3 overs)
first_match_data = df[(df['match_id'] == df['match_id'].iloc[0]) & (df['innings'] == 1)].head(18)

print(f"\nExtracted {len(first_match_data)} balls for testing")
print(f"Match ID: {first_match_data['match_id'].iloc[0]}")
print(f"Teams: {first_match_data['batting_team'].iloc[0]} vs {first_match_data['bowling_team'].iloc[0]}")
print(f"Ball range: {first_match_data['ball'].min()} to {first_match_data['ball'].max()}")

# Prepare the ball-by-ball CSV string format
# Select only the columns needed by predict function
required_cols = ['ball', 'striker', 'non_striker', 'runs_off_ball', 'extras', 'wides', 'noballs', 'byes', 'legbyes', 'wicket_type', 'player_dismissed']
ball_data = first_match_data[required_cols].copy()

# Convert to CSV string
csv_string = ball_data.to_csv(index=False)

print("\nBall-by-ball data (first 5 balls):")
print(csv_string.split('\n')[0:6])

# Run prediction
print("\n" + "="*60)
print("Running prediction...")
print("="*60)
result = predict("Test Match", csv_string)

print("\n" + "="*60)
print(f"PREDICTION RESULT: {result}")
print("="*60)
