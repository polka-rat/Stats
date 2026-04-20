import pandas as pd
import os

# Get directory where script is located
dir = os.path.dirname(os.path.abspath(__file__))

# Read the CSV
csv_path = os.path.join(dir, "ipl_combined_cleaned.csv")
df = pd.read_csv(csv_path)

# Convert ball numbers from 1-120 to 0.1-19.6 format
def convert_ball_number(ball_num):
    """Convert ball number (1-120) to cricket over format (0.1-19.6)"""
    ball_num = int(ball_num)
    over = (ball_num - 1) // 6
    ball_in_over = (ball_num - 1) % 6 + 1
    return float(f"{over}.{ball_in_over}")

# Apply conversion
df['ball'] = df['ball'].apply(convert_ball_number)

# Save back to CSV
df.to_csv(csv_path, index=False)
print(f"Converted ball format in {csv_path}")
print(f"Sample conversions:")
print(df[['ball', 'batting_team', 'bowler']].head(10))
