from io import StringIO
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load trained model and scaler
print("Loading trained model...")
with open('powerplay_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load batsman statistics
print("Loading batsman statistics...")
batsman_stats_df = pd.read_csv('batsman_statistics_2022_2025.csv')
batsman_stats = {}

for _, row in batsman_stats_df.iterrows():
    # If batsman has faced less than 50 powerplay balls, use total stats
    if row['powerplay_balls_faced'] < 50:
        # Use total statistics but name them as powerplay
        batsman_stats[row['batter']] = {
            'powerplay_run_rate': row['run_rate'],
            'powerplay_avg_runs_per_ball': row['avg_runs_per_ball'],
            'powerplay_std_dev_runs_per_ball': row['std_dev_runs_per_ball'],
            'powerplay_strike_rate': row['strike_rate'],
            'powerplay_dismissal_rate_per_100': row['dismissal_rate_per_100'],
            'powerplay_consistency_score': row['consistency_score'],
            'overall_run_rate': row['run_rate'],
            'overall_dismissal_rate': row['dismissal_rate_per_100']
        }
    else:
        # Use powerplay statistics
        batsman_stats[row['batter']] = {
            'powerplay_run_rate': row['powerplay_run_rate'],
            'powerplay_avg_runs_per_ball': row['powerplay_avg_runs_per_ball'],
            'powerplay_std_dev_runs_per_ball': row['powerplay_std_dev_runs_per_ball'],
            'powerplay_strike_rate': row['powerplay_strike_rate'],
            'powerplay_dismissal_rate_per_100': row['powerplay_dismissal_rate_per_100'],
            'powerplay_consistency_score': row['powerplay_consistency_score'],
            'overall_run_rate': row['run_rate'],
            'overall_dismissal_rate': row['dismissal_rate_per_100']
        }

print(f"Loaded statistics for {len(batsman_stats)} batsmen")

# Compute average batsman statistics for fallback
print("Computing average batsman statistics...")
avg_run_rate = np.mean([stats['powerplay_run_rate'] for stats in batsman_stats.values()])
avg_avg_runs_per_ball = np.mean([stats['powerplay_avg_runs_per_ball'] for stats in batsman_stats.values()])
avg_std_dev = np.mean([stats['powerplay_std_dev_runs_per_ball'] for stats in batsman_stats.values()])
avg_strike_rate = np.mean([stats['powerplay_strike_rate'] for stats in batsman_stats.values()])
avg_dismissal_rate = np.mean([stats['powerplay_dismissal_rate_per_100'] for stats in batsman_stats.values()])
avg_consistency = np.mean([stats['powerplay_consistency_score'] for stats in batsman_stats.values()])
avg_overall_run_rate = np.mean([stats['overall_run_rate'] for stats in batsman_stats.values()])
avg_overall_dismissal = np.mean([stats['overall_dismissal_rate'] for stats in batsman_stats.values()])

# Create default/fallback batsman stats using averages
default_batsman_stats = {
    'powerplay_run_rate': avg_run_rate,
    'powerplay_avg_runs_per_ball': avg_avg_runs_per_ball,
    'powerplay_std_dev_runs_per_ball': avg_std_dev,
    'powerplay_strike_rate': avg_strike_rate,
    'powerplay_dismissal_rate_per_100': avg_dismissal_rate,
    'powerplay_consistency_score': avg_consistency,
    'overall_run_rate': avg_overall_run_rate,
    'overall_dismissal_rate': avg_overall_dismissal
}

print(f"Average stats computed for fallback")


def get_batting_state_at_over(match_balls, target_over):
    """Get batting state after overs [0, target_over)."""
    
    runs_so_far = 0
    wickets_so_far = 0
    balls_count = 0
    striker = None
    non_striker = None
    
    for ball in match_balls:
        try:
            ball_num = int(ball['ball'])
            over = (ball_num - 1) // 6
            if over >= target_over:
                break
            
            striker = ball['striker']
            non_striker = ball['non_striker']
            
            # Accumulate runs
            runs_so_far += ball['runs_off_ball']
            runs_so_far += sum([ball.get(x, 0) or 0 for x in ['extras', 'wides', 'noballs', 'byes', 'legbyes']])
            
            # Count wickets
            if pd.notna(ball['wicket_type']) and ball['wicket_type']:
                wickets_so_far += 1
            
            balls_count += 1
        except (ValueError, KeyError, TypeError):
            continue
    
    return {
        'striker': striker,
        'non_striker': non_striker,
        'runs': runs_so_far,
        'wickets': wickets_so_far,
        'balls': balls_count
    }


def get_next_incoming_batter(match_balls, current_striker, current_non_striker):
    """Get the next incoming batter after current striker gets out"""
    
    # Convert to list to access by index
    balls_list = list(match_balls)
    
    for i, ball in enumerate(balls_list):
        try:
            dismissal = ball['wicket_type'] if pd.notna(ball['wicket_type']) else None
            player_dismissed = ball.get('player_dismissed', None)
            
            # Check that CURRENT STRIKER was actually dismissed
            if dismissal and player_dismissed == current_striker:
                # The next incoming batter is on the NEXT ball
                if i + 1 < len(balls_list):
                    next_ball = balls_list[i + 1]
                    next_striker = next_ball['striker']
                    # Verify it's actually a new batter
                    if next_striker and next_striker != current_striker and next_striker != current_non_striker:
                        return next_striker
                return None
        except (KeyError, TypeError):
            continue
    
    return None


def predict(match_data, ball_by_ball_data):
    """
    Predict the score at the end of 6 overs using data from first 3 overs.
    
    Parameters:
    - match_data: str, metadata about the match (for logging)
    - ball_by_ball_data: str, CSV format ball-by-ball data for first 3 overs
    
    Returns:
    - float: Predicted total runs at end of 6 overs
    """
    
    # Convert the CSV string into a pandas DataFrame
    df = pd.read_csv(StringIO(ball_by_ball_data))
    
    # Convert runs and extras to numeric, replace NaNs with 0
    df["runs_off_ball"] = pd.to_numeric(df["runs_off_ball"], errors="coerce").fillna(0)
    df["extras"] = pd.to_numeric(df["extras"], errors="coerce").fillna(0)
    
    # Convert to list of dictionaries for processing
    match_balls = [row.to_dict() for _, row in df.iterrows()]
    
    try:
        # Get state at end of first 3 overs
        state_3 = get_batting_state_at_over(match_balls, 3)
        
        striker = state_3['striker']
        non_striker = state_3['non_striker']
        runs_3 = state_3['runs']
        wickets_3 = state_3['wickets']
        balls_3 = state_3['balls']
        
        # Validate we have minimum required data
        if not striker or not non_striker or balls_3 < 18:
            print("ERROR: Insufficient data for first 3 overs")
            return None
        
        # Get batsman stats (fallback to average if not found)
        striker_stats = batsman_stats.get(striker, default_batsman_stats)
        non_striker_stats = batsman_stats.get(non_striker, default_batsman_stats)
        
        if not striker_stats or not non_striker_stats:
            print(f"ERROR: Invalid stats for {striker} or {non_striker}")
            return None
        
        # Get next incoming batter stats (fallback to average if not found)
        next_batter = get_next_incoming_batter(match_balls, striker, non_striker)
        next_batter_stats = None
        
        if next_batter:
            next_batter_stats = batsman_stats.get(next_batter, default_batsman_stats)
        
        # Calculate current run rate
        current_run_rate = runs_3 / balls_3 if balls_3 > 0 else 0
        
        # Get dismissal rates (percentage values)
        striker_dismissal = striker_stats['powerplay_dismissal_rate_per_100']
        non_striker_dismissal = non_striker_stats['powerplay_dismissal_rate_per_100']
        
        # Calculate weights based on dismissal rates
        striker_risk = min(striker_dismissal / 100, 1.0)
        non_striker_risk = min(non_striker_dismissal / 100, 1.0)
        next_weight = (striker_risk + non_striker_risk) / 2
        next_weight = min(next_weight, 1.0)
        
        # Build feature vector (12 features)
        features = [
            # Striker metrics (3)
            striker_stats['powerplay_avg_runs_per_ball'],
            striker_stats['powerplay_std_dev_runs_per_ball'],
            striker_dismissal,
            
            # Non-striker metrics (3)
            non_striker_stats['powerplay_avg_runs_per_ball'],
            non_striker_stats['powerplay_std_dev_runs_per_ball'],
            non_striker_dismissal,
            
            # Next batter metrics (2, weighted)
            next_batter_stats['powerplay_avg_runs_per_ball'] * next_weight if next_batter_stats else 0,
            next_batter_stats['powerplay_dismissal_rate_per_100'] * next_weight if next_batter_stats else 0,
            
            # Current match state (4)
            current_run_rate,
            wickets_3,
            runs_3,
            balls_3 / 6,  # Normalize balls to overs
        ]
        
        # Scale features using the trained scaler
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Make prediction using trained model
        predicted_6_over_runs = model.predict(features_scaled)[0]
        
        return predicted_6_over_runs
        
    except Exception as e:
        print(f"ERROR during prediction: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage
    print("IPL Powerplay Score Prediction Model")
    print("=" * 60)
    print("Model ready. Call predict(match_data, ball_by_ball_data) to get predictions.")
