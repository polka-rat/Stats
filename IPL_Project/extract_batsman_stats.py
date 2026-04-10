"""
Extract batsman statistics (run rate and dismissal rate) from years 2022-2025
and create a comprehensive dataset with mean and standard deviation metrics.
"""

import pandas as pd
import csv
import os
from collections import defaultdict
import statistics

def parse_recent_match_data():
    """Parse recent_match_data.csv and extract batsman statistics for 2022-2025"""
    
    # Try multiple possible paths
    possible_paths = [
        r"c:\Users\gupta\Downloads\ipl_csv2\recent_match_data.csv",
        r"recent_match_data.csv",
        "./recent_match_data.csv"
    ]
    
    filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            filepath = path
            break
    
    if filepath is None:
        print("Error: Could not find recent_match_data.csv in any of the expected locations:")
        for path in possible_paths:
            print(f"  - {path}")
        raise FileNotFoundError("recent_match_data.csv not found")
    
    # Read the recent match data
    print(f"Reading recent match data from: {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df['year'] = df['date'].dt.year
    
    # Filter for years 2022-2025
    df_filtered = df[(df['year'] >= 2022) & (df['year'] <= 2025)]
    
    print(f"Total records: {len(df)}")
    print(f"Records from 2022-2025: {len(df_filtered)}")
    print(f"Year range in filtered data: {df_filtered['year'].min()} to {df_filtered['year'].max()}")
    
    return df_filtered

def calculate_batsman_statistics(df):
    """Calculate run rate and dismissal rate for each batsman using vectorized operations"""
    
    print("\nCalculating batsman statistics (vectorized)...")
    
    # Clean the data
    df_clean = df[df['batter'].notna() & (df['batter'] != '')].copy()
    df_clean['batter'] = df_clean['batter'].str.strip()
    df_clean['runs_batter'] = df_clean['runs_batter'].fillna(0)
    df_clean['dismissal_kind'] = df_clean['dismissal_kind'].fillna('')
    
    # Parse overs information
    # Overs are in format 0, 1, 2, ... representing the over number
    # Powerplay is first 6 overs (0-5 inclusive, which is overs 1-6)
    df_clean['over_number'] = df_clean['over'].astype(int)
    
    # Powerplay is first 6 overs (0-5 in zero-indexed, representing overs 1-6)
    df_clean['is_powerplay'] = df_clean['over_number'] <= 5
    
    # Group by batter
    batsman_groups = df_clean.groupby('batter')
    
    batsman_stats = {}
    
    for batter, group in batsman_groups:
        # All balls
        all_runs = group['runs_batter'].tolist()
        all_balls = len(group)
        all_dismissals = (group['dismissal_kind'] != '').sum()
        
        # Powerplay balls (first 6 overs)
        powerplay_group = group[group['is_powerplay']]
        powerplay_runs = powerplay_group['runs_batter'].tolist()
        powerplay_balls = len(powerplay_group)
        powerplay_dismissals = (powerplay_group['dismissal_kind'] != '').sum()
        
        batsman_stats[batter] = {
            # All overs stats
            'runs': all_runs,
            'balls': all_balls,
            'dismissals': all_dismissals,
            # Powerplay stats
            'powerplay_runs': powerplay_runs,
            'powerplay_balls': powerplay_balls,
            'powerplay_dismissals': powerplay_dismissals,
            # Match info
            'matches': set(group['match_id'].unique()),
            'teams': set(group['batting_team'].unique())
        }
    
    print(f"Total unique batsmen: {len(batsman_stats)}")
    
    return batsman_stats

def create_batsman_dataset(batsman_stats):
    """Create a dataset with mean and standard deviation for each batsman, including powerplay stats"""
    
    dataset = []
    
    print("\nCreating batsman dataset with statistical metrics...")
    
    for batsman_name, stats in batsman_stats.items():
        if stats['balls'] == 0:
            continue
        
        # ===== ALL OVERS STATISTICS =====
        # Calculate run rate metrics
        total_runs = sum(stats['runs'])
        run_rate = total_runs / stats['balls']
        
        # Calculate runs per run scored (only non-zero runs)
        non_zero_runs = [r for r in stats['runs'] if r > 0]
        
        # Calculate mean and std dev for runs
        if len(stats['runs']) > 1:
            avg_runs_per_ball = statistics.mean(stats['runs'])
            std_dev_runs = statistics.stdev(stats['runs'])
        else:
            avg_runs_per_ball = run_rate
            std_dev_runs = 0
        
        # Calculate dismissal rate (dismissals per 100 balls)
        dismissal_rate = (stats['dismissals'] / stats['balls']) * 100 if stats['balls'] > 0 else 0
        
        # Strike rate (runs per 100 balls faced)
        strike_rate = (total_runs / stats['balls']) * 100 if stats['balls'] > 0 else 0
        
        # Boundary percentage (sixes and fours)
        boundary_runs = 0
        for r in stats['runs']:
            if r == 4 or r == 6:
                boundary_runs += 1
        boundary_percentage = (boundary_runs / stats['balls']) * 100 if stats['balls'] > 0 else 0
        
        # Dot ball percentage
        dot_balls = sum(1 for r in stats['runs'] if r == 0)
        dot_percentage = (dot_balls / stats['balls']) * 100 if stats['balls'] > 0 else 0
        
        # Average dismissal (avg runs when getting out)
        if stats['dismissals'] > 0 and len(non_zero_runs) > 0:
            avg_on_dismissal = statistics.mean(non_zero_runs)
        else:
            avg_on_dismissal = 0
        
        # Consistency score for all overs
        consistency_all = round((1 - (std_dev_runs / (avg_runs_per_ball + 1))) * 100, 2)
        
        # ===== POWERPLAY STATISTICS =====
        # Powerplay (first 6 overs)
        powerplay_total_runs = sum(stats['powerplay_runs'])
        powerplay_balls = stats['powerplay_balls']
        
        if powerplay_balls > 0:
            powerplay_run_rate = powerplay_total_runs / powerplay_balls
            
            # Powerplay mean and std dev
            if len(stats['powerplay_runs']) > 1:
                powerplay_avg_per_ball = statistics.mean(stats['powerplay_runs'])
                powerplay_std_dev = statistics.stdev(stats['powerplay_runs'])
            else:
                powerplay_avg_per_ball = powerplay_run_rate
                powerplay_std_dev = 0
            
            # Powerplay strike rate
            powerplay_strike_rate = (powerplay_total_runs / powerplay_balls) * 100
            
            # Powerplay boundary percentage
            powerplay_boundary_runs = sum(1 for r in stats['powerplay_runs'] if r == 4 or r == 6)
            powerplay_boundary_percentage = (powerplay_boundary_runs / powerplay_balls) * 100
            
            # Powerplay dot ball percentage
            powerplay_dot_balls = sum(1 for r in stats['powerplay_runs'] if r == 0)
            powerplay_dot_percentage = (powerplay_dot_balls / powerplay_balls) * 100
            
            # Powerplay dismissal rate
            powerplay_dismissal_rate = (stats['powerplay_dismissals'] / powerplay_balls) * 100
            
            # Powerplay consistency score
            powerplay_consistency = round((1 - (powerplay_std_dev / (powerplay_avg_per_ball + 1))) * 100, 2)
        else:
            powerplay_run_rate = 0
            powerplay_avg_per_ball = 0
            powerplay_std_dev = 0
            powerplay_strike_rate = 0
            powerplay_boundary_percentage = 0
            powerplay_dot_percentage = 0
            powerplay_dismissal_rate = 0
            powerplay_consistency = 0
        
        dataset.append({
            # Basic info
            'batter': batsman_name,
            'total_balls_faced': stats['balls'],
            'total_matches': len(stats['matches']),
            'total_runs': total_runs,
            'total_dismissals': stats['dismissals'],
            
            # All overs metrics
            'run_rate': round(run_rate, 3),
            'strike_rate': round(strike_rate, 2),
            'avg_runs_per_ball': round(avg_runs_per_ball, 3),
            'std_dev_runs_per_ball': round(std_dev_runs, 3),
            'dismissal_rate_per_100': round(dismissal_rate, 2),
            'boundary_percentage': round(boundary_percentage, 2),
            'dot_ball_percentage': round(dot_percentage, 2),
            'avg_runs_on_dismissal': round(avg_on_dismissal, 2),
            'consistency_score': consistency_all,
            
            # Powerplay metrics
            'powerplay_balls_faced': powerplay_balls,
            'powerplay_total_runs': powerplay_total_runs,
            'powerplay_dismissals': stats['powerplay_dismissals'],
            'powerplay_run_rate': round(powerplay_run_rate, 3),
            'powerplay_strike_rate': round(powerplay_strike_rate, 2),
            'powerplay_avg_runs_per_ball': round(powerplay_avg_per_ball, 3),
            'powerplay_std_dev_runs_per_ball': round(powerplay_std_dev, 3),
            'powerplay_dismissal_rate_per_100': round(powerplay_dismissal_rate, 2),
            'powerplay_boundary_percentage': round(powerplay_boundary_percentage, 2),
            'powerplay_dot_ball_percentage': round(powerplay_dot_percentage, 2),
            'powerplay_consistency_score': powerplay_consistency,
        })
    
    # Sort by total balls faced (descending)
    dataset.sort(key=lambda x: x['total_balls_faced'], reverse=True)
    
    print(f"Created dataset with {len(dataset)} batsmen (including powerplay stats)")
    
    return dataset

def save_batsman_dataset(dataset, filename='batsman_statistics_2022_2025.csv'):
    """Save the batsman dataset to CSV"""
    
    filepath = os.path.join(r"c:\Users\gupta\Downloads\ipl_csv2", filename)
    
    print(f"\nSaving batsman statistics to {filename}...")
    
    df = pd.DataFrame(dataset)
    df.to_csv(filepath, index=False)
    
    print(f"Successfully saved {len(dataset)} batsmen statistics to {filepath}")
    
    # Print summary
    print("\n=== BATSMAN STATISTICS SUMMARY ===")
    print(f"Total batsmen analyzed: {len(dataset)}")
    print(f"\nTop 10 batsmen by overall run rate:")
    top_10_overall = df.nlargest(10, 'run_rate')[['batter', 'total_balls_faced', 'run_rate', 'powerplay_run_rate', 'consistency_score']]
    print(top_10_overall.to_string(index=False))
    
    print(f"\nTop 10 batsmen by powerplay run rate:")
    top_10_pp = df.nlargest(10, 'powerplay_run_rate')[['batter', 'powerplay_balls_faced', 'powerplay_run_rate', 'powerplay_strike_rate', 'powerplay_consistency_score']]
    print(top_10_pp.to_string(index=False))
    
    print(f"\nTop 10 batsmen by balls faced (experience):")
    experienced = df.nlargest(10, 'total_balls_faced')[['batter', 'total_balls_faced', 'total_matches', 'run_rate', 'powerplay_run_rate']]
    print(experienced.to_string(index=False))
    
    return filepath


def main():
    """Main execution"""
    
    print("="*60)
    print("BATSMAN STATISTICS EXTRACTOR FOR 2022-2025")
    print("="*60)
    
    # Parse recent match data
    df = parse_recent_match_data()
    
    # Calculate statistics
    batsman_stats = calculate_batsman_statistics(df)
    
    # Create dataset
    dataset = create_batsman_dataset(batsman_stats)
    
    # Save to CSV
    filepath = save_batsman_dataset(dataset)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    
    return filepath


if __name__ == "__main__":
    filepath = main()
