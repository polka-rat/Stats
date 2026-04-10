"""
Train a Linear Regression model to predict 6-over powerplay runs
using batsman statistics, dismissal rates, and current match state.

Features:
- Striker's powerplay metrics (run_rate, std_dev, dismissal_rate)
- Non-striker's powerplay metrics
- Next incoming batsman's metrics (weighted by current dismissal rates)
- Current run rate and wickets at end of 3 overs
"""

import pandas as pd
import numpy as np
import csv
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load batsman statistics
print("Loading batsman statistics...")
batsman_stats_df = pd.read_csv('batsman_statistics_2022_2025.csv')
batsman_stats = {}

for _, row in batsman_stats_df.iterrows():
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

def parse_ball_by_ball_data():
    """Parse recent_match_data.csv and extract powerplay innings data."""
    print("\nParsing match data...")
    
    matches = defaultdict(list)
    
    with open('recent_match_data.csv', 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                match_id = row['match_id']
                batting_team = row.get('batting_team', '').strip()
                over = int(row['over'])
                
                # Split innings by batting team within each match.
                # Keep only first 6 overs (0-5).
                if batting_team and over < 6:
                    innings_key = (match_id, batting_team)
                    matches[innings_key].append(row)
            except (ValueError, KeyError):
                continue
    
    print(f"Found {len(matches)} innings with powerplay data")
    return matches

def get_batting_state_at_over(match_balls, target_over):
    """Get batting state after overs [0, target_over)."""
    
    balls_in_match = []
    runs_so_far = 0
    wickets_so_far = 0
    
    for ball in match_balls:
        try:
            over = int(ball['over'])
            if over >= target_over:
                break
            
            balls_in_match.append(ball)
            runs_so_far += int(ball['runs_batter'])
            runs_so_far += int(ball.get('runs_extras', 0) or 0)
            
            if ball['dismissal_kind'] and ball['dismissal_kind'].strip():
                wickets_so_far += 1
        except (ValueError, KeyError):
            continue
    
    # Find current striker and non-striker at end of target_over
    striker = None
    non_striker = None
    
    if balls_in_match:
        # We need to track who's batting
        # For simplicity, track the last batter in the over
        last_ball = balls_in_match[-1]
        
        # Try to determine striker/non-striker from bowling pattern
        striker = last_ball.get('batter', '').strip()
        
        # Try to find previous batter to determine non-striker
        if len(balls_in_match) > 1:
            non_striker = balls_in_match[-2].get('batter', '').strip()
            # Check if they're different
            if non_striker == striker and len(balls_in_match) > 5:
                non_striker = balls_in_match[-6].get('batter', '').strip()
    
    return {
        'striker': striker,
        'non_striker': non_striker,
        'runs': runs_so_far,
        'wickets': wickets_so_far,
        'balls': len(balls_in_match)
    }

def get_target_runs_at_over(match_balls, start_over, end_over):
    """Get total runs between overs [start_over, end_over)."""
    
    runs = 0
    for ball in match_balls:
        try:
            over = int(ball['over'])
            if start_over <= over < end_over:
                runs += int(ball['runs_batter'])
                runs += int(ball.get('runs_extras', 0) or 0)
        except (ValueError, KeyError):
            continue
    
    return runs

def get_next_incoming_batter(match_balls, current_striker, current_non_striker):
    """Get the next incoming batter after current striker gets out"""
    
    # Find the batter that comes after the striker gets out
    found_dismissal = False
    
    for ball in match_balls:
        try:
            batter = ball.get('batter', '').strip()
            dismissal = ball.get('dismissal_kind', '').strip() if ball.get('dismissal_kind') else ''
            
            if batter == current_striker and dismissal:
                found_dismissal = True
                continue
            
            if found_dismissal and batter != current_striker and batter != current_non_striker:
                return batter
        except (KeyError, AttributeError):
            continue
    
    return None

def create_training_data():
    """Create training dataset from match data"""
    
    print("\nCreating training dataset...")
    
    matches = parse_ball_by_ball_data()
    
    X = []
    y = []
    valid_samples = 0
    
    for innings_key, match_balls in matches.items():
        try:
            # Get state at end of first 3 overs (overs 0,1,2)
            state_3 = get_batting_state_at_over(match_balls, 3)
            
            # Get runs in overs 3,4,5 (next 3 overs)
            runs_3_to_6 = get_target_runs_at_over(match_balls, 3, 6)
            
            striker = state_3['striker']
            non_striker = state_3['non_striker']
            runs_3 = state_3['runs']
            wickets_3 = state_3['wickets']
            balls_3 = state_3['balls']
            
            # Skip if missing critical data
            if not striker or not non_striker or balls_3 < 18:  # At least 18 balls = 3 overs
                continue
            
            # Get batsman stats - use powerplay metrics
            striker_stats = batsman_stats.get(striker)
            non_striker_stats = batsman_stats.get(non_striker)
            
            if not striker_stats or not non_striker_stats:
                continue
            
            # Get next incoming batter
            next_batter = get_next_incoming_batter(match_balls, striker, non_striker)
            next_batter_stats = None
            
            if next_batter:
                next_batter_stats = batsman_stats.get(next_batter)
            
            # Calculate current run rate
            current_run_rate = runs_3 / balls_3 if balls_3 > 0 else 0
            
            # Get dismissal rates (use percentage values)
            striker_dismissal = striker_stats['powerplay_dismissal_rate_per_100']
            non_striker_dismissal = non_striker_stats['powerplay_dismissal_rate_per_100']
            next_dismissal = next_batter_stats['powerplay_dismissal_rate_per_100'] if next_batter_stats else 0
            
            # Calculate weights based on dismissal rates
            # Higher dismissal rate = more weight to next batter
            striker_risk = min(striker_dismissal / 100, 1.0)  # Normalize to 0-1
            non_striker_risk = min(non_striker_dismissal / 100, 1.0)
            next_weight = (striker_risk + non_striker_risk) / 2  # Average risk
            
            # Scale next_weight to be 0-1
            next_weight = min(next_weight, 1.0)
            current_weight = 1.0 - next_weight  # Current batsmen weight
            
            # Build feature vector
            features = [
                # Striker metrics (removed run_rate and strike_rate - collinear with avg_runs_per_ball)
                striker_stats['powerplay_avg_runs_per_ball'],
                striker_stats['powerplay_std_dev_runs_per_ball'],
                striker_dismissal,
                striker_stats['powerplay_consistency_score'],
                
                # Non-striker metrics (removed run_rate and strike_rate - collinear with avg_runs_per_ball)
                non_striker_stats['powerplay_avg_runs_per_ball'],
                non_striker_stats['powerplay_std_dev_runs_per_ball'],
                non_striker_dismissal,
                non_striker_stats['powerplay_consistency_score'],
                
                # Next batter metrics (weighted) (removed run_rate - collinear with avg_runs_per_ball)
                next_batter_stats['powerplay_avg_runs_per_ball'] * next_weight if next_batter_stats else 0,
                next_batter_stats['powerplay_dismissal_rate_per_100'] * next_weight if next_batter_stats else 0,
                next_batter_stats['powerplay_consistency_score'] * next_weight if next_batter_stats else 0,
                
                # Current match state
                current_run_rate,
                wickets_3,
                runs_3,
                balls_3 / 6,  # Normalize balls to overs
                current_weight  # Weight of current batsmen
            ]
            
            X.append(features)
            y.append(runs_3 + runs_3_to_6)  # Total runs at end of 6 overs
            valid_samples += 1
            
        except Exception as e:
            continue
    
    print(f"Created {valid_samples} valid training samples")
    
    return np.array(X), np.array(y)

def evaluate_simple_baseline_model(y_actual):
    """
    Evaluate a simple baseline model that predicts:
    6-over_runs = 2 × runs_at_3_overs
    
    This model assumes the run rate in overs 4-6 is same as overs 1-3.
    """
    
    print("\n" + "="*60)
    print("BASELINE MODEL EVALUATION (Simple 2x Extrapolation)")
    print("="*60)
    
    # Extract runs_at_3_overs from the dataset (feature index 17)
    # We need to reconstruct this from the full training data
    # For baseline, we'll use y_actual / 2 as an estimate
    runs_at_3_approx = y_actual / 2
    
    # Baseline prediction: 2 × runs_at_3_overs
    y_pred_baseline = runs_at_3_approx * 2
    
    # Calculate metrics
    mse_baseline = mean_squared_error(y_actual, y_pred_baseline)
    rmse_baseline = np.sqrt(mse_baseline)
    mae_baseline = mean_absolute_error(y_actual, y_pred_baseline)
    r2_baseline = r2_score(y_actual, y_pred_baseline)
    
    print(f"Baseline RMSE: {rmse_baseline:.4f} runs")
    print(f"Baseline MAE:  {mae_baseline:.4f} runs")
    print(f"Baseline R²:   {r2_baseline:.4f}")
    print(f"Baseline MSE:  {mse_baseline:.4f}")
    
    return {
        'rmse': rmse_baseline,
        'mae': mae_baseline,
        'r2': r2_baseline,
        'mse': mse_baseline
    }


def evaluate_simple_baseline_model_with_features(X, y):
    """
    Evaluate baseline model with actual feature extraction.
    Uses runs_at_3_overs (feature index 13) and multiplies by 2.
    
    Feature indices (16 features total):
    0-3: Striker
    4-7: Non-striker
    8-10: Next batter (weighted)
    11: current_run_rate
    12: wickets_down
    13: runs_at_3_overs
    14: overs_completed
    15: current_batsmen_weight
    """
    
    print("\n" + "="*60)
    print("BASELINE MODEL EVALUATION (2x Runs at 3 Overs)")
    print("="*60)
    
    # Extract runs_at_3_overs from features (index 13)
    runs_at_3_overs = X[:, 13]
    
    # Baseline prediction: 2 × runs_at_3_overs
    y_pred_baseline = runs_at_3_overs * 2
    
    # Calculate metrics
    mse_baseline = mean_squared_error(y, y_pred_baseline)
    rmse_baseline = np.sqrt(mse_baseline)
    mae_baseline = mean_absolute_error(y, y_pred_baseline)
    r2_baseline = r2_score(y, y_pred_baseline)
    
    print(f"Baseline RMSE: {rmse_baseline:.4f} runs")
    print(f"Baseline MAE:  {mae_baseline:.4f} runs")
    print(f"Baseline R²:   {r2_baseline:.4f}")
    print(f"Baseline MSE:  {mse_baseline:.4f}")
    
    return {
        'rmse': rmse_baseline,
        'mae': mae_baseline,
        'r2': r2_baseline,
        'mse': mse_baseline,
        'predictions': y_pred_baseline
    }


def train_and_evaluate_model():
    """Train linear regression model and evaluate"""
    
    # Create training data
    X, y = create_training_data()
    
    if len(X) < 10:
        print("ERROR: Not enough training samples!")
        return None
    
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Target (6-over runs) - Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Normalize features (CRITICAL for interpreting coefficients)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nFeature scaling applied (StandardScaler)")
    
    # Train model
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"Training RMSE: {rmse_train:.4f} runs")
    print(f"Test RMSE:     {rmse_test:.4f} runs")
    print(f"Training MAE:  {mae_train:.4f} runs")
    print(f"Test MAE:      {mae_test:.4f} runs")
    print(f"Training R²:   {r2_train:.4f}")
    print(f"Test R²:       {r2_test:.4f}")
    
    # Model coefficients
    print("\n" + "="*60)
    print("MODEL COEFFICIENTS")
    print("="*60)
    feature_names = [
        # Striker (removed run_rate and strike_rate)
        'striker_avg_per_ball', 'striker_std_dev',
        'striker_dismissal_rate', 'striker_consistency',
        # Non-striker (removed run_rate and strike_rate)
        'non_striker_avg_per_ball', 'non_striker_std_dev',
        'non_striker_dismissal_rate', 'non_striker_consistency',
        # Next batter (weighted, removed run_rate)
        'next_avg_per_ball_weighted', 'next_dismissal_weighted',
        'next_consistency_weighted',
        # Match state
        'current_run_rate', 'wickets_down', 'runs_at_3_overs', 'overs_completed',
        'current_batsmen_weight'
    ]
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(coef_df.head(10).to_string(index=False))
    
    print(f"\nIntercept: {model.intercept_:.4f}")
    
    # Evaluate baseline model and compare
    baseline_metrics = evaluate_simple_baseline_model_with_features(X_test, y_test)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON: Linear Regression vs Baseline (2x)")
    print("="*60)
    
    improvement_rmse = ((baseline_metrics['rmse'] - rmse_test) / baseline_metrics['rmse']) * 100
    improvement_mae = ((baseline_metrics['mae'] - mae_test) / baseline_metrics['mae']) * 100
    improvement_r2 = ((r2_test - baseline_metrics['r2']) / abs(baseline_metrics['r2'])) * 100 if baseline_metrics['r2'] != 0 else 0
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Linear Reg':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'RMSE':<20} {baseline_metrics['rmse']:<15.4f} {rmse_test:<15.4f} {improvement_rmse:>13.2f}%")
    print(f"{'MAE':<20} {baseline_metrics['mae']:<15.4f} {mae_test:<15.4f} {improvement_mae:>13.2f}%")
    print(f"{'R²':<20} {baseline_metrics['r2']:<15.4f} {r2_test:<15.4f} {improvement_r2:>13.2f}%")
    
    if improvement_rmse > 0:
        print(f"\n✓ Linear Regression is BETTER than baseline by {improvement_rmse:.2f}% on RMSE")
    else:
        print(f"\n✗ Baseline is better than Linear Regression by {abs(improvement_rmse):.2f}% on RMSE")
    
    # Validate test predictions
    print("\n" + "="*60)
    print("SAMPLE TEST PREDICTIONS (First 10 test samples)")
    print("="*60)
    print(f"\n{'Actual 6-Over':<15} {'Predicted':<15} {'Error':<15} {'Error %':<10}")
    print("-" * 55)
    
    sample_size = min(10, len(y_test))
    for i in range(sample_size):
        actual = y_test[i]
        predicted = y_pred_test[i]
        error = actual - predicted
        error_pct = (error / actual * 100) if actual != 0 else 0
        print(f"{actual:<15.2f} {predicted:<15.2f} {error:<15.2f} {error_pct:>8.2f}%")
    
    print(f"\nTest set runs range: {y_test.min():.1f} - {y_test.max():.1f} runs")
    print(f"Predictions range:   {y_pred_test.min():.1f} - {y_pred_test.max():.1f} runs")
    print(f"Mean actual:         {y_test.mean():.2f} runs")
    print(f"Mean predicted:      {y_pred_test.mean():.2f} runs")
    
    # Save model
    with open('powerplay_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved to powerplay_regression_model.pkl")
        # Save scaler (needed for predictions on new data)
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Feature scaler saved to feature_scaler.pkl")
        # Save feature names
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    return model, X_test, y_test, rmse_test


if __name__ == "__main__":
    print("="*60)
    print("POWERPLAY RUNS PREDICTION - LINEAR REGRESSION MODEL")
    print("="*60)
    
    model, X_test, y_test, rmse = train_and_evaluate_model()
    
    if model:
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\nThe model can now be used with predict_powerplay_runs() function")
