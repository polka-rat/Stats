import pandas as pd
import numpy as np
from io import StringIO
from main import predict
import warnings
warnings.filterwarnings('ignore')

def get_actual_6_over_score(match_balls):
    """
    Calculate actual runs scored in first 6 overs.
    
    Parameters:
    - match_balls: list of ball dictionaries
    
    Returns:
    - int: Total runs in first 6 overs
    """
    runs = 0
    
    for ball in match_balls:
        try:
            ball_num = int(ball['ball'])
            over = (ball_num - 1) // 6
            
            # Only count balls in first 6 overs
            if over >= 6:
                break
            
            # Count runs from ball
            runs += int(ball['runs_off_bat']) if pd.notna(ball['runs_off_bat']) else 0
            
            # Count extras
            runs += int(ball['extras']) if pd.notna(ball['extras']) else 0
            
        except (ValueError, KeyError, TypeError):
            continue
    
    return runs


def get_first_3_overs_csv(match_balls):
    """
    Extract ball-by-ball data for first 3 overs and return as CSV string.
    
    Parameters:
    - match_balls: DataFrame of ball-by-ball data
    
    Returns:
    - str: CSV format of first 3 overs data
    """
    # Convert to DataFrame if it's a list
    if isinstance(match_balls, list):
        df = pd.DataFrame(match_balls)
    else:
        df = match_balls.copy()
    
    # Filter for first 18 balls (3 overs * 6 balls)
    df['ball_num'] = pd.to_numeric(df['ball'], errors='coerce')

    df_3_overs = df[df['ball_num'] <= 18].copy()
    
    # Convert back to CSV string
    print("Meoeoeoeoeoeooew")
    print(df_3_overs.to_csv(index=False))
    return df_3_overs.to_csv(index=False)


def backtest_on_dataset(csv_file_path, max_matches=None):
    """
    Run backtester on entire dataset.
    
    Parameters:
    - csv_file_path: path to the CSV file with ball-by-ball data
    - max_matches: maximum number of matches to test (None for all)
    
    Returns:
    - dict: Statistics including RMSE, MAE, correlation
    """
    
    print("Loading dataset...")
    df = pd.read_csv(csv_file_path)
    
    # Get unique matches
    unique_matches = df['match_id'].unique()
    if max_matches:
        unique_matches = unique_matches[:max_matches]
    
    print(f"Total matches in dataset: {len(unique_matches)}")
    if max_matches:
        print(f"Testing on: {max_matches} matches")
    
    predictions = []
    actuals = []
    errors = []
    failed_predictions = 0
    successful_predictions = 0
    
    print("\nRunning predictions...")
    print("-" * 80)
    
    for idx, match_id in enumerate(unique_matches):
        try:
            # Get all balls for this match, first innings only
            match_data = df[(df['match_id'] == match_id) & (df['innings'] == 1)]
            
            if len(match_data) < 18:
                # Not enough data for first 3 overs
                failed_predictions += 1
                continue
            
            # Sort by ball number to ensure correct order
            match_data = match_data.sort_values('ball').reset_index(drop=True)
            
            # Convert to list of dictionaries for easier processing
            match_balls = match_data.to_dict('records')
            
            # Get first 3 overs as CSV string
            first_3_overs_csv = get_first_3_overs_csv(match_data)
            
            # Get actual 6 over score
            actual_score = get_actual_6_over_score(match_balls)
            
            # Make prediction
            predicted_score = predict(f"Match {match_id}", first_3_overs_csv)
            
            if predicted_score is None:
                failed_predictions += 1
                continue
            
            # Store results
            predictions.append(predicted_score)
            actuals.append(actual_score)
            error = abs(predicted_score - actual_score)
            errors.append(error)
            successful_predictions += 1
            
            # Print progress every 10 matches
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(unique_matches)} matches - RMSE so far: {calculate_rmse(predictions, actuals):.4f}")
        
        except Exception as e:
            failed_predictions += 1
            continue
    
    print("-" * 80)
    
    if successful_predictions == 0:
        print("ERROR: No successful predictions made!")
        return None
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = np.array(errors)
    
    rmse = calculate_rmse(predictions, actuals)
    mae = np.mean(errors)
    std_error = np.std(errors)
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING RESULTS")
    print(f"{'='*80}")
    print(f"Total matches tested:       {len(unique_matches)}")
    print(f"Successful predictions:     {successful_predictions}")
    print(f"Failed predictions:         {failed_predictions}")
    print(f"\nMetrics:")
    print(f"  RMSE:                     {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error):{mae:.4f}")
    print(f"  Std Dev of Error:         {std_error:.4f}")
    print(f"  Correlation:              {correlation:.4f}")
    print(f"\nActual Score Statistics:")
    print(f"  Min:                      {np.min(actuals):.0f}")
    print(f"  Max:                      {np.max(actuals):.0f}")
    print(f"  Mean:                     {np.mean(actuals):.4f}")
    print(f"  Std Dev:                  {np.std(actuals):.4f}")
    print(f"\nPredicted Score Statistics:")
    print(f"  Min:                      {np.min(predictions):.0f}")
    print(f"  Max:                      {np.max(predictions):.0f}")
    print(f"  Mean:                     {np.mean(predictions):.4f}")
    print(f"  Std Dev:                  {np.std(predictions):.4f}")
    print(f"{'='*80}\n")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'std_error': std_error,
        'correlation': correlation,
        'successful_predictions': successful_predictions,
        'failed_predictions': failed_predictions,
        'predictions': predictions,
        'actuals': actuals,
        'errors': errors
    }


def calculate_rmse(predictions, actuals):
    """Calculate RMSE between predictions and actuals."""
    return np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))


if __name__ == "__main__":
    # Run backtester on the combined cleaned dataset
    # You can adjust max_matches parameter to test on fewer matches for faster iteration
    results = backtest_on_dataset('ipl_combined_cleaned.csv', max_matches=100)
    
    if results:
        print("Backtesting completed successfully!")
    else:
        print("Backtesting failed.")
