import pandas as pd
import numpy as np
from io import StringIO
from main import predict
import warnings
warnings.filterwarnings('ignore')

def get_actual_6_over_score(match_balls):
    """
    Calculate actual runs scored in first 6 overs.
    Counts all deliveries in overs 0-5 (including extra deliveries from wides/noballs).
    
    Parameters:
    - match_balls: list of ball dictionaries
    
    Returns:
    - int: Total runs in first 6 overs
    """
    runs = 0
    
    for ball in match_balls:
        try:
            # Ball format: over.ball_number (e.g., 0.1, 1.5, 2.7, 5.7, etc.)
            ball_str = str(ball['ball'])
            parts = ball_str.split('.')
            over = int(parts[0])
            
            # Only count first 6 overs (0-5)
            # This includes extra deliveries (7+) from wides/noballs in these overs
            if over >= 6:
                break
            
            # Count runs from bat
            ball_runs = int(ball['runs_off_bat']) if pd.notna(ball['runs_off_bat']) else 0
            runs += ball_runs
            
            # Count extras only once (extras column already includes wides+noballs+byes+legbyes)
            extras = int(ball['extras']) if pd.notna(ball['extras']) else 0
            runs += extras
            
        except (ValueError, KeyError, TypeError):
            continue
    
    return runs


def get_actual_6_over_score_debug(match_balls, match_id):
    """
    Calculate actual runs scored in first 6 overs with detailed breakdown.
    Includes all deliveries in overs 0-5 (including extras from wides/noballs).
    """
    runs = 0
    over_runs = {}
    
    for ball in match_balls:
        try:
            ball_str = str(ball['ball'])
            parts = ball_str.split('.')
            over = int(parts[0])
            
            if over >= 6:
                break
            
            if over not in over_runs:
                over_runs[over] = 0
            
            # Count runs from ball
            ball_runs = int(ball['runs_off_bat']) if pd.notna(ball['runs_off_bat']) else 0
            runs += ball_runs
            over_runs[over] += ball_runs
            
            # Count extras only (not wides/noballs/etc. separately)
            extras = int(ball['extras']) if pd.notna(ball['extras']) else 0
            runs += extras
            over_runs[over] += extras
            
        except (ValueError, KeyError, TypeError):
            continue
    
    # Print breakdown
    print(f"\n=== Match {match_id} Breakdown ===")
    for over in sorted(over_runs.keys()):
        print(f"  Over {over}: {over_runs[over]} runs")
    print(f"  Total: {runs} runs")
    
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
    
    # Filter for first 18 balls (3 overs: 0, 1, 2)
    df['ball_over'] = df['ball'].astype(float).apply(lambda x: int(x))
    df_3_overs = df[df['ball_over'] < 3].copy()
    
    # Convert back to CSV string
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


def backtest_on_txt_file(txt_file_path):
    """
    Process CSV strings from text file and perform predictions.
    
    Parameters:
    - txt_file_path: path to the text file with match CSV strings
    
    Returns:
    - dict: Statistics including RMSE, MAE, correlation
    """
    
    print("Loading match CSV strings from text file...")
    
    # Parse the text file to extract match data
    matches = {}
    current_match_id = None
    current_csv_lines = []
    
    try:
        with open(txt_file_path, 'r') as f:
            for line in f:
                if line.startswith('===== MATCH'):
                    # Save previous match if exists
                    if current_match_id is not None:
                        matches[current_match_id] = ''.join(current_csv_lines).strip()
                    
                    # Extract match_id from header like "===== MATCH 1527688 ====="
                    parts = line.strip().split()
                    # Format: ===== MATCH match_id =====
                    # So parts[2] should be the match_id
                    current_match_id = int(parts[2])
                    current_csv_lines = []
                else:
                    current_csv_lines.append(line)
            
            # Save last match
            if current_match_id is not None:
                matches[current_match_id] = ''.join(current_csv_lines).strip()
    
    except Exception as e:
        print(f"ERROR reading file: {str(e)}")
        return None
    
    print(f"Loaded {len(matches)} matches from text file")
    
    predictions = []
    actuals = []
    errors = []
    failed_predictions = 0
    successful_predictions = 0
    
    print("\nRunning predictions...")
    print("-" * 80)
    
    for idx, (match_id, csv_string) in enumerate(matches.items()):
        try:
            # Parse CSV string
            df = pd.read_csv(StringIO(csv_string))
            
            if len(df) < 18:
                # Not enough data for first 3 overs
                failed_predictions += 1
                continue
            
            # Convert to list of dictionaries
            match_balls = df.to_dict('records')
            
            # Get actual 6 over score
            actual_score = get_actual_6_over_score(match_balls)
            
            # Get first 3 overs CSV
            first_3_overs_csv = get_first_3_overs_csv(df)
            
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
            
            print(f"Match {match_id}: Actual={actual_score:.1f}, Predicted={predicted_score:.2f}, Error={error:.2f}")
        
        except Exception as e:
            failed_predictions += 1
            print(f"Match {match_id}: ERROR - {str(e)}")
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
    print(f"BACKTESTING RESULTS (TXT FILE)")
    print(f"{'='*80}")
    print(f"Total matches tested:       {len(matches)}")
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


if __name__ == "__main__":
    # Run backtester on the combined cleaned dataset
    # You can adjust max_matches parameter to test on fewer matches for faster iteration
    
    # results = backtest_on_dataset('ipl_combined_cleaned.csv', max_matches=100)
  
    print("\n" + "="*80)
    print("BACKTESTER - RECENT MATCH DATA")
    print("="*80 + "\n")
    
    results_txt = backtest_on_txt_file('match_csv_strings.txt')
    # if results:
    #     print("Backtesting completed successfully!")
    if results_txt:
        print("Recent match predictions completed successfully!")
    else:
        print("Backtesting failed.")
