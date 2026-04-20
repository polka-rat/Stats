import pandas as pd
import os

def process_recent_match_data():
    """
    Process all CSV files from recent_match_Data folder.
    Reads all matches and creates CSV strings for each match.
    
    Returns:
    - dict: Dictionary with match_id as key and CSV string as value
    """
    
    data_folder = "recent_match_Data"
    
    print("Loading recent match data...")
    all_matches = []
    
    # Read all CSV files from the folder
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        file_path = os.path.join(data_folder, csv_file)
        try:
            df = pd.read_csv(file_path)
            all_matches.append(df)
            print(f"  Loaded {csv_file}: {len(df)} rows")
        except Exception as e:
            print(f"  ERROR reading {csv_file}: {str(e)}")
    
    # Concatenate all match data
    if all_matches:
        combined_df = pd.concat(all_matches, ignore_index=True)
        print(f"\nTotal rows combined: {len(combined_df)}")
        print(f"Unique matches: {combined_df['match_id'].nunique()}")
    else:
        print("No match data found!")
        return {}
    
    # Group by match_id and create CSV strings
    match_csv_strings = {}
    
    for match_id, match_group in combined_df.groupby('match_id'):
        # Sort by ball to ensure correct order
        match_group = match_group.sort_values('ball').reset_index(drop=True)
        
        # Create CSV string for the entire match
        match_csv_string = match_group.to_csv(index=False)
        match_csv_strings[match_id] = match_csv_string
        
        # Extract metadata for display
        first_row = match_group.iloc[0]
        batting_team = str(first_row['batting_team'])
        bowling_team = str(first_row['bowling_team'])
        start_date = str(first_row['start_date'])
        venue = str(first_row['venue'])
        total_balls = len(match_group)
        max_over = int(float(match_group['ball'].max()))
        
        print(f"\nMatch {match_id}:")
        print(f"  {batting_team} vs {bowling_team}")
        print(f"  Date: {start_date}")
        print(f"  Venue: {venue}")
        print(f"  Total balls: {total_balls}")
        print(f"  Overs: 0-{max_over}")
    
    return match_csv_strings, combined_df


def get_first_n_overs_csv(match_df, n_overs):
    """
    Extract ball-by-ball data for first N overs and return as CSV string.
    
    Parameters:
    - match_df: DataFrame of ball-by-ball data for a single match
    - n_overs: Number of overs to extract (e.g., 3, 6)
    
    Returns:
    - str: CSV format of first N overs data
    """
    match_df = match_df.copy()
    match_df['ball_over'] = match_df['ball'].astype(float).apply(lambda x: int(x))
    df_n_overs = match_df[match_df['ball_over'] < n_overs].copy()
    
    return df_n_overs.to_csv(index=False)


def save_match_csv_strings(match_csv_strings, output_file='match_csv_strings.txt'):
    """
    Save match CSV strings to a text file for later use with backtester.
    
    Parameters:
    - match_csv_strings: Dictionary of match_id -> CSV string
    - output_file: Output text file path
    """
    print(f"\nSaving match CSV strings to {output_file}...")
    with open(output_file, 'w') as f:
        for match_id, csv_string in match_csv_strings.items():
            f.write(f"===== MATCH {match_id} =====\n")
            f.write(csv_string)
            f.write("\n")
    print(f"Saved {len(match_csv_strings)} matches to {output_file}")


def load_match_csv_strings(input_file='match_csv_strings.txt'):
    """
    Load match CSV strings from text file.
    
    Parameters:
    - input_file: Input text file path
    
    Returns:
    - dict: Dictionary of match_id -> CSV string
    """
    match_csv_strings = {}
    current_match_id = None
    current_csv = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('===== MATCH'):
                # Save previous match if exists
                if current_match_id is not None:
                    match_csv_strings[current_match_id] = ''.join(current_csv).strip() + '\n'
                
                # Extract match_id from header
                current_match_id = int(line.split()[-1].rstrip('=').strip())
                current_csv = []
            else:
                current_csv.append(line)
        
        # Save last match
        if current_match_id is not None:
            match_csv_strings[current_match_id] = ''.join(current_csv).strip() + '\n'
    
    return match_csv_strings


if __name__ == "__main__":
    print("="*80)
    print("RECENT MATCH DATA PROCESSOR")
    print("="*80 + "\n")
    
    # Process all recent match data
    match_csv_strings, combined_df = process_recent_match_data()
    
    # Save to text file for use with backtester
    save_match_csv_strings(match_csv_strings, 'match_csv_strings.txt')
    
    print("\n" + "="*80)
    print(f"PROCESSING COMPLETE")
    print(f"Total matches processed: {len(match_csv_strings)}")
    print(f"CSV strings saved to: match_csv_strings.txt")
    print("="*80)
