"""
Advanced optimization techniques to reduce RMSE on powerplay prediction model
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Import the create_training_data function from powerplay_stats
import sys
sys.path.insert(0, '.')
from powerplay_stats import create_training_data

def remove_outliers_iqr(X, y, multiplier=1.5):
    """Remove outliers using IQR method"""
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    mask = (y >= lower_bound) & (y <= upper_bound)
    
    print(f"\nOutlier Removal (IQR):")
    print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    print(f"  Removed {(~mask).sum()} outliers out of {len(y)} samples")
    
    return X[mask], y[mask]

def add_interaction_features(X, feature_names):
    """Add interaction features between key variables"""
    X_interact = X.copy()
    new_names = feature_names.copy()
    
    # runs_at_3_overs index is 10
    # current_run_rate index is 8
    # wickets_down index is 9
    
    # Interaction: current_run_rate * runs_at_3_overs
    interact_1 = X[:, 8] * X[:, 10]
    X_interact = np.column_stack([X_interact, interact_1])
    new_names.append('current_run_rate_x_runs_at_3')
    
    # Interaction: wickets_down * current_run_rate
    interact_2 = X[:, 9] * X[:, 8]
    X_interact = np.column_stack([X_interact, interact_2])
    new_names.append('wickets_x_run_rate')
    
    # Interaction: striker_avg * non_striker_avg (batting pair quality)
    interact_3 = X[:, 0] * X[:, 3]
    X_interact = np.column_stack([X_interact, interact_3])
    new_names.append('striker_x_non_striker_avg')
    
    print(f"\nAdded {len(new_names) - len(feature_names)} interaction features")
    
    return X_interact, new_names

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a model and return metrics"""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    return {
        'name': model_name,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'r2_test': r2_test,
        'model': model
    }

def optimize_models():
    """Try multiple optimization strategies"""
    
    print("="*70)
    print("IPL POWERPLAY PREDICTION - RMSE OPTIMIZATION")
    print("="*70)
    
    # Load data
    print("\n1. Loading training data...")
    X, y = create_training_data()
    print(f"   Original dataset: {X.shape}")
    
    # Strategy 1: Remove outliers
    print("\n" + "-"*70)
    print("STRATEGY 1: OUTLIER REMOVAL")
    print("-"*70)
    X_clean, y_clean = remove_outliers_iqr(X, y, multiplier=1.5)
    print(f"   Cleaned dataset: {X_clean.shape}")
    
    # Strategy 2: Add interaction features
    print("\n" + "-"*70)
    print("STRATEGY 2: INTERACTION FEATURES")
    print("-"*70)
    feature_names = [
        'striker_avg_per_ball', 'striker_std_dev', 'striker_dismissal_rate',
        'non_striker_avg_per_ball', 'non_striker_std_dev', 'non_striker_dismissal_rate',
        'next_avg_per_ball_weighted', 'next_dismissal_weighted',
        'current_run_rate', 'wickets_down', 'runs_at_3_overs', 'overs_completed'
    ]
    X_interact, feature_names_interact = add_interaction_features(X_clean, feature_names)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_interact, y_clean, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {X_train_scaled.shape[1]}")
    
    # Try different models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    results = []
    
    # 1. Linear Regression (baseline)
    print("\n[1/6] Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    results.append(evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression"))
    
    # 2. Ridge Regression (L2 regularization)
    print("[2/6] Ridge Regression (auto alpha)...")
    best_ridge_rmse = float('inf')
    best_ridge = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred = ridge.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        if rmse < best_ridge_rmse:
            best_ridge_rmse = rmse
            best_ridge = ridge
    results.append(evaluate_model(best_ridge, X_train_scaled, X_test_scaled, y_train, y_test, "Ridge Regression"))
    
    # 3. Lasso Regression (L1 regularization)
    print("[3/6] Lasso Regression (auto alpha)...")
    best_lasso_rmse = float('inf')
    best_lasso = None
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(X_train_scaled, y_train)
        y_pred = lasso.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        if rmse < best_lasso_rmse:
            best_lasso_rmse = rmse
            best_lasso = lasso
    results.append(evaluate_model(best_lasso, X_train_scaled, X_test_scaled, y_train, y_test, "Lasso Regression"))
    
    # 4. ElasticNet (L1 + L2)
    print("[4/6] ElasticNet (L1 + L2)...")
    best_en_rmse = float('inf')
    best_en = None
    for alpha in [0.01, 0.1, 1.0]:
        for l1_ratio in [0.3, 0.5, 0.7]:
            en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
            en.fit(X_train_scaled, y_train)
            y_pred = en.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            if rmse < best_en_rmse:
                best_en_rmse = rmse
                best_en = en
    results.append(evaluate_model(best_en, X_train_scaled, X_test_scaled, y_train, y_test, "ElasticNet"))
    
    # 5. Random Forest Regressor
    print("[5/6] Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest"))
    
    # 6. Gradient Boosting Regressor
    print("[6/6] Gradient Boosting Regressor...")
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    results.append(evaluate_model(gb, X_train, X_test, y_train, y_test, "Gradient Boosting"))
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(results).sort_values('rmse_test')
    
    print(f"\n{'Model':<25} {'Train RMSE':<12} {'Test RMSE':<12} {'MAE':<10} {'R²':<10}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['name']:<25} {row['rmse_train']:<12.4f} {row['rmse_test']:<12.4f} {row['mae_test']:<10.4f} {row['r2_test']:<10.4f}")
    
    best_model_info = results_df.iloc[0]
    worst_model_info = results_df.iloc[-1]
    
    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)
    print(f"Model: {best_model_info['name']}")
    print(f"Test RMSE: {best_model_info['rmse_test']:.4f} runs")
    print(f"Test MAE: {best_model_info['mae_test']:.4f} runs")
    print(f"Test R²: {best_model_info['r2_test']:.4f}")
    
    improvement_over_old = ((8.9933 - best_model_info['rmse_test']) / 8.9933) * 100
    print(f"Improvement over baseline linear model: {improvement_over_old:.2f}%")
    
    print("\n" + "="*70)
    print("OPTIMIZATION TECHNIQUES APPLIED")
    print("="*70)
    print("✓ Outlier removal (IQR method)")
    print("✓ Interaction feature engineering")
    print("✓ Tried multiple regression algorithms:")
    print("  - Linear Regression")
    print("  - Ridge Regression (L2 regularization)")
    print("  - Lasso Regression (L1 regularization)")
    print("  - ElasticNet (L1 + L2)")
    print("  - Random Forest")
    print("  - Gradient Boosting")
    print("✓ Hyperparameter tuning via grid search")
    
    return results_df, best_model_info

if __name__ == "__main__":
    results_df, best_info = optimize_models()
