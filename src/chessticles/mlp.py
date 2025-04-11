import json
import os
import pickle
import sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Create output directories if they don't exist
def create_directories():
    """Create necessary directories for saving models, plots, and metrics"""
    dirs = ["models", "plots", "metrics"]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


# Connect to SQLite database
def load_data(db_path):
    """Load chess game data from SQLite database and extract features.

    Args:
        db_path: Path to the SQLite database

    Returns:
        DataFrame with extracted features
    """
    conn = sqlite3.connect(db_path)

    # Query to get all relevant game data including Elo ratings
    query = """
    SELECT 
        a.game_id, 
        a.white_acl, a.black_acl, 
        a.white_blunders, a.white_mistakes, a.white_inaccuracies,
        a.black_blunders, a.black_mistakes, a.black_inaccuracies,
        a.time_eval_data, a.time_control, a.estimated_time, a.game_type,
        b.whiteelo, b.blackelo
    FROM game_analysis a
    inner join games b on a.game_id = b.id;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


# Feature engineering functions
def extract_time_eval_features(df):
    """Extract features from the time_eval_data JSON blob.

    Args:
        df: DataFrame containing the game data

    Returns:
        DataFrame with additional time/eval features
    """
    # Create new columns for the extracted features
    feature_df = df.copy()

    # Features to extract
    feature_df["white_time_usage"] = np.nan
    feature_df["black_time_usage"] = np.nan
    feature_df["white_eval_volatility"] = np.nan
    feature_df["black_eval_volatility"] = np.nan
    feature_df["white_avg_time_per_move"] = np.nan
    feature_df["black_avg_time_per_move"] = np.nan
    feature_df["white_time_pressure_moves"] = np.nan
    feature_df["black_time_pressure_moves"] = np.nan
    feature_df["position_complexity"] = np.nan
    feature_df["max_eval_advantage_white"] = np.nan
    feature_df["max_eval_advantage_black"] = np.nan

    # Process each game
    for idx, row in df.iterrows():
        try:
            # Parse JSON
            data = json.loads(row["time_eval_data"])

            # Extract time usage patterns
            white_times = np.array(data["white"]["times"])
            black_times = np.array(data["black"]["times"])

            # Time usage (difference between start and end)
            white_time_usage = white_times[0] - white_times[-1]
            black_time_usage = black_times[0] - black_times[-1]

            # Calculate time deltas between moves
            white_time_deltas = np.diff(white_times)
            black_time_deltas = np.diff(black_times)

            # Average time per move
            white_avg_time = np.mean(np.abs(white_time_deltas))
            black_avg_time = np.mean(np.abs(black_time_deltas))

            # Count moves made under time pressure (less than 10 seconds)
            white_time_pressure = sum(np.diff(white_times) > 10)
            black_time_pressure = sum(np.diff(black_times) > 10)

            # Extract evaluation patterns
            white_evals = np.array(data["white"]["evals"])
            black_evals = np.array(data["black"]["evals"])

            # Cap extreme evaluation values for better feature stability
            white_evals = np.clip(white_evals, -1000, 1000)
            black_evals = np.clip(black_evals, -1000, 1000)

            # Evaluation volatility (standard deviation of evaluations)
            white_eval_volatility = np.std(white_evals)
            black_eval_volatility = np.std(black_evals)

            # Maximum advantage
            max_eval_advantage_white = np.max(white_evals)
            max_eval_advantage_black = np.max(-np.array(black_evals))  # Negate for black's perspective

            # Position complexity (average of absolute changes in evaluation)
            white_eval_changes = np.abs(np.diff(white_evals))
            black_eval_changes = np.abs(np.diff(black_evals))
            position_complexity = (np.mean(white_eval_changes) + np.mean(black_eval_changes)) / 2

            # Store extracted features
            feature_df.loc[idx, "white_time_usage"] = white_time_usage
            feature_df.loc[idx, "black_time_usage"] = black_time_usage
            feature_df.loc[idx, "white_eval_volatility"] = white_eval_volatility
            feature_df.loc[idx, "black_eval_volatility"] = black_eval_volatility
            feature_df.loc[idx, "white_avg_time_per_move"] = white_avg_time
            feature_df.loc[idx, "black_avg_time_per_move"] = black_avg_time
            feature_df.loc[idx, "white_time_pressure_moves"] = white_time_pressure
            feature_df.loc[idx, "black_time_pressure_moves"] = black_time_pressure
            feature_df.loc[idx, "position_complexity"] = position_complexity
            feature_df.loc[idx, "max_eval_advantage_white"] = max_eval_advantage_white
            feature_df.loc[idx, "max_eval_advantage_black"] = max_eval_advantage_black

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Handle corrupted JSON or missing keys
            print(f"Error processing game {row['game_id']}: {e}")
            continue

    # Drop rows with NaN values created during feature extraction
    feature_df = feature_df.dropna()

    return feature_df


def prepare_features_targets(df):
    """Prepare feature sets and target variables for the ML model.

    Args:
        df: DataFrame with extracted features

    Returns:
        X_white, y_white, X_black, y_black: Features and targets for white and black players
    """
    # Common features for both white and black
    common_features = [
        "white_acl",
        "black_acl",  # Average centipawn loss
        "position_complexity",
        "game_type_encoded",  # Encoded game type
        "estimated_time",  # Estimated game duration
    ]

    # Features specific to white player
    white_features = common_features + [
        "white_blunders",
        "white_mistakes",
        "white_inaccuracies",
        "white_time_usage",
        "white_eval_volatility",
        "white_avg_time_per_move",
        "white_time_pressure_moves",
        "max_eval_advantage_white",
        # "opponent_elo",  # Black's Elo (known for prediction)
    ]

    # Features specific to black player
    black_features = common_features + [
        "black_blunders",
        "black_mistakes",
        "black_inaccuracies",
        "black_time_usage",
        "black_eval_volatility",
        "black_avg_time_per_move",
        "black_time_pressure_moves",
        "max_eval_advantage_black",
        # "opponent_elo",  # White's Elo (known for prediction)
    ]

    # Create opponent Elo columns
    df_white = df.copy()
    df_black = df.copy()

    # df_white["opponent_elo"] = df_white["BlackElo"]
    # df_black["opponent_elo"] = df_black["WhiteElo"]

    # Encode categorical variables
    game_type_mapping = {gt: i for i, gt in enumerate(df["game_type"].unique())}
    df_white["game_type_encoded"] = df_white["game_type"].map(game_type_mapping)
    df_black["game_type_encoded"] = df_black["game_type"].map(game_type_mapping)

    # Extract features and targets
    X_white = df_white[white_features]
    y_white = df_white["WhiteElo"]

    X_black = df_black[black_features]
    y_black = df_black["BlackElo"]

    return X_white, y_white, X_black, y_black, white_features, black_features, game_type_mapping


# Model training and evaluation
def train_evaluate_model(X, y, feature_names, player_color, model_type="mlp", grid_search=True):
    """Train and evaluate a model for Elo prediction.

    Args:
        X: Feature matrix
        y: Target variable (Elo ratings)
        feature_names: Names of features used in the model
        player_color: 'white' or 'black' to indicate which player's model
        model_type: Type of model to train ('mlp' for MLPRegressor)
        grid_search: Whether to perform grid search for hyperparameter tuning

    Returns:
        Trained model, test data, test predictions, and evaluation metrics
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(random_state=42))])

    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"{player_color}_{timestamp}"

    # Dictionary to store metrics
    metrics = {
        "model_version": model_version,
        "player_color": player_color,
        "feature_names": feature_names.tolist(),
        "training_size": len(X_train),
        "testing_size": len(X_test),
        "timestamp": timestamp,
    }

    # Hyperparameter tuning with grid search
    if grid_search:
        param_grid = {
            "model__hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
            "model__activation": ["identity", "relu", "tanh"],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate": ["constant", "adaptive", "invscaling"],
            "model__max_iter": [5000],
        }

        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid.fit(X_train, y_train)

        # Get best model
        model = grid.best_estimator_
        print(f"Best parameters: {grid.best_params_}")

        # Store best parameters in metrics
        metrics["best_parameters"] = grid.best_params_
    else:
        # Use default parameters
        model = pipeline
        model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate model
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Calculate RMSE
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    # Store metrics in the dictionary
    metrics.update(
        {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
        },
    )

    # Display results
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Testing MAE: {test_mae:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}")

    # Save model to disk
    model_filename = f"models/{player_color}_model_{timestamp}.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_filename}")

    # Save metrics to disk
    metrics_filename = f"metrics/{player_color}_metrics_{timestamp}.json"
    with open(metrics_filename, "w") as file:
        json.dump(metrics, file, indent=4)
    print(f"Metrics saved to {metrics_filename}")

    return model, X_test, y_test, y_pred_test, metrics


def visualize_results(y_test, y_pred, player_color, timestamp):
    """Visualize model predictions against actual Elo ratings.

    Args:
        y_test: Actual Elo ratings
        y_pred: Predicted Elo ratings
        player_color: 'White' or 'Black' to indicate which player's model
        timestamp: Timestamp for file naming
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot of actual vs predicted values
    plt.scatter(y_test, y_pred, alpha=0.5)

    # Plot the perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plt.xlabel("Actual Elo Rating")
    plt.ylabel("Predicted Elo Rating")
    plt.title(f"{player_color} Player Elo: Actual vs Predicted")
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plot_filename = f"plots/{player_color}_actual_vs_predicted_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filename}")

    # Error distribution plot
    errors = y_pred - y_test

    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Prediction Errors ({player_color} Player)")
    plt.axvline(x=0, color="r", linestyle="--")
    plt.grid(True)

    plt.tight_layout()

    # Save the error distribution plot
    error_plot_filename = f"plots/{player_color}_error_distribution_{timestamp}.png"
    plt.savefig(error_plot_filename, dpi=300)
    plt.close()
    print(f"Error distribution plot saved to {error_plot_filename}")

    return plot_filename, error_plot_filename


def analyze_feature_importance(model, X, feature_names, player_color, timestamp):
    """Analyze feature importance for the trained model and save the visualization.

    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        player_color: 'White' or 'Black' to indicate which player's model
        timestamp: Timestamp for file naming

    Returns:
        DataFrame with feature importances and plot filename
    """
    # For MLPRegressor, we can use the connection weights as a proxy for importance
    # This is a simplified approach - more sophisticated methods exist

    # Extract the MLPRegressor from the pipeline
    mlp = model.named_steps["model"]

    # Get the weights for the first layer
    weights = mlp.coefs_[0]

    # Calculate the absolute sum of weights for each feature
    importances = np.sum(np.abs(weights), axis=1)

    # Normalize the importances
    importances = importances / np.sum(importances)

    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

    # Sort by importance
    importance_df = importance_df.sort_values("Importance", ascending=False)

    # Save feature importance data
    importance_filename = f"metrics/{player_color}_feature_importance_{timestamp}.csv"
    importance_df.to_csv(importance_filename, index=False)
    print(f"Feature importance data saved to {importance_filename}")

    # Visualize
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title(f"Feature Importance ({player_color} Player)")
    plt.tight_layout()

    # Save the plot
    importance_plot_filename = f"plots/{player_color}_feature_importance_{timestamp}.png"
    plt.savefig(importance_plot_filename, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to {importance_plot_filename}")

    return importance_df, importance_plot_filename


def predict_elo(model, game_data, player_color="white"):
    """Predict Elo rating for a new game.

    Args:
        model: Trained model
        game_data: Dictionary with game features
        player_color: 'white' or 'black'

    Returns:
        Predicted Elo rating
    """
    # Extract and prepare features based on player color
    if player_color.lower() == "white":
        # Prepare features for white player prediction
        features = [
            game_data.get("white_acl", 0),
            game_data.get("black_acl", 0),
            game_data.get("position_complexity", 0),
            game_data.get("game_type_encoded", 0),
            game_data.get("estimated_time", 0),
            game_data.get("white_blunders", 0),
            game_data.get("white_mistakes", 0),
            game_data.get("white_inaccuracies", 0),
            game_data.get("white_time_usage", 0),
            game_data.get("white_eval_volatility", 0),
            game_data.get("white_avg_time_per_move", 0),
            game_data.get("white_time_pressure_moves", 0),
            game_data.get("max_eval_advantage_white", 0),
            # game_data.get("opponent_elo", 0),  # Black's Elo
        ]
    else:
        # Prepare features for black player prediction
        features = [
            game_data.get("white_acl", 0),
            game_data.get("black_acl", 0),
            game_data.get("position_complexity", 0),
            game_data.get("game_type_encoded", 0),
            game_data.get("estimated_time", 0),
            game_data.get("black_blunders", 0),
            game_data.get("black_mistakes", 0),
            game_data.get("black_inaccuracies", 0),
            game_data.get("black_time_usage", 0),
            game_data.get("black_eval_volatility", 0),
            game_data.get("black_avg_time_per_move", 0),
            game_data.get("black_time_pressure_moves", 0),
            game_data.get("max_eval_advantage_black", 0),
            # game_data.get("opponent_elo", 0),  # White's Elo
        ]

    # Reshape for single prediction
    X_new = np.array(features).reshape(1, -1)

    # Make prediction
    predicted_elo = model.predict(X_new)[0]

    return predicted_elo


# Function to save and load models
def save_model(model, filename):
    """Save a model to disk"""
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    return filename


def load_model(filename):
    """Load a model from disk"""
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model


# Main execution function
def main(db_path):
    """Main function to execute the Elo prediction pipeline.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Dictionary with model paths and metrics
    """
    # Create necessary directories
    create_directories()

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data...")
    df = load_data(db_path)
    print(f"Loaded {len(df)} games from database.")

    print("\nExtracting features from time and evaluation data...")
    df_features = extract_time_eval_features(df)
    print(f"Extracted features for {len(df_features)} games.")

    print("\nPreparing features and targets...")
    X_white, y_white, X_black, y_black, white_features, black_features, game_type_mapping = prepare_features_targets(
        df_features,
    )

    # Save feature lists and mappings for future use
    feature_data = {
        "white_features": white_features,
        "black_features": black_features,
        "game_type_mapping": game_type_mapping,
        "timestamp": timestamp,
    }

    with open(f"metrics/feature_metadata_{timestamp}.json", "w") as file:
        json.dump(feature_data, file, indent=4)

    # Train white model
    print("\nTraining and evaluating model for white players...")
    white_model, X_test_white, y_test_white, y_pred_white, white_metrics = train_evaluate_model(
        X_white,
        y_white,
        X_white.columns,
        "white",
        grid_search=True,
    )

    # Train black model
    print("\nTraining and evaluating model for black players...")
    black_model, X_test_black, y_test_black, y_pred_black, black_metrics = train_evaluate_model(
        X_black,
        y_black,
        X_black.columns,
        "black",
        grid_search=True,
    )

    print("\nVisualizing results for white players...")
    white_plots = visualize_results(y_test_white, y_pred_white, "White", timestamp)

    print("\nVisualizing results for black players...")
    black_plots = visualize_results(y_test_black, y_pred_black, "Black", timestamp)

    print("\nAnalyzing feature importance for white model...")
    white_importance, white_importance_plot = analyze_feature_importance(
        white_model,
        X_white,
        X_white.columns,
        "White",
        timestamp,
    )

    print("\nAnalyzing feature importance for black model...")
    black_importance, black_importance_plot = analyze_feature_importance(
        black_model,
        X_black,
        X_black.columns,
        "Black",
        timestamp,
    )

    # Save a summary of all results
    run_summary = {
        "run_timestamp": timestamp,
        "white_model": f"models/white_model_{timestamp}.pkl",
        "black_model": f"models/black_model_{timestamp}.pkl",
        "white_metrics": white_metrics,
        "black_metrics": black_metrics,
        "white_plots": white_plots,
        "black_plots": black_plots,
        "white_importance_plot": white_importance_plot,
        "black_importance_plot": black_importance_plot,
        "feature_metadata": f"metrics/feature_metadata_{timestamp}.json",
    }

    with open(f"metrics/run_summary_{timestamp}.json", "w") as file:
        json.dump(run_summary, file, indent=4)
    print(f"\nRun summary saved to metrics/run_summary_{timestamp}.json")

    return white_model, black_model, run_summary


if __name__ == "__main__":
    # Replace with your actual database path
    db_path = "data/db.ocgdb.db3"

    # Execute the pipeline
    white_model, black_model, run_summary = main(db_path)

    # Example of predicting Elo for a new game
    print("\nExample: Predicting Elo for a new game...")

    # Sample game data (you would extract this from a real game)
    new_game = {
        "white_acl": 120.5,
        "black_acl": 118.2,
        "white_blunders": 3,
        "white_mistakes": 5,
        "white_inaccuracies": 7,
        "black_blunders": 4,
        "black_mistakes": 6,
        "black_inaccuracies": 8,
        "position_complexity": 45.3,
        "white_time_usage": 250.5,
        "black_time_usage": 275.2,
        "white_eval_volatility": 120.7,
        "black_eval_volatility": 135.2,
        "white_avg_time_per_move": 12.5,
        "black_avg_time_per_move": 13.8,
        "white_time_pressure_moves": 5,
        "black_time_pressure_moves": 6,
        "max_eval_advantage_white": 250,
        "max_eval_advantage_black": 150,
        "game_type_encoded": 1,  # Rapid
        "estimated_time": 600,
        "opponent_elo": 1800,  # Known Elo of opponent
    }

    # Predict for white player
    white_predicted_elo = predict_elo(white_model, new_game, "white")
    print(f"Predicted White Elo: {white_predicted_elo:.0f}")

    # Update opponent Elo for black player prediction
    # new_game["opponent_elo"] = white_predicted_elo

    # Predict for black player
    black_predicted_elo = predict_elo(black_model, new_game, "black")
    print(f"Predicted Black Elo: {black_predicted_elo:.0f}")

    timestamp = (datetime.now().strftime("%Y%m%d_%H%M%S"),)
    # Save prediction example
    prediction_example = {
        "timestamp": timestamp,
        "game_data": new_game,
        "white_predicted_elo": float(white_predicted_elo),
        "black_predicted_elo": float(black_predicted_elo),
    }

    with open(f"metrics/prediction_example_{timestamp}.json", "w") as file:
        json.dump(prediction_example, file, indent=4)
    print(f"Prediction example saved to metrics/prediction_example_{timestamp}.json")
