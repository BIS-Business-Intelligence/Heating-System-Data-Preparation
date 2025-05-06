import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Define file paths
model_path = "data/output/setback_prediction_model.txt"
output_dir = "data/output/"

def prepare_prediction_data(df):
    """
    Apply the same data preparation steps used during training
    """
    print(f"Original data shape: {df.shape}")
    
    # Convert timestamp to datetime if it's not already
    if df['Timestamp'].dtype == 'object':
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Extract time-based features
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['DayOfMonth'] = df['Timestamp'].dt.day
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Feature to capture time of day
    df['TimeOfDay'] = df['Hour'].apply(lambda x: 
                                       'Night' if 0 <= x < 6 else
                                       'Morning' if 6 <= x < 12 else
                                       'Afternoon' if 12 <= x < 18 else
                                       'Evening')
    
    # Convert categorical to numeric
    le = LabelEncoder()
    df['TimeOfDay_Encoded'] = le.fit_transform(df['TimeOfDay'])
    
    # Calculate temperature differences and ratios where possible
    mask = (df['SupplyTemp'].notna() & df['ReturnTemp'].notna() & (df['ReturnTemp'] != 0))
    df.loc[mask, 'SupplyReturnRatio'] = df.loc[mask, 'SupplyTemp'] / df.loc[mask, 'ReturnTemp']
    
    # Handle missing values
    # For temperature data, forward fill is logical as temperature changes slowly
    df['SupplyTemp'] = df['SupplyTemp'].ffill().bfill()
    df['ReturnTemp'] = df['ReturnTemp'].ffill().bfill()
    df['OutsideTemp'] = df['OutsideTemp'].ffill().bfill()
    
    # Fill missing values in SupplyTempReturnTempDerivation based on the formula
    df['SupplyTempReturnTempDerivation'] = df.apply(
        lambda row: abs(row['SupplyTemp'] - row['ReturnTemp']) 
                    if pd.isna(row['SupplyTempReturnTempDerivation']) 
                    else row['SupplyTempReturnTempDerivation'], 
        axis=1
    )
    
    print(f"Processed data shape: {df.shape}")
    
    return df

def make_predictions(model, data):
    """
    Generate predictions using the trained model
    """
    # Select the same features used during training
    features = [
        'SupplyTemp', 'ReturnTemp', 'OutsideTemp', 'SupplyTempReturnTempDerivation',
        'Hour', 'DayOfWeek', 'Month', 'DayOfMonth', 'IsWeekend', 'TimeOfDay_Encoded',
        'SupplyReturnRatio'
    ]
    
    # Create feature matrix, ensuring all features are available
    X = data[features].copy()
    
    # Fill any missing values with column means
    X = X.fillna(X.mean())
    
    # Generate predictions
    y_pred_proba = model.predict(X)
    y_pred = np.round(y_pred_proba)
    
    # Add predictions to the original data
    data['SetbackActive_Predicted'] = y_pred
    data['SetbackActive_Probability'] = y_pred_proba
    
    return data

def visualize_predictions(data, time_window=None):
    """
    Visualize the predictions and actual values over time
    """
    # Create a copy of the data to avoid modification warnings
    plot_data = data.copy()
    
    # Filter data for a specified time window if provided
    if time_window:
        start_time, end_time = time_window
        plot_data = plot_data[(plot_data['Timestamp'] >= start_time) & 
                              (plot_data['Timestamp'] <= end_time)]
    
    # Set up the figure
    plt.figure(figsize=(14, 8))
    
    # Plot actual setback values if available
    if 'SetbackActive' in plot_data.columns:
        plt.scatter(plot_data['Timestamp'], plot_data['SetbackActive'], 
                   color='blue', alpha=0.6, s=20, label='Actual')
    
    # Plot predicted probabilities
    plt.scatter(plot_data['Timestamp'], plot_data['SetbackActive_Probability'], 
               color='red', alpha=0.6, s=20, label='Predicted Probability')
    
    # Plot threshold line at 0.5
    plt.axhline(y=0.5, color='green', linestyle='--', label='Threshold (0.5)')
    
    # Format the plot
    plt.title('Setback Prediction Over Time')
    plt.xlabel('Time')
    plt.ylabel('Setback Status (Actual/Predicted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}setback_predictions_over_time.png")
    plt.close()
    
    print(f"Prediction visualization saved to {output_dir}setback_predictions_over_time.png")

def evaluate_predictions(data):
    """
    If actual values are available, evaluate prediction performance
    """
    if 'SetbackActive' not in data.columns:
        print("No actual values available for evaluation")
        return
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Create a copy of the data to work with
    eval_data = data.copy()
    
    # Drop rows with NaN values in SetbackActive
    nan_count = eval_data['SetbackActive'].isna().sum()
    if nan_count > 0:
        print(f"Warning: Dropping {nan_count} rows with NaN values in SetbackActive for evaluation")
        eval_data = eval_data.dropna(subset=['SetbackActive'])
    
    y_true = eval_data['SetbackActive']
    y_pred = eval_data['SetbackActive_Predicted']
    y_pred_proba = eval_data['SetbackActive_Probability']
    
    print("\nPrediction Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    if len(np.unique(y_true)) > 1:
        print(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
    else:
        print("ROC AUC: Not calculated (only one class present in test set)")
    
    return eval_data

def main():
    """
    Main function to load model and make predictions
    """
    # Start time
    start_time = datetime.now()
    print(f"Starting prediction process at {start_time}")
    
    # Load the trained model
    print(f"Loading model from {model_path}")
    model = lgb.Booster(model_file=model_path)
    
    # Two demo methods to load data for prediction:
    # 1. Use existing data from merged file (simulates new data)
    demo_data_path = "data/output/all_buildings_merged.csv"
    df = pd.read_csv(demo_data_path)
    
    # 2. Use raw source files (uncomment below to use this method instead)
    # from helpers import data_cleaner_helper as dc_helper
    # df = dc_helper.load_building_data("Building 2", "data/input/", "Building 2")
    
    # Take a sample of the data to simulate new data
    sample_size = min(1000, len(df))
    new_data = df.sample(sample_size, random_state=42)
    
    # Process the data
    processed_data = prepare_prediction_data(new_data)
    
    # Make predictions
    print("Generating predictions...")
    results = make_predictions(model, processed_data)
    
    # Save prediction results
    results_path = f"{output_dir}prediction_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Prediction results saved to {results_path}")
    
    # Visualize predictions
    visualize_predictions(results)
    
    # Evaluate predictions if actual values are available
    evaluate_predictions(results)
    
    # End time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Prediction process completed at {end_time}")
    print(f"Total duration: {duration}")

if __name__ == "__main__":
    main()