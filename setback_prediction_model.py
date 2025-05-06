import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Try importing LightGBM
try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM is not installed. Installing it now...")
    import pip
    pip.main(['install', 'lightgbm'])
    import lightgbm as lgb

# Define file paths
input_file = "data/output/all_buildings_merged.csv"
output_dir = "data/output/"

def prepare_data(df):
    """
    Prepare the dataset for model training by handling missing values and creating features
    """
    print(f"Original dataset shape: {df.shape}")
    
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
    
    # For SetbackActive, we'll use 0 as the default (no setback)
    df['SetbackActive'] = df['SetbackActive'].fillna(0)
    
    print(f"Processed dataset shape: {df.shape}")
    print(f"Missing values after processing:\n{df.isnull().sum()}")
    
    return df

def train_model(df):
    """
    Train a LightGBM model to predict SetbackActive
    """
    # Select features for model
    features = [
        'SupplyTemp', 'ReturnTemp', 'OutsideTemp', 'SupplyTempReturnTempDerivation',
        'Hour', 'DayOfWeek', 'Month', 'DayOfMonth', 'IsWeekend', 'TimeOfDay_Encoded'
    ]
    
    # Only include SupplyReturnRatio if it doesn't have too many NaN values
    if df['SupplyReturnRatio'].isna().sum() / len(df) < 0.1:  # If less than 10% missing
        features.append('SupplyReturnRatio')
    
    # Define target
    target = 'SetbackActive'
    
    # Prepare X and y
    X = df[features].copy()
    # Replace any remaining NaN values with column means
    X = X.fillna(X.mean())
    y = df[target]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Features used: {', '.join(features)}")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Define model parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Train model
    print("Training LightGBM model...")
    callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, test_data],
        callbacks=callbacks
    )
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.round(y_pred_proba)
    
    # Evaluate model
    evaluate_model(y_test, y_pred, y_pred_proba, model, features)
    
    return model, X_test, y_test

def evaluate_model(y_true, y_pred, y_pred_proba, model, features):
    """
    Evaluate the model performance using various metrics
    """
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    # Calculate ROC AUC only if there are both positive and negative classes
    if len(np.unique(y_true)) > 1:
        print(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
    else:
        print("ROC AUC: Not calculated (only one class present in test set)")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negative: {cm[0][0]}, False Positive: {cm[0][1]}")
    print(f"False Negative: {cm[1][0]}, True Positive: {cm[1][1]}")
    
    try:
        # Create simple confusion matrix visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Not Active', 'Active'], rotation=45)
        plt.yticks(tick_marks, ['Not Active', 'Active'])
        
        # Add text annotations to the confusion matrix cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{output_dir}confusion_matrix.png")
        plt.close()
        
        # Plot feature importance
        importance = model.feature_importance()
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(importance_df['Feature'], importance_df['Importance'])
        plt.title('Feature Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}feature_importance.png")
        plt.close()
        
        print(f"\nEvaluation plots saved to {output_dir}")
    except Exception as e:
        print(f"Warning: Could not create plots. Error: {str(e)}")
        print("Continuing without plots...")

def main():
    # Start time
    start_time = datetime.now()
    print(f"Starting SetbackActive prediction model at {start_time}")
    
    # Load the data
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Prepare the data
    df = prepare_data(df)
    
    # Train and evaluate the model
    model, X_test, y_test = train_model(df)
    
    # Save the model
    model_path = f"{output_dir}setback_prediction_model.txt"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # End time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Process completed at {end_time}")
    print(f"Total duration: {duration}")

if __name__ == "__main__":
    main()