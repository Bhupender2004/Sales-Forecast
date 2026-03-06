import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import pickle
import sys

# Ensure src modules can be imported
sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_data, clean_data
from feature_engineering import feature_engineering

def prepare_data(df):
    """
    Defines the target variable y and features X.
    Splits the dataset into train and test sets.
    """
    print("Preparing data for modeling...")
    y = df['Units Sold']
    
    # Drop target and data leakage features
    cols_to_drop = ['Units Sold']
    if 'Demand Forecast' in df.columns:
        cols_to_drop.append('Demand Forecast')
    if 'Units Ordered' in df.columns:
        cols_to_drop.append('Units Ordered')
        
    X = df.drop(cols_to_drop, axis=1)

    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains an XGBoost Regressor model.
    Evaluates using MAE, RMSE, and R2.
    """
    print("Training XGBoost Regressor...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return model

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance of the trained model.
    """
    print("Plotting feature importance...")
    importance = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    # Take top 15 features to avoid a cluttered plot
    top_n = min(15, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importance[top_indices]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Top 15)")
    plt.barh(range(top_n), top_importances[::-1], align="center") # Reverse to have highest on top
    plt.yticks(range(top_n), top_features[::-1])
    plt.xlabel("Importance")
    plt.tight_layout()
    
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()
    print("Feature importance plot saved to notebooks/plots/feature_importance.png")

def save_model(model, label_encoders=None):
    """
    Saves the trained model and any label encoders using pickle.
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'sales_model.pkl')
    
    # Save both the model and the label encoders used
    save_data = {
        'model': model,
        'label_encoders': label_encoders,
        # Default categorical columns order if using pd.get_dummies
        # This can be helpful to ensure predictions have the same columns
    }
    
    with open(model_path, 'wb') as file:
        pickle.dump(save_data, file)
        
    print(f"Model saved successfully at {model_path}")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_store_inventory.csv')
    df_raw = load_data(data_path)
    
    if df_raw is not None:
        df_cleaned = clean_data(df_raw)
        df_features, label_encoders = feature_engineering(df_cleaned)
        
        X_train, X_test, y_train, y_test = prepare_data(df_features)
        
        model = train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Save training columns for predict script to align features
        training_cols = X_train.columns.tolist()
        if label_encoders is None:
            label_encoders = {}
        label_encoders['training_columns'] = training_cols
        
        plot_feature_importance(model, X_train.columns)
        save_model(model, label_encoders)
