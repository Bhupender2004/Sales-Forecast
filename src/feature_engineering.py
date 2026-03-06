import pandas as pd
import numpy as np
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_data, clean_data

def feature_engineering(df):
    """
    Performs feature engineering:
    1. Creates time-based features (year, month, day, day_of_week)
    2. Encodes categorical variables (Label Encoding or One-Hot Encoding)
    """
    print("Starting feature engineering...")
    
    # Create a copy to avoid modifying the original
    df_feat = df.copy()

    # 1. Time-based features
    if 'Date' in df_feat.columns:
        df_feat['year'] = df_feat['Date'].dt.year
        df_feat['month'] = df_feat['Date'].dt.month
        df_feat['day'] = df_feat['Date'].dt.day
        df_feat['day_of_week'] = df_feat['Date'].dt.dayofweek
        
        # Optionally, drop the original Date column if using tree-based models
        df_feat = df_feat.drop('Date', axis=1)
        print("Time-based features created (year, month, day, day_of_week).")

    # 2. Encode categorical variables
    categorical_cols = df_feat.select_dtypes(include=['object']).columns.tolist()
    
    # Let's see which ones have high cardinality vs low cardinality
    high_cardinality = []
    low_cardinality = []
    
    for col in categorical_cols:
        if df_feat[col].nunique() > 10:
            high_cardinality.append(col)
        else:
            low_cardinality.append(col)
            
    # For high cardinality (like Store ID, Product ID), we can use Label Encoding to keep dimensions reasonable
    # For low cardinality (like Region, Weather Condition, Seasonality), we can use One-Hot Encoding (pd.get_dummies)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    label_encoders = {} # Store these if needed for later prediction
    
    for col in high_cardinality:
        print(f"Applying Label Encoding to {col}")
        df_feat[col] = le.fit_transform(df_feat[col].astype(str))
        label_encoders[col] = le
        
    for col in low_cardinality:
        print(f"Applying One-Hot Encoding to {col}")
        df_feat = pd.get_dummies(df_feat, columns=[col], drop_first=True)
        
    print(f"Feature engineering completed. Final shape: {df_feat.shape}")
    return df_feat, label_encoders

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_store_inventory.csv')
    df_raw = load_data(data_path)
    if df_raw is not None:
        df_cleaned = clean_data(df_raw)
        df_features, _ = feature_engineering(df_cleaned)
        print(df_features.head())
        print(df_features.info())
