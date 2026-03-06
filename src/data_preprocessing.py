import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """Loads the dataset from the given filepath."""
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def clean_data(df):
    """Cleans the dataset by handling missing values, duplicates, and datatypes."""
    print("Starting data cleaning...")
    
    # Check for missing values
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    
    # Convert Date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
    # Handle missing values - simple imputation
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # For categorical columns, fill with mode
    categoric_cols = df.select_dtypes(include=['object']).columns
    for col in categoric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Remove duplicates
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        print(f"Removing {duplicates_count} duplicate rows.")
        df = df.drop_duplicates()
        
    # Sort data by Date
    if 'Date' in df.columns:
        df = df.sort_values(by='Date').reset_index(drop=True)
        
    print("Data cleaning completed.")
    return df

if __name__ == "__main__":
    # Test the functions
    data_path = os.path.join("..", "data", "retail_store_inventory.csv")
    df = load_data(data_path)
    if df is not None:
        df_cleaned = clean_data(df)
        print(df_cleaned.head())
