import pandas as pd
import os
import pickle
import sys

# Ensure src modules can be imported
sys.path.append(os.path.dirname(__file__))
from feature_engineering import feature_engineering

def load_trained_model():
    """
    Loads the trained model and label encoders from the models directory.
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, 'sales_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None
        
    with open(model_path, 'rb') as file:
        saved_data = pickle.load(file)
        
    print(f"Model loaded successfully from {model_path}")
    return saved_data['model'], saved_data.get('label_encoders')

def predict_sales(df_input, model, label_encoders):
    """
    Predicts sales (Units Sold) using the loaded model and input data.
    """
    df_feat = df_input.copy()
    
    # 1. Time-based features
    if 'Date' in df_feat.columns:
        df_feat['year'] = df_feat['Date'].dt.year
        df_feat['month'] = df_feat['Date'].dt.month
        df_feat['day'] = df_feat['Date'].dt.day
        df_feat['day_of_week'] = df_feat['Date'].dt.dayofweek
        df_feat = df_feat.drop('Date', axis=1)

    # 2. Categorical features
    training_cols = label_encoders.get('training_columns', [])
    
    # Encode high cardinality using saved label encoders
    for col, le in label_encoders.items():
        if col != 'training_columns' and col in df_feat.columns:
            # Handle unseen labels by mapping to a known class (first class)
            known_classes = set(le.classes_)
            df_feat[col] = df_feat[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            df_feat[col] = le.transform(df_feat[col].astype(str))
            
    # Perform get_dummies on the rest of the categorical columns
    # We don't drop first, because we align exactly to training_cols anyway
    categorical_cols = df_feat.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df_feat = pd.get_dummies(df_feat, columns=categorical_cols, drop_first=False)
        
    # Ensure types match before predicting
    # Add missing columns with 0
    if training_cols:
        for c in training_cols:
            if c not in df_feat.columns:
                df_feat[c] = 0
                
        # Ensure the order of columns matches the training data
        df_feat = df_feat[training_cols]
        
    y_pred = model.predict(df_feat)
    
    return y_pred

if __name__ == "__main__":
    # Test the prediction script with an example
    model, label_encoders = load_trained_model()
    
    if model is not None:
        # Create a dummy dataframe matching the raw data format to test prediction
        dummy_data = {
            'Date': ['2023-01-01'],
            'Store ID': ['S001'],
            'Product ID': ['P0001'],
            'Category': ['Electronics'],
            'Region': ['North'],
            'Inventory Level': [100],
            'Units Ordered': [50],
            'Demand Forecast': [120],
            'Price': [29.99],
            'Discount': [5.0],
            'Weather Condition': ['Sunny'],
            'Holiday/Promotion': [0],
            'Competitor Pricing': [28.99],
            'Seasonality': ['Winter']
        }
        df_dummy = pd.DataFrame(dummy_data)
        df_dummy['Date'] = pd.to_datetime(df_dummy['Date'])
        
        prediction = predict_sales(df_dummy, model, label_encoders)
        print(f"Predicted Units Sold: {prediction[0]:.2f}")
