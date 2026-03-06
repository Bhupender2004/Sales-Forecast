# Retail Sales Forecasting System

## Project Overview
The **Retail Sales Forecasting System** is an end-to-end Machine Learning project designed to predict future product sales (Units Sold) based on historical retail data. It includes data preprocessing, exploratory data analysis, feature engineering, model training (using XGBoost), and an interactive web dashboard built with Streamlit.

## Dataset Description
The dataset (`data/retail_store_inventory.csv`) contains historical retail records with the following columns:
- `Date`: Record date
- `Store ID` & `Product ID`: Identifiers
- `Category`: Product category (e.g., Electronics, Toys, Clothing, Groceries)
- `Region`: Store location (e.g., North, South, East, West)
- `Inventory Level`: Current stock
- `Units Ordered`: Replenishment orders
- `Target -> Units Sold`: The value we are forecasting
- `Demand Forecast`: Base demand predicted by baseline systems
- `Price` & `Discount`: Pricing attributes
- `Weather Condition`: Categorical weather (e.g., Sunny, Rainy)
- `Holiday/Promotion`: Binary flag for special events
- `Competitor Pricing`: Related market price
- `Seasonality`: The time of year (e.g., Winter, Summer)

## Technologies Used
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Data splitting, encoding, and evaluation metrics
- **XGBoost**: Advanced Gradient Boosting machine learning model
- **Streamlit**: Interactive web application framework

## Project Structure
```text
retail-sales-forecasting/
│
├── data/
│   └── retail_store_inventory.csv     # Raw dataset
│
├── notebooks/
│   ├── eda_script.py                  # Exploratory Data Analysis script
│   └── plots/                         # Generated EDA visualizations
│
├── src/
│   ├── data_preprocessing.py          # Data cleaning and formatting
│   ├── feature_engineering.py         # Feature creation and variable encoding
│   ├── train_model.py                 # Model training and evaluation
│   └── predict.py                     # Prediction script wrapper
│
├── models/
│   └── sales_model.pkl                # Trained model and encoders
│
├── app/
│   └── streamlit_app.py               # Streamlit Dashboard
│
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

## How to Run the Project

1. **Install Dependencies**
   It's recommended to use a virtual environment. Install required packages using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Exploratory Data Analysis (EDA) Plots**
   Run the EDA script to analyze the data and save visualizations to the `notebooks/plots` directory:
   ```bash
   python notebooks/eda_script.py
   ```

3. **Train the Machine Learning Model**
   Run the training pipeline. This script will preprocess the data, perform feature engineering, train an XGBoost model, evaluate its performance, and save the artifact to `models/sales_model.pkl`:
   ```bash
   python src/train_model.py
   ```

4. **Run the Streamlit Dashboard**
   Once the model is trained, start the interactive web application to begin predicting sales:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Open the provided URL (typically `http://localhost:8501`) in your web browser.

## Future Improvements
- **Model Tuning:** Perform hyperparameter optimization (e.g., GridSearchCV, Optuna) to improve prediction accuracy.
- **Advanced Feature Engineering:** Add rolling averages, lag features, and external data sources (e.g., macroeconomic indicators).
- **Time-Series Models:** Experiment with specialized time-series forecasting algorithms (e.g., Prophet, ARIMA, LSTMs).
- **Deployment:** Containerize the application with Docker and deploy to a cloud platform like AWS, Google Cloud, or Heroku.
