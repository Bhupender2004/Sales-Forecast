import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import load_data, clean_data

def run_eda(df):
    """Generates visualizations for Exploratory Data Analysis."""
    print("Starting Exploratory Data Analysis...")
    
    # Create an output directory for plots if it doesn't exist
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # 1. Sales trend over time
    if 'Date' in df.columns:
        plt.figure(figsize=(12, 6))
        daily_sales = df.groupby('Date')['Units Sold'].sum().reset_index()
        sns.lineplot(data=daily_sales, x='Date', y='Units Sold')
        plt.title('Total Units Sold Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'sales_trend.png'))
        plt.close()
        print("Completed: Sales trend over time")

    # 2. Sales by region
    if 'Region' in df.columns:
        plt.figure(figsize=(10, 6))
        region_sales = df.groupby('Region')['Units Sold'].sum().reset_index().sort_values('Units Sold', ascending=False)
        sns.barplot(data=region_sales, x='Region', y='Units Sold', palette='viridis')
        plt.title('Total Units Sold by Region')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'sales_by_region.png'))
        plt.close()
        print("Completed: Sales by region")

    # 3. Sales by product category
    if 'Category' in df.columns:
        plt.figure(figsize=(12, 6))
        category_sales = df.groupby('Category')['Units Sold'].sum().reset_index().sort_values('Units Sold', ascending=False)
        sns.barplot(data=category_sales, x='Category', y='Units Sold', palette='rocket')
        plt.title('Total Units Sold by Product Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'sales_by_category.png'))
        plt.close()
        print("Completed: Sales by category")

    # 4. Price vs Units Sold
    if 'Price' in df.columns and 'Units Sold' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Price', y='Units Sold', alpha=0.5)
        plt.title('Price vs Units Sold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'price_vs_sales.png'))
        plt.close()
        print("Completed: Price vs Units Sold")

    # 5. Discount vs Units Sold
    if 'Discount' in df.columns and 'Units Sold' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Discount', y='Units Sold', palette='Set2')
        plt.title('Discount vs Units Sold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'discount_vs_sales.png'))
        plt.close()
        print("Completed: Discount vs Units Sold")

    # 6. Weather condition vs sales
    if 'Weather Condition' in df.columns:
        plt.figure(figsize=(10, 6))
        weather_sales = df.groupby('Weather Condition')['Units Sold'].mean().reset_index().sort_values('Units Sold', ascending=False)
        sns.barplot(data=weather_sales, x='Weather Condition', y='Units Sold', palette='coolwarm')
        plt.title('Average Units Sold by Weather Condition')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'weather_vs_sales.png'))
        plt.close()
        print("Completed: Weather condition vs sales")

    # 7. Holiday/promotion impact on sales
    if 'Holiday/Promotion' in df.columns:
        plt.figure(figsize=(8, 6))
        holiday_sales = df.groupby('Holiday/Promotion')['Units Sold'].mean().reset_index()
        holiday_sales['Holiday/Promotion'] = holiday_sales['Holiday/Promotion'].map({0: 'No', 1: 'Yes'})
        sns.barplot(data=holiday_sales, x='Holiday/Promotion', y='Units Sold', palette='pastel')
        plt.title('Average Units Sold on Regular Days vs Holidays/Promotions')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'holiday_vs_sales.png'))
        plt.close()
        print("Completed: Holiday/promotion impact on sales")

    print(f"EDA completed. Plots saved to '{plots_dir}'.")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_store_inventory.csv')
    df_raw = load_data(data_path)
    if df_raw is not None:
        df_cleaned = clean_data(df_raw)
        run_eda(df_cleaned)
