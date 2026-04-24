import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import load_trained_model, predict_sales
from data_preprocessing import load_data, clean_data

# Set page config
st.set_page_config(
    page_title="Retail Insights | AI Demand Forecasting",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply UI Styling - Simple Green Theme
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Soft Green Background for entire app */
    .stApp {
        background-color: #F0FDF4; /* Light emerald scale */
    }

    /* Simple Header */
    .dashboard-title {
        color: #064E3B; /* Dark green */
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .dashboard-subtitle {
        font-size: 1.2rem;
        color: #047857; /* Medium green */
        font-weight: 500;
        margin-bottom: 20px;
    }
    
    /* Horizontal Divider */
    .gradient-divider {
        height: 3px;
        background-color: #10B981; /* Emerald */
        border: none;
        margin-top: 0px;
        margin-bottom: 30px;
    }

    /* Simple Container */
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 25px !important;
        border: 1px solid #D1FAE5 !important; /* Light green border */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }

    /* Simple Button */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #059669 !important; /* Emerald-600 */
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 2px 4px rgba(5, 150, 105, 0.2) !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #047857 !important; /* Emerald-700 */
        box-shadow: 0 4px 6px rgba(5, 150, 105, 0.3) !important;
    }

    /* Chart Cards */
    .stPlotlyChart {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #E5E7EB;
    }

    /* Static Custom KPI Cards */
    .kpi-container {
        display: flex;
        flex-direction: column;
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #10B981; /* Green accent bar on left */
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        text-align: left;
        height: 100%;
    }
    .kpi-icon {
        font-size: 1.8rem;
        margin-bottom: 10px;
        display: inline-block;
    }
    .kpi-title {
        font-size: 0.9rem;
        color: #4B5563;
        font-weight: 600;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #111827;
        margin: 5px 0 5px 0;
    }
    .kpi-value.gradient-text {
        color: #059669; /* Green text instead of gradient */
    }
    .kpi-desc {
        font-size: 0.85rem;
        color: #6B7280;
    }

    /* Static Insight Cards */
    .insight-card {
        border-radius: 12px;
        padding: 25px;
        color: #1F2937;
        background-color: #ECFDF5; /* Very light emerald */
        border: 1px solid #A7F3D0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .insight-card .icon { 
        font-size: 2.5rem; 
        margin-bottom: 15px;
    }
    .insight-card .title {
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #065F46; /* Dark green text */
        margin-bottom: 8px;
    }
    .insight-card .text {
        font-size: 1rem;
        line-height: 1.4;
    }
    
    /* Remove the individual colored backgrounds for simplicity */
    .card-purple, .card-blue, .card-pink, .card-teal { background: none; }

    /* Input text formatting */
    .stSelectbox label, .stSlider label, .stNumberInput label, .stDateInput label {
        font-weight: 600 !important;
        color: #374151 !important;
        font-size: 0.95rem !important;
    }
    
    /* Add Visible Borders to Inputs */
    .stSelectbox > div[data-baseweb="select"],
    .stDateInput > div > div[data-baseweb="input"],
    .stNumberInput > div {
        border: 1px solid #10B981 !important; /* Emerald green border */
        border-radius: 6px !important;
        background-color: #F8FAFC !important;
        transition: border-color 0.2s;
        overflow: hidden; /* Ensures child elements don't bleed out */
    }
    
    .stSelectbox > div[data-baseweb="select"]:hover,
    .stDateInput > div > div[data-baseweb="input"]:hover,
    .stNumberInput > div:hover {
        border-color: #059669 !important; /* Darker green on hover */
    }
    
    /* Ensure internal inputs within number_input and date_input don't have their own conflicting borders */
    .stNumberInput input, .stDateInput input {
        border: none !important;
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data at startup
@st.cache_data
def load_and_preprocess_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_store_inventory.csv')
    df = load_data(data_path)
    if df is not None:
        return clean_data(df)
    return None

def main():
    # --- HEADER SECTION ---
    st.markdown('<div class="dashboard-title">Retail Sales Forecast Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">✦ AI-Powered Retail Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)
    
    # Load historical data
    df_hist = load_and_preprocess_data()
    
    # Check if model exists
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, 'sales_model.pkl')
    
    if not os.path.exists(model_path):
        st.error("⚠️ Model not found! Please run `src/train_model.py` first to train and save the model.")
        return

    # Load model
    model, label_encoders = load_trained_model()
    if model is None:
        st.error("Failed to load the model.")
        return
        
    # Prepare dynamic options from data
    if df_hist is not None:
        store_ids = sorted(df_hist['Store ID'].unique().tolist())
        product_ids = sorted(df_hist['Product ID'].unique().tolist())
        categories = sorted(df_hist['Category'].unique().tolist())
        regions = sorted(df_hist['Region'].unique().tolist())
        weather_conds = sorted(df_hist['Weather Condition'].unique().tolist())
    else:
        store_ids = ['S001', 'S002']
        product_ids = ['P0001', 'P0002']
        categories = ['Electronics', 'Clothing', 'Furniture', 'Toys']
        regions = ['North', 'South', 'East', 'West']
        weather_conds = ['Sunny', 'Rainy', 'Cloudy', 'Snowy']

    # --- CONTROL PANEL ---
    st.markdown("### 🎛️ Forecast Control Panel")
    with st.form("prediction_main_form"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            store_id = st.selectbox("🏬 Store ID", store_ids)
            product_id = st.selectbox("📦 Product ID", product_ids)
            category = st.selectbox("🏷️ Category", categories)
            
        with col2:
            region = st.selectbox("🌍 Region", regions)
            price = st.slider("💵 Price ($)", min_value=1.0, max_value=500.0, value=50.0, step=0.5)
            discount = st.slider("🎉 Discount (%)", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
            
        with col3:
            inventory = st.number_input("📥 Current Inventory", min_value=0, value=150)
            weather = st.selectbox("🌤️ Weather", weather_conds)
            holiday = st.checkbox("🎊 Holiday / Promotion")
            
        with col4:
            target_date = st.date_input("📅 Target Date", datetime.date.today())
            st.write("")
            st.write("") # Spacing
            submit_button = st.form_submit_button(label="✨ Generate Forecast", use_container_width=True)

        # Calculate derived inputs
        season = "Summer"
        comp_price = price * 0.98
        units_ord = 50
        demand_forecast = 100
        holiday_val = 1 if holiday else 0

    st.write("")
    st.write("")

    # Calculate baseline metrics for KPIs
    avg_sales_all = df_hist['Units Sold'].mean() if df_hist is not None else 0
    total_sales_all = df_hist['Units Sold'].sum() if df_hist is not None else 0

    # Execute Prediction
    if submit_button:
        input_data = pd.DataFrame([{
            'Date': pd.to_datetime(target_date),
            'Store ID': store_id,
            'Product ID': product_id,
            'Category': category,
            'Region': region,
            'Inventory Level': inventory,
            'Units Ordered': units_ord,
            'Demand Forecast': demand_forecast,
            'Price': price,
            'Discount': discount,
            'Weather Condition': weather,
            'Holiday/Promotion': holiday_val,
            'Competitor Pricing': comp_price,
            'Seasonality': season
        }])
        
        with st.spinner("✨ Running AI inference..."):
            try:
                prediction = predict_sales(input_data, model, label_encoders)
                predicted_units = max(0, int(round(prediction[0])))
                prediction_success = True
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                prediction_success = False
                predicted_units = 0
    else:
        predicted_units = 0
        prediction_success = False

    # --- KPI SECTION ---
    price_impact = f"+{(discount * 1.5):.1f}%" if discount > 0 else "0%"
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-icon">🔮</div>
            <div class="kpi-title">Predicted Sales</div>
            <div class="kpi-value gradient-text">{predicted_units if prediction_success else '--'}</div>
            <div class="kpi-desc">Forecasted Units</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-icon">📊</div>
            <div class="kpi-title">Average Sales</div>
            <div class="kpi-value">{avg_sales_all:,.0f}</div>
            <div class="kpi-desc">Historical Benchmark</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        inv_warn = "color: #EF4444;" if (prediction_success and predicted_units > inventory) else "color: #10B981;"
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-icon">📦</div>
            <div class="kpi-title">Inventory Level</div>
            <div class="kpi-value" style="{inv_warn}">{inventory:,.0f}</div>
            <div class="kpi-desc">Current Stock</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-icon">🏷️</div>
            <div class="kpi-title">Price Impact</div>
            <div class="kpi-value">{price_impact}</div>
            <div class="kpi-desc">Discount Velocity Boost</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # --- FORECAST VISUALIZATION SECTION ---
    if prediction_success and df_hist is not None:
        
        filtered_hist = df_hist[(df_hist['Category'] == category) & (df_hist['Region'] == region)].copy()
        if not filtered_hist.empty:
            daily_hist = filtered_hist.groupby('Date')['Units Sold'].sum().reset_index().tail(14) # last 14 days
            daily_hist['Type'] = 'Actual'
            
            # Append predicted value to the chart
            pred_row = pd.DataFrame({
                'Date': [pd.to_datetime(target_date)],
                'Units Sold': [predicted_units],
                'Type': ['Predicted']
            })
            daily_combo = pd.concat([daily_hist, pred_row], ignore_index=True)
            
            # Convert Date to string so Plotly treats it as a categorical axis, 
            # preventing huge empty gaps when target_date is far in the future
            daily_combo['Date_str'] = daily_combo['Date'].dt.strftime('%b %d, %Y')
            
            # Display recent actuals and the future forecast as separate, clear visual elements
            fig_col1, fig_col2 = st.columns([1.5, 1])
            
            with fig_col1:
                # Historical trend + new forecast
                fig_hist = px.bar(daily_combo, x='Date_str', y='Units Sold', color='Type',
                                  title=f"14-Day Trend & Forecast ({category} - {region})",
                                  labels={'Date_str': 'Date'},
                                  color_discrete_map={'Actual': '#10B981', 'Predicted': '#3B82F6'})
                fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10, l=10, r=10), font=dict(family="Inter"))
                fig_hist.update_xaxes(type='category') # Enforce categorical axis
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with fig_col2:
                # The isolated future forecast as a gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = predicted_units,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"<span style='font-size:1.2em; font-family:Inter; font-weight:700;'>Future Forecast</span><br><span style='font-size:0.8em;color:gray'>{target_date.strftime('%b %d, %Y')}</span>"},
                    delta = {'reference': daily_hist['Units Sold'].mean(), 'position': "bottom", 'relative': False},
                    gauge = {
                        'axis': {'range': [None, max(predicted_units * 1.5, daily_hist['Units Sold'].max() * 1.2)]},
                        'bar': {'color': "#059669"},
                        'steps': [
                            {'range': [0, daily_hist['Units Sold'].mean()], 'color': "#F0FDF4"}
                        ],
                        'threshold': {
                            'line': {'color': "#047857", 'width': 3},
                            'thickness': 0.75,
                            'value': inventory
                        }
                    }
                ))
                fig_gauge.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

            # --- BUSINESS INSIGHTS CARDS ---
            st.write("")
            ins1, ins2, ins3, ins4 = st.columns(4)
            
            recent_avg = daily_hist['Units Sold'].mean()
            demand_trend = "Surging" if predicted_units > recent_avg * 1.1 else "Stable" if predicted_units > recent_avg * 0.9 else "Declining"
            trend_icon = "🚀" if demand_trend == "Surging" else "📉"
            best_region = df_hist.groupby('Region')['Units Sold'].sum().idxmax()
            rec_inv = int(predicted_units * 1.2)
            
            with ins1:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="icon">{trend_icon}</div>
                    <div class="title">Demand Trend</div>
                    <div class="text">Local demand is <b>{demand_trend}</b> compared to the 14-day average.</div>
                </div>
                """, unsafe_allow_html=True)
                
            with ins2:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="icon">💸</div>
                    <div class="title">Discount Impact</div>
                    <div class="text">Your {discount}% discount strategy is yielding a {price_impact} artificial boost in forecasted sales.</div>
                </div>
                """, unsafe_allow_html=True)
                
            with ins3:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="icon">🌍</div>
                    <div class="title">Top Market</div>
                    <div class="text">The <b>{best_region}</b> region continues to be the highest revenue generating market.</div>
                </div>
                """, unsafe_allow_html=True)
                
            with ins4:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="icon">📦</div>
                    <div class="title">Target Inventory</div>
                    <div class="text">To safely meet the forecasted demand, hold at least <b>{rec_inv}</b> units in stock.</div>
                </div>
                """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # --- CHARTS SECTION ---
    if df_hist is not None:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            trend_df = df_hist.groupby('Date')['Units Sold'].sum().reset_index().tail(30)
            fig_trend = px.area(trend_df, x='Date', y='Units Sold', title="Overall Sales Trend (30 Days)")
            fig_trend.update_traces(line_color="#10B981", fillcolor="rgba(16, 185, 129, 0.2)")
            fig_trend.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20, l=20, r=20), font=dict(family="Inter"))
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with chart_col2:
            scatter_df = df_hist.sample(min(1000, len(df_hist)))
            fig_scatter = px.scatter(scatter_df, x='Price', y='Units Sold', color='Category', opacity=0.7, title="Price Elasticity Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20, l=20, r=20), font=dict(family="Inter"))
            st.plotly_chart(fig_scatter, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            reg_df = df_hist.groupby('Region')['Units Sold'].sum().reset_index()
            fig_reg = px.bar(reg_df, x='Region', y='Units Sold', color='Region', text_auto='.2s', title="Geographic Sales Distribution", color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_reg.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20, l=20, r=20), font=dict(family="Inter"))
            st.plotly_chart(fig_reg, use_container_width=True)
            
        with chart_col4:
            seas_df = df_hist.groupby('Seasonality')['Units Sold'].mean().reset_index()
            fig_seas = px.line(seas_df, x='Seasonality', y='Units Sold', markers=True, title="Seasonality Impact Curve")
            fig_seas.update_traces(line=dict(width=5, color="#059669", shape="spline"), marker=dict(size=12, color="#047857", line=dict(width=2, color="white")))
            fig_seas.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20, l=20, r=20), font=dict(family="Inter"))
            st.plotly_chart(fig_seas, use_container_width=True)

if __name__ == "__main__":
    main()
