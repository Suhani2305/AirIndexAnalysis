"""
✅ COMPLETED TASKS:
1. Data Cleaning & Preprocessing:
   - DateTime conversion
   - AQI calculation
   - Data structuring

2. Visualizations:
   - Line charts for pollution trends
   - Scatter plots for temperature vs AQI
   - Pie charts for pollutant distributions
   - Correlation heatmaps
   - Distribution plots

3. EDA & Statistical Analysis:
   - Time-based analysis
   - Correlation analysis
   - Distribution analysis
   - Daily patterns

❌ PENDING TASKS:
1. Machine Learning:
   - Implement prediction models
   - Add anomaly detection
   
2. Additional Analysis:
   - Seasonal trends
   - Advanced statistical tests
   - Outlier detection and handling

3. Documentation:
   - Add detailed methodology
   - Include statistical summaries
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import numpy as np
from ml_models import AirQualityModel, perform_advanced_analysis
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from prophet import Prophet

# Configure plotly to remove the logo and add download options
config = {
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'air_quality_chart',
        'height': 500,
        'width': 700,
        'scale': 2
    }
}

# ✅ COMPLETED: Basic Setup and Styling
# Set Seaborn style
sns.set_style("darkgrid")
plt.style.use("dark_background")

# Set page config
st.set_page_config(
    page_title="Air Quality Analysis",
    # page_icon="🌍",
    layout="wide"

)

# ✅ COMPLETED: Data Loading and Preprocessing
@st.cache_data
def load_data():
    # Load data
    df = pd.read_csv('airquality.csv')
    
    # ✅ COMPLETED: DateTime Processing
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Hour'] = df['DateTime'].dt.hour
    
    # ✅ COMPLETED: AQI Calculation
    df['AQI'] = (
        df['CO(GT)'] * 10 +
        df['NOx(GT)'] +
        df['NO2(GT)']
    ) / 3
    
    return df

# Load data first
df = load_data()

# Theme toggle and settings in sidebar
with st.sidebar:
    st.title("Dashboard Settings")
    
    # Theme toggle
    theme = st.selectbox(
        "Choose Theme",
        ["Dark", "Light"],
        help="Select dashboard theme"
    )
    
    # Apply theme
    if theme == "Light":
        plt.style.use("default")
        st.markdown("""
            <style>
            .main {
                background-color: #FFFFFF;
                color: #000000;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        plt.style.use("dark_background")
        st.markdown("""
            <style>
            .main {
                background-color: #1E3D59;
                color: white;
            }
            
            /* Data table styling */
            .stDataFrame {
                width: 100%;
            }
            
            /* Making sure the table uses full width */
            [data-testid="stDataFrame"] > div {
                width: 100%;
                max-width: 100%;
            }
            
            /* Table header styling */
            .dataframe thead th {
                background-color: #1E3D59 !important;
                color: white !important;
                font-weight: bold !important;
                text-align: center !important;
            }
            
            /* Table cell styling */
            .dataframe tbody td {
                text-align: center !important;
                background-color: #0E1117 !important;
                color: white !important;
            }
            
            /* Hover effect on table rows */
            .dataframe tr:hover td {
                background-color: #2E3D59 !important;
            }
            
            /* Remove table borders */
            .dataframe {
                border: none !important;
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
                height: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: #1E3D59;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #3E5D79;
                border-radius: 5px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #4E6D89;
            }
            </style>
        """, unsafe_allow_html=True)
    
    # Add interactive filters
    st.subheader("Data Filters")
    
    # Date range selector
    st.write("Select Date Range")
    min_date = df['DateTime'].min().date()
    max_date = df['DateTime'].max().date()
    
    start_date = st.date_input('Start Date', min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input('End Date', max_date, min_value=min_date, max_value=max_date)
    
    # Pollutant selection
    st.write("Select Pollutants to Display")
    selected_pollutants = st.multiselect(
        "Choose pollutants",
        ['CO(GT)', 'NOx(GT)', 'NO2(GT)'],
        default=['CO(GT)', 'NOx(GT)', 'NO2(GT)'],
        help="Select which pollutants to show in the charts"
    )
    
    # Location filter (if available)
    if 'Location' in df.columns:
        selected_location = st.selectbox(
            "Select Location",
            df['Location'].unique(),
            help="Choose the monitoring station location"
        )
    
    # Download options
    st.subheader("Download Options")
    
    def get_csv_download_link(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="air_quality_data.csv">Download CSV File</a>'
        return href
    
    st.markdown(get_csv_download_link(df), unsafe_allow_html=True)
    
    # Add tooltips
    st.markdown("""
        <style>
        div[data-baseweb="tooltip"] {
            background-color: #1E3D59;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

# Filter data based on selections
mask = (df['DateTime'].dt.date >= start_date) & (df['DateTime'].dt.date <= end_date)
filtered_df = df[mask]

# ✅ COMPLETED: Dashboard Title and Metrics
# st.title("  Air Quality Analysis  ")
st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin: 1rem 0;'>
        <h2 style='color: #1f77b4; margin-bottom: 0.5rem;'>Real-time Air Quality Monitoring and Analysis</h2>
        <p style='color: #2c3e50; margin-bottom: 0;'>Comprehensive dashboard for analyzing and visualizing air quality data</p>
    </div>
""", unsafe_allow_html=True)

# ✅ COMPLETED: Real-time Metrics Display
col1, col2, col3 = st.columns(3)

with col1:
    current_aqi = filtered_df['AQI'].iloc[-1]
    avg_aqi = filtered_df['AQI'].mean()
    delta = ((current_aqi - avg_aqi) / avg_aqi) * 100
    st.metric(
        "Current AQI", 
        f"{current_aqi:.1f}", 
        f"{delta:.1f}% vs avg",
        help="Air Quality Index - Composite measure of air pollution"
    )

with col2:
    co_level = filtered_df['CO(GT)'].iloc[-1]
    avg_co = filtered_df['CO(GT)'].mean()
    delta_co = ((co_level - avg_co) / avg_co) * 100
    st.metric(
        "CO Level (GT)", 
        f"{co_level:.1f}", 
        f"{delta_co:.1f}% vs avg",
        help="Carbon Monoxide Level in Ground Truth measurements"
    )

with col3:
    nox_level = filtered_df['NOx(GT)'].iloc[-1]
    avg_nox = filtered_df['NOx(GT)'].mean()
    delta_nox = ((nox_level - avg_nox) / avg_nox) * 100
    st.metric(
        "NOx Level (GT)", 
        f"{nox_level:.1f}", 
        f"{delta_nox:.1f}% vs avg",
        help="Nitrogen Oxides Level in Ground Truth measurements"
    )

# ✅ COMPLETED: Main Visualizations
# Create two columns for charts
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Pollutant Trends")
    with st.spinner('Loading pollutant trends...'):
        fig = px.line(
            filtered_df, 
            x='DateTime', 
            y=selected_pollutants,
            labels={'value': 'Pollutant Level', 'DateTime': 'Time'},
            title='Pollutant Levels Over Time'
        )
        fig.update_layout(
            hovermode='x unified',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True, config=config)

with right_column:
    st.subheader("Temperature vs Air Quality")
    with st.spinner('Loading temperature analysis...'):
        fig = px.scatter(
            filtered_df, 
            x='T', 
            y='AQI', 
            color='RH',
            title='Temperature vs AQI (Color: Relative Humidity)',
            labels={
                'T': 'Temperature (°C)', 
                'AQI': 'Air Quality Index', 
                'RH': 'Relative Humidity (%)'
            },
            hover_data=['DateTime']
        )
        st.plotly_chart(fig, use_container_width=True, config=config)

# ✅ COMPLETED: Pollutant Distribution Analysis
st.subheader("Pollutant Distribution Analysis")

# Create three columns for pollutant pie charts
col1, col2, col3 = st.columns(3)

with col1:
    # CO Distribution by Level
    co_levels = pd.cut(filtered_df['CO(GT)'], 
                      bins=[0, 2, 4, 6, 8, float('inf')],
                      labels=['Very Low (0-2)', 'Low (2-4)', 
                             'Moderate (4-6)', 'High (6-8)', 'Very High (8+)'])
    co_dist = co_levels.value_counts()
    
    fig_co = px.pie(
        values=co_dist.values,
        names=co_dist.index,
        title='CO Level Distribution',
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.Reds
    )
    fig_co.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_co, use_container_width=True, config=config)

with col2:
    # NOx Distribution by Level
    nox_levels = pd.cut(filtered_df['NOx(GT)'], 
                       bins=[0, 50, 100, 150, 200, float('inf')],
                       labels=['Very Low (0-50)', 'Low (50-100)', 
                              'Moderate (100-150)', 'High (150-200)', 'Very High (200+)'])
    nox_dist = nox_levels.value_counts()
    
    fig_nox = px.pie(
        values=nox_dist.values,
        names=nox_dist.index,
        title='NOx Level Distribution',
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.Blues
    )
    fig_nox.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_nox, use_container_width=True, config=config)

with col3:
    # NO2 Distribution by Level
    no2_levels = pd.cut(filtered_df['NO2(GT)'], 
                       bins=[0, 40, 80, 120, 160, float('inf')],
                       labels=['Very Low (0-40)', 'Low (40-80)', 
                              'Moderate (80-120)', 'High (120-160)', 'Very High (160+)'])
    no2_dist = no2_levels.value_counts()
    
    fig_no2 = px.pie(
        values=no2_dist.values,
        names=no2_dist.index,
        title='NO2 Level Distribution',
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.Greens
    )
    fig_no2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_no2, use_container_width=True, config=config)

# ✅ COMPLETED: Time and Temperature Analysis
col1, col2 = st.columns(2)

with col1:
    # Time of Day Distribution
    filtered_df['TimeOfDay'] = pd.cut(filtered_df['Hour'], 
                            bins=[0, 6, 12, 18, 24], 
                            labels=['Night (0-6)', 'Morning (6-12)', 
                                   'Afternoon (12-18)', 'Evening (18-24)'])
    
    time_distribution = filtered_df.groupby('TimeOfDay')['AQI'].mean()
    
    fig_time = px.pie(
        values=time_distribution.values,
        names=time_distribution.index,
        title='AQI Distribution by Time of Day',
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_time.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_time, use_container_width=True, config=config)

with col2:
    # Temperature Impact on Pollution
    temp_levels = pd.cut(filtered_df['T'], 
                        bins=[0, 10, 20, 30, float('inf')],
                        labels=['Cold (<10°C)', 'Mild (10-20°C)', 
                               'Warm (20-30°C)', 'Hot (>30°C)'])
    temp_aqi = filtered_df.groupby(temp_levels)['AQI'].mean()
    
    fig_temp = px.pie(
        values=temp_aqi.values,
        names=temp_aqi.index,
        title='AQI Distribution by Temperature Range',
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    fig_temp.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_temp, use_container_width=True, config=config)

# ✅ COMPLETED: Detailed Analysis Tabs
st.subheader("Detailed Analysis")
tabs = st.tabs([
    "Overview", 
    "ML Predictions",
    "Forecasting", 
    "Health Impact", 
    "Weather Correlation",
    "Advanced Analysis",
    "Raw Data & Statistics"
])

with tabs[0]:
    st.subheader("Daily Patterns")
    with st.spinner('Loading daily patterns...'):
        hourly_avg = filtered_df.groupby('Hour')[selected_pollutants + ['AQI']].mean()
        fig = px.line(
            hourly_avg, 
            labels={'value': 'Average Level', 'Hour': 'Hour of Day'},
            title='Average Daily Patterns'
        )
        st.plotly_chart(fig, use_container_width=True, config=config)

with tabs[1]:
    st.subheader("Machine Learning Predictions")
    
    # Initialize and train model
    model = AirQualityModel()
    
    # Train model if not already trained
    if 'ml_results' not in st.session_state:
        with st.spinner('Training ML model... This may take a moment.'):
            st.session_state.ml_results = model.train(filtered_df)
            model.save_model('air_quality_model.joblib')
    
    # Display model metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "R² Score", 
            f"{st.session_state.ml_results['r2']:.3f}",
            help="R-squared score (1 is perfect prediction)"
        )
    
    with col2:
        st.metric(
            "MSE", 
            f"{st.session_state.ml_results['mse']:.3f}",
            help="Mean Squared Error (lower is better)"
        )
    
    with col3:
        st.metric(
            "Anomaly Ratio", 
            f"{st.session_state.ml_results['anomaly_ratio']:.2%}",
            help="Percentage of data points detected as anomalies"
        )
    
    # Plot actual vs predicted values
    st.subheader("Actual vs Predicted AQI Values")
    with st.spinner('Generating prediction plot...'):
        fig = px.scatter(
            x=st.session_state.ml_results['test_actual'],
            y=st.session_state.ml_results['test_predictions'],
            labels={'x': 'Actual AQI', 'y': 'Predicted AQI'},
            title='Model Predictions vs Actual Values'
        )
        
        # Add perfect prediction line
        fig.add_trace(go.Scatter(
            x=[filtered_df['AQI'].min(), filtered_df['AQI'].max()],
            y=[filtered_df['AQI'].min(), filtered_df['AQI'].max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(
            xaxis_title="Actual AQI Values",
            yaxis_title="Model Predicted AQI Values",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Feature Importance Plot
    st.subheader("Feature Importance Analysis")
    with st.spinner('Calculating feature importance...'):
        if hasattr(model.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'Hour', 'Month', 'DayOfWeek'],
                'importance': model.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance in Predicting AQI'
            )
            
            fig.update_layout(
                xaxis_title="Relative Importance",
                yaxis_title="Features",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True, config=config)
        else:
            st.warning("Model needs to be trained first. Please wait for training to complete.")
    
    # Add explanation of the model at the end
    with st.expander("🤖 How does the ML model work?", expanded=True):
        st.markdown("""
        This machine learning model uses a **Random Forest Regressor** to predict Air Quality Index (AQI) values based on various features:
        
        ### 1. Input Features:
        - Pollutant levels (CO, NOx, NO2)
        - Temperature and Humidity
        - Time-based features (Hour, Month, Day of Week)
        
        ### 2. Model Components:
        - Random Forest Regressor for AQI prediction
        - Isolation Forest for anomaly detection
        - StandardScaler for feature normalization
        
        ### 3. Performance Metrics:
        - R² Score: How well the model fits the data (1 is perfect)
        - MSE: Average squared difference between predictions and actual values
        - Anomaly Ratio: Percentage of unusual patterns detected
        
        ### 4. Feature Importance:
        - Shows which factors most strongly influence AQI predictions
        - Higher importance means the feature has more impact on predictions
        
        ### 5. How to Use:
        1. The model automatically trains on your data
        2. View the prediction accuracy metrics
        3. Check the actual vs predicted plot
        4. Analyze feature importance to understand what drives air quality
        """)

with tabs[2]:
    st.subheader("24-Hour Forecast")
    with st.spinner('Generating forecast...'):
        # Prepare data for Prophet
        forecast_df = filtered_df[['DateTime', 'AQI']].rename(columns={
            'DateTime': 'ds',
            'AQI': 'y'
        })
        
        # Train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.fit(forecast_df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=24, freq='H')
        forecast = model.predict(future)
        
        # Plot forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df['DateTime'],
            y=filtered_df['AQI'],
            name='Historical AQI',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'].tail(24),
            y=forecast['yhat'].tail(24),
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'].tail(24),
            y=forecast['yhat_upper'].tail(24),
            fill=None,
            mode='lines',
            line=dict(color='rgba(255,0,0,0)'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'].tail(24),
            y=forecast['yhat_lower'].tail(24),
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(255,0,0,0)'),
            name='Confidence Interval'
        ))
        fig.update_layout(title='24-Hour AQI Forecast')
        st.plotly_chart(fig, use_container_width=True, config=config)

with tabs[3]:
    st.subheader("Health Impact Analysis")
    
    def get_health_impact(aqi):
        if aqi <= 50:
            return "Good", "No health impacts expected", "🟢"
        elif aqi <= 100:
            return "Moderate", "Unusually sensitive individuals should consider reducing prolonged outdoor exposure", "🟡"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups", "Active children and adults should limit prolonged outdoor exposure", "🟠"
        elif aqi <= 200:
            return "Unhealthy", "Everyone may begin to experience health effects", "🔴"
        elif aqi <= 300:
            return "Very Unhealthy", "Health warnings of emergency conditions", "🟣"
        else:
            return "Hazardous", "Health alert: everyone may experience serious health effects", "⚫"
    
    current_status, health_msg, color = get_health_impact(current_aqi)
    
    st.markdown(f"## Current Status: {color} {current_status}")
    st.info(health_msg)
    
    # Health impact distribution
    health_dist = filtered_df['AQI'].apply(lambda x: get_health_impact(x)[0])
    health_counts = health_dist.value_counts()
    
    # Create custom colors for health impact levels
    health_colors = {
        'Good': '#00E400',  # Green
        'Moderate': '#FFFF00',  # Yellow
        'Unhealthy for Sensitive Groups': '#FF7E00',  # Orange
        'Unhealthy': '#FF0000',  # Red
        'Very Unhealthy': '#8F3F97',  # Purple
        'Hazardous': '#7E0023'  # Maroon
    }
    
    fig = px.pie(
        values=health_counts.values,
        names=health_counts.index,
        title='Distribution of Air Quality Health Impact',
        color=health_counts.index,
        color_discrete_map=health_colors
    )
    st.plotly_chart(fig, use_container_width=True, config=config)

with tabs[4]:
    st.subheader("Weather Correlation Analysis")
    
    # Calculate correlations
    weather_cols = ['T', 'RH', 'AQI'] + selected_pollutants
    corr_matrix = filtered_df[weather_cols].corr()
    
    # Plot correlation heatmap
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        title="Weather-Pollutant Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Temperature impact analysis
    temp_ranges = pd.cut(
        filtered_df['T'],
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=['Cold (<10°C)', 'Mild (10-20°C)', 'Warm (20-30°C)', 'Hot (>30°C)']
    )
    
    temp_impact = filtered_df.groupby(temp_ranges)[['AQI'] + selected_pollutants].mean()
    
    fig = px.bar(
        temp_impact,
        title="Temperature Impact on Pollutants",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True, config=config)

with tabs[5]:
    st.subheader("Advanced Statistical Analysis")
    
    # Correlation Analysis using Seaborn
    st.subheader("Correlation Heatmap (Seaborn)")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = filtered_df[['AQI', 'CO(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.title('Correlation Matrix of Air Quality Parameters')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Distribution Analysis using Seaborn
    st.subheader("Distribution Analysis (Seaborn)")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(filtered_df['AQI'], kde=True, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title('AQI Distribution')
    sns.histplot(filtered_df['CO(GT)'], kde=True, ax=axes[0, 1], color='red')
    axes[0, 1].set_title('CO Distribution')
    sns.histplot(filtered_df['T'], kde=True, ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Temperature Distribution')
    sns.histplot(filtered_df['RH'], kde=True, ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Relative Humidity Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Perform advanced analysis if not already done
    if 'advanced_results' not in st.session_state:
        with st.spinner('Performing advanced analysis...'):
            st.session_state.advanced_results, st.session_state.decomposition = perform_advanced_analysis(filtered_df.copy())
    
    # Display normality test results
    st.write("### Normality Test Results")
    st.write(f"P-value: {st.session_state.advanced_results['normality_test']['p_value']:.4f}")
    st.write(f"Data is {'normally' if st.session_state.advanced_results['normality_test']['is_normal'] else 'not normally'} distributed")
    
    # Plot seasonal decomposition
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual')
    )
    
    fig.add_trace(go.Scatter(y=st.session_state.decomposition.observed, name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(y=st.session_state.decomposition.trend, name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(y=st.session_state.decomposition.seasonal, name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(y=st.session_state.decomposition.resid, name='Residual'), row=4, col=1)
    
    fig.update_layout(height=800, title_text="Seasonal Decomposition of AQI")
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Display Granger Causality results
    st.write("### Granger Causality Test Results")
    for pollutant, result in st.session_state.advanced_results['granger_causality'].items():
        st.write(f"{pollutant}: p-value = {result['min_p_value']:.4f}")
    
    # Display outlier analysis
    st.write("### Outlier Analysis")
    st.write(f"Number of outliers: {st.session_state.advanced_results['outliers']['count']}")
    st.write(f"Percentage of outliers: {st.session_state.advanced_results['outliers']['percentage']:.2f}%")

with tabs[6]:
    st.subheader("Raw Data & Statistics")
    
    # Summary Statistics Section
    with st.expander("📊 Summary Statistics", expanded=True):
        st.write("### Basic Statistics")
        stats_df = filtered_df.describe()
        
        # Format the statistics
        stats_df = stats_df.round(2)
        st.dataframe(stats_df, use_container_width=True)
        
        # Additional Statistics
        st.write("### Additional Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Missing Values**")
            missing_data = filtered_df.isnull().sum()
            st.dataframe(pd.DataFrame({
                'Column': missing_data.index,
                'Missing Values': missing_data.values,
                'Percentage': (missing_data.values / len(filtered_df) * 100).round(2)
            }))
        
        with col2:
            st.write("**Data Types**")
            dtypes = filtered_df.dtypes
            st.dataframe(pd.DataFrame({
                'Column': dtypes.index,
                'Data Type': dtypes.values
            }))
    
    # Data Quality Report Section
    with st.expander("📋 Data Quality Report", expanded=True):
        st.write("### Data Quality Analysis")
        
        # Calculate quality metrics
        total_rows = len(filtered_df)
        total_columns = len(filtered_df.columns)
        missing_values = filtered_df.isnull().sum().sum()
        duplicates = filtered_df.duplicated().sum()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", total_rows)
        
        with col2:
            st.metric("Total Columns", total_columns)
        
        with col3:
            st.metric("Missing Values", missing_values)
        
        with col4:
            st.metric("Duplicate Rows", duplicates)
        
        # Column-wise analysis
        st.write("### Column-wise Analysis")
        
        # Create column analysis DataFrame with consistent lengths
        column_analysis = pd.DataFrame({
            'Column': filtered_df.columns,
            'Data Type': filtered_df.dtypes,
            'Missing Values': filtered_df.isnull().sum(),
            'Missing %': (filtered_df.isnull().sum() / len(filtered_df) * 100).round(2),
            'Unique Values': filtered_df.nunique()
        })
        
        # Add memory usage separately to ensure consistent lengths
        memory_usage = filtered_df.memory_usage(deep=True)
        memory_usage_mb = memory_usage / 1024 / 1024
        column_analysis['Memory Usage (MB)'] = memory_usage_mb.round(2)
        
        st.dataframe(column_analysis, use_container_width=True)

# Footer with project information
st.markdown("---")
st.markdown("Dashboard created with Streamlit, Matplotlib, Seaborn, and Plotly - Analyzing Air Quality Data")

# Add loading animations and tooltips to all charts
st.markdown("""
    <style>
    /* Loading animation */
    .stSpinner {
        text-align: center;
        max-width: 50%;
        margin: 0 auto;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)

# Make all charts more interactive
for fig in [fig for fig in locals() if isinstance(fig, go.Figure)]:

    fig.update_layout(
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )

