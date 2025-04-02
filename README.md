# 🌍 Air Quality Analysis Dashboard

A comprehensive dashboard for analyzing and visualizing air quality data using Python, Streamlit, and various data science libraries.

## 🎯 Project Objectives

1. **Real-time Air Quality Monitoring**
   - Track current AQI levels and pollutant concentrations
   - Monitor temperature and humidity correlations
   - Provide instant alerts for hazardous conditions
   - Display historical trends and patterns

2. **Predictive Analysis & Forecasting**
   - Develop ML models to predict future AQI values
   - Generate 24-hour air quality forecasts
   - Identify potential pollution hotspots
   - Enable proactive decision-making

3. **Health Impact Assessment**
   - Evaluate health risks based on AQI levels
   - Provide safety guidelines and recommendations
   - Track health impact distribution
   - Generate health advisories for different groups

4. **Environmental Pattern Analysis**
   - Analyze weather-pollutant relationships
   - Identify seasonal patterns and trends
   - Study correlation between different pollutants
   - Detect anomalies and unusual patterns

5. **Data-Driven Decision Support**
   - Provide comprehensive statistical analysis
   - Enable data exploration and visualization
   - Support policy-making with evidence
   - Facilitate research and studies

## 📋 Project Overview

This dashboard combines real-time monitoring, machine learning predictions, and advanced statistical analysis to provide insights into air quality patterns and trends. It enables data-driven decision making and facilitates health impact assessment.

## 🎯 Project Goals

1. Provide real-time monitoring of air quality parameters
2. Enable predictive analysis using machine learning
3. Offer comprehensive data visualization
4. Support data-driven decision making
5. Facilitate health impact assessment

## 🚀 Features

### 1. Real-time Monitoring
- Current AQI levels
- Pollutant concentrations (CO, NOx, NO2)
- Temperature and humidity correlations
- Historical trends

### 2. Interactive Visualizations
- Line charts for pollutant trends
- Scatter plots for temperature vs AQI
- Pie charts for pollutant distributions
- Correlation heatmaps
- Distribution plots
- Time-based analysis

### 3. Machine Learning Predictions
- Random Forest model for AQI prediction
- Feature importance analysis
- Model performance metrics
- Anomaly detection

### 4. Advanced Analysis
- 24-hour forecasting using Prophet
- Health impact assessment
- Weather correlation analysis
- Statistical analysis
- Seasonal decomposition
- Granger causality tests
- Outlier detection

### 5. Data Management
- Raw data exploration
- Statistical summaries
- Data quality reports
- Download options (CSV/Excel)

## 🛠️ Technical Architecture

### Core Components

1. **Data Processing Layer**
   - Pandas for data manipulation
   - NumPy for numerical operations
   - Data cleaning and preprocessing

2. **Visualization Layer**
   - Streamlit for web interface
   - Plotly for interactive charts
   - Matplotlib/Seaborn for statistical plots

3. **Machine Learning Layer**
   - Scikit-learn for ML models
   - Prophet for time series forecasting
   - Statsmodels for statistical analysis

### Technologies Used

- Python 3.12
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Prophet
- Statsmodels

## 📊 Dashboard Sections

1. **Overview**
   - Real-time metrics
   - Basic visualizations
   - Daily patterns

2. **ML Predictions**
   - Model performance
   - Predictions vs Actual
   - Feature importance

3. **Forecasting**
   - 24-hour AQI forecast
   - Confidence intervals
   - Trend analysis

4. **Health Impact**
   - Health risk assessment
   - Impact distribution
   - Safety guidelines

5. **Weather Correlation**
   - Weather-pollutant relationships
   - Temperature impact
   - Humidity effects

6. **Advanced Analysis**
   - Statistical tests
   - Seasonal patterns
   - Outlier detection

7. **Raw Data & Statistics**
   - Detailed statistics
   - Data quality report
   - Column analysis

## 🔧 Technical Details

### Data Processing
- DateTime conversion
- AQI calculation
- Feature engineering
- Data normalization

### Machine Learning Models
1. **Random Forest Regressor**
   - Features: CO, NOx, NO2, T, RH, Hour, Month, DayOfWeek
   - Target: AQI
   - Performance metrics: R², MSE, Anomaly Ratio

2. **Prophet Model**
   - Time series forecasting
   - Seasonal decomposition
   - Trend analysis

### Statistical Analysis
- Normality tests
- Correlation analysis
- Seasonal decomposition
- Granger causality
- Outlier detection

## 📈 Performance Metrics

### Model Performance
- R² Score: Model fit quality
- MSE: Prediction accuracy
- Anomaly Ratio: Unusual patterns

### Data Quality Metrics
- Missing values
- Duplicate entries
- Data completeness
- Memory usage

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Suhani2305/AirIndexAnalysis.git
   cd AirIndexAnalysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

## 📝 Project Structure

```
air-quality-dashboard/
├── app.py                 # Main dashboard application
├── ml_models.py          # Machine learning models
├── requirements.txt      # Project dependencies
├── airquality.csv        # Air quality dataset
└── README.md            # Project documentation
```

## 🔍 Data Sources

### Input Data
- Carbon Monoxide (CO)
- Nitrogen Oxides (NOx)
- Nitrogen Dioxide (NO2)
- Temperature
- Relative Humidity
- Time-based features

### Data Format
```python
{
    'DateTime': datetime,
    'CO(GT)': float,
    'NOx(GT)': float,
    'NO2(GT)': float,
    'T': float,
    'RH': float,
    'AQI': float
}
```

## 🚀 Future Enhancements

1. **Real-time Integration**
   - API integration
   - Live data streaming
   - Automated updates

2. **Advanced Features**
   - Multiple location support
   - Custom ML models
   - Advanced forecasting

3. **User Experience**
   - Mobile optimization
   - Custom themes
   - Export options

## 🔒 Security Considerations

- Data validation
- Input sanitization
- Error handling
- Resource management

## 📊 Performance Optimization

- Data caching
- Lazy loading
- Memory management
- Query optimization

## 🤝 Contributing Guidelines

1. Fork repository
2. Create feature branch
3. Submit pull request
4. Follow code style

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- All contributors and users of the dashboard 