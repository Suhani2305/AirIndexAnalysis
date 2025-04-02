# 🌍 Air Quality Analysis Dashboard

A comprehensive dashboard for analyzing and visualizing air quality data using Python, Streamlit, and various data science libraries.

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

## 🛠️ Technologies Used

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

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/air-quality-dashboard.git
   cd air-quality-dashboard
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
├── README.md            # Project documentation
└── DOCUMENTATION.md     # Detailed documentation
```

## 🔍 Data Sources

The dashboard uses air quality data including:
- Carbon Monoxide (CO)
- Nitrogen Oxides (NOx)
- Nitrogen Dioxide (NO2)
- Temperature
- Relative Humidity
- Time-based features

## 📈 Future Enhancements

1. Real-time data integration
2. Multiple location support
3. Advanced ML models
4. Mobile app version
5. API integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- All contributors and users of the dashboard 