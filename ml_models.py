import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class AirQualityModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def prepare_data(self, df):
        # Feature engineering
        df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
        df['Month'] = pd.to_datetime(df['DateTime']).dt.month
        df['DayOfWeek'] = pd.to_datetime(df['DateTime']).dt.dayofweek
        
        # Select features
        features = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'Hour', 'Month', 'DayOfWeek']
        target = 'AQI'
        
        return df[features], df[target]
    
    def train(self, df):
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_train_scaled)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.predict(X_test_scaled)
        anomaly_ratio = (anomalies == -1).sum() / len(anomalies)
        
        return {
            'mse': mse,
            'r2': r2,
            'test_predictions': y_pred,
            'test_actual': y_test,
            'anomaly_ratio': anomaly_ratio,
            'anomalies': anomalies == -1
        }
    
    def predict(self, df):
        # Prepare data
        X, _ = self.prepare_data(df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.predict(X_scaled)
        
        return predictions, anomalies == -1
    
    def save_model(self, path):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'anomaly_detector': self.anomaly_detector
        }, path)
    
    def load_model(self, path):
        saved_model = joblib.load(path)
        self.model = saved_model['model']
        self.scaler = saved_model['scaler']
        self.anomaly_detector = saved_model['anomaly_detector']

def perform_advanced_analysis(df):
    """Perform advanced statistical analysis on the data"""
    from scipy import stats
    import statsmodels.api as sm
    
    results = {}
    
    # Seasonal Decomposition
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    decomposition = sm.tsa.seasonal_decompose(df['AQI'], period=24)
    
    # Statistical Tests
    # 1. Test for normality
    _, p_value_normal = stats.normaltest(df['AQI'])
    results['normality_test'] = {
        'p_value': p_value_normal,
        'is_normal': p_value_normal > 0.05
    }
    
    # 2. Correlation Analysis
    pollutants = ['CO(GT)', 'NOx(GT)', 'NO2(GT)']
    correlation_matrix = df[pollutants].corr()
    results['correlations'] = correlation_matrix
    
    # 3. Granger Causality Test
    granger_results = {}
    for pollutant in pollutants:
        granger = sm.tsa.stattools.grangercausalitytests(
            df[['AQI', pollutant]], maxlag=24, verbose=False
        )
        granger_results[pollutant] = {
            'min_p_value': min([granger[i][0]['ssr_chi2test'][1] for i in range(1, 25)])
        }
    results['granger_causality'] = granger_results
    
    # 4. Outlier Detection using Z-score
    z_scores = np.abs(stats.zscore(df['AQI']))
    results['outliers'] = {
        'count': (z_scores > 3).sum(),
        'percentage': (z_scores > 3).mean() * 100
    }
    
    # Reset index for further use
    df.reset_index(inplace=True)
    
    return results, decomposition

if __name__ == "__main__":
    # Test the model
    df = pd.read_csv('airquality.csv')
    model = AirQualityModel()
    results = model.train(df)
    print("Model Performance:")
    print(f"MSE: {results['mse']:.2f}")
    print(f"R2 Score: {results['r2']:.2f}")
    print(f"Anomaly Ratio: {results['anomaly_ratio']:.2%}")
    
    # Save the model
    model.save_model('air_quality_model.joblib') 