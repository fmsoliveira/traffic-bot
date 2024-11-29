from sklearn.ensemble import IsolationForest

import pandas as pd

def train_model_time_series():
    df = pd.read_csv('./geospatial_data.csv')

    # Prepare data for anomaly detection
    data = df[['air_quality_index', 'traffic_density', 'temperature', 'humidity']]

    # Fit Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(data)

    # Anomalies will have label -1
    anomalies = df[df['anomaly'] == -1]
    print("Detected anomalies:\n", anomalies)
