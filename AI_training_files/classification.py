import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_model_classification():

    # read data file
    df = pd.read_csv('./geospatial_data.csv')

    # solve the encoding problem of the categorical variables
    encoder = LabelEncoder()
    df['vehicle_type_encoded'] = encoder.fit_transform(df['vehicle_type'])

    # change the timestamp data, which is a string and needs to be numeric values
    df['timestamp_converted'] = pd.to_datetime(df['timestamp'])

    # Extract time-based features
    df['hour'] = pd.to_datetime(df['timestamp_converted']).dt.hour
    df['day'] = pd.to_datetime(df['timestamp_converted']).dt.day
    df['month'] = pd.to_datetime(df['timestamp_converted']).dt.month


    # Define bins and labels for classification
    #bins = [0, 50, 100, 150, 200]  # Define thresholds
    #labels = ['Good', 'Moderate', 'Unhealthy', 'Very Unhealthy']  # Class labels
    #df['air_quality_class'] = pd.cut(df['air_quality_index'], bins=bins, labels=labels)

    bins = [0 - 1e-5, 0.5, 0.80, 0.90, 1 + 0.5, 4]  # Slightly expand bin edges
    labels = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
    df['accident_risk_class'] = pd.cut(df['accident_risk'], bins=bins, labels=labels)

    #X = df.drop(columns=['latitude', 'longitude', 'air_quality_index', 'air_quality_class','favorite_color', 'ice_cream_preference', 'vehicle_type', 'timestamp', 'timestamp_converted'])
    X = df[['congestion_level', 'traffic_density', 'population_density', 'building_density']]
    y = df['accident_risk_class']

    # get train set - 40% of the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4,random_state=42)

    # split the other 60% of the data evenly into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,random_state=42)

    # Convert to DataFrame if needed (only for easier inspection)
    X_train_df = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
    y_train_df = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train

    # Check for missing values in `X_train`
    print("Missing values in X_train:")
    print(X_train_df.isnull().sum())

    # Check for missing values in `y_train`
    print("Missing values in y_train:")
    print(y_train_df.isnull().sum())

    # Logistic Regression ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    lr = LogisticRegression(max_iter=5000)
    lr.fit(X_train, y_train)
    y_val_pred_lr = lr.predict(X_val)
    acc_lr = accuracy_score(y_val, y_val_pred_lr)
    print("Logistic Regression accuracy: ", acc_lr)

    # KNN ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X_train, y_train)
    y_val_pred_knn = knn.predict(X_val)
    acc_knn = accuracy_score(y_val, y_val_pred_knn)
    print("KNN accuracy score: ", acc_knn)

    # SVC ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    svc = SVC()
    svc.fit(X_train, y_train)
    y_val_pred_svc = svc.predict(X_val)
    acc_svc = accuracy_score(y_val, y_val_pred_svc)
    print("SVC accuracy: " , acc_svc)

    # accuracy
    model_performance = {'KNN': acc_knn,'Logistic Regression': acc_lr, 'SVC': acc_svc}
    best_model_name = max(model_performance, key=model_performance.get)
    print(f"Melhor modelo baseado na acurácia de validação: {best_model_name}")

    # save model
    if best_model_name == 'KNN':
        best_model = knn
    elif best_model_name == 'Logistic Regression':
        best_model = lr
    else:
        best_model = svc
        
    print("Best Classification Model: ", best_model)
    joblib.dump(best_model, f"{best_model_name}_classification.pkl")



