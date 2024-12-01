# Importing necessary libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def train_model_ensemble_methods():
    # read data
    df = pd.read_csv("geospatial_data.csv")

    # solve the encoding problem of the categorical variables
    encoder = LabelEncoder()
    df['vehicle_type_encoded'] = encoder.fit_transform(df['vehicle_type'])

    # change the timestamp data, which is a string and needs to be numeric values
    df['timestamp_converted'] = pd.to_datetime(df['timestamp'])

    # Extract time-based features
    df['hour'] = pd.to_datetime(df['timestamp_converted']).dt.hour
    df['day'] = pd.to_datetime(df['timestamp_converted']).dt.day
    df['month'] = pd.to_datetime(df['timestamp_converted']).dt.month

    # Let's drop any columns that are not useful for the model (you can adjust this based on your dataset)
    # Columns remaining in df: latitude, longitude, air_quality_index, temperature, humidity, traffic_density, average_speed, population_density
    # building_density, accident_risk, congestion_level, vehicle_type_encoded, hour, day, month
    df = df.drop(columns=['favorite_color', 'ice_cream_preference', 'vehicle_type', 'timestamp', 'timestamp_converted'])

    bins = [0 - 1e-5, 0.5, 0.80, 0.90, 1 + 1e-5]  # Slightly expand bin edges
    labels = ['Low', 'Moderate', 'High', 'Very High']
    df['accident_risk_class'] = pd.cut(df['accident_risk'], bins=bins, labels=labels)


    # Split data into features (X) and target (y)
    X = df.drop(columns=['accident_risk_class', 'accident_risk'])  # Features (independent variables)
    y = df['accident_risk_class']  # Target variable (dependent)

    # Optional: Encode the target variable if it's categorical
    y = y.astype('category').cat.codes  # Convert categories to numeric labels if necessary

    # get train set - 40% of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8,random_state=42)

    # split the other 60% of the data evenly into validation and test sets
    #X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,random_state=42)

    # Random Forest +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Initialize the Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model on the training data
    rf_model.fit(X_train, y_train)


    # Avaliar o modelo
    y_pred_rf = rf_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_rf)
    print(f'Accuracy: {accuracy:.2f}')

    # Print a classification report (precision, recall, F1-score)
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf))

    auc = roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr')
    print("AUC (Random Forest):", auc)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    ensemble_results = {"Random Forest": {"accuracy": accuracy, "AUC": auc}}

    # Decision Tree +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Treinar o modelo de Árvore de Decisão
    modelo_decision_tree = DecisionTreeClassifier(random_state=42)
    modelo_decision_tree.fit(X_train, y_train)
    # Avaliar o modelo
    y_pred_dt = modelo_decision_tree.predict(X_test)

    decision_tree_accuracy = accuracy_score(y_test, y_pred_dt)
    print("Acurácia (Árvore de Decisão):", decision_tree_accuracy)

    decision_tree_auc = roc_auc_score(y_test, modelo_decision_tree.predict_proba(X_test), multi_class='ovr')
    print("AUC (Árvore de Decisão):", decision_tree_auc)

    ensemble_results["Decision tree"] = {"accuracy": decision_tree_accuracy, "auc": decision_tree_auc}


    # select best model
    best_accuracy = 0
    best_model_name = ""
    best_model = None
    for model in ensemble_results:
        model_results = ensemble_results[model]
        model_accuracy = model_results.get('accuracy')
        if model_accuracy > best_accuracy:
            best_accuracy = model_accuracy
            best_model_name = model




    print("Best model: ", best_model_name, "best accuracy:", best_accuracy)

    if best_model_name == "Random Forest":
        best_model = rf_model
    else:
        best_model = modelo_decision_tree

    # save the model to pickle file
    joblib.dump(best_model, f"{best_model_name}_ensemble_methods.pkl")

    


