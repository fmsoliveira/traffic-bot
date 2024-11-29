import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

def train_model_supervised_learning():
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
    
    
    X = df.drop(columns=['accident_risk','favorite_color', 'ice_cream_preference', 'vehicle_type', 'timestamp_converted', 'timestamp'])

    y = df['accident_risk']
 

    # get train set - 40% of the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4,random_state=42)

    # split the other 60% of the data evenly into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,random_state=42)

    # Multiple Linear Regression ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    lr_multiple = LinearRegression()

    # train the model with all x (independent variables)
    lr_multiple.fit(X_train, y_train)

    # predict with validation set
    y_val_pred_multiple = lr_multiple.predict(X_val)

    # get metrics for validation group
    r2_multiple = r2_score(y_val, y_val_pred_multiple)
    mae_multiple = mean_absolute_error(y_val, y_val_pred_multiple)

    model_performance = {'Regressao_Linear_Multipla_val' : (lr_multiple, r2_multiple, mae_multiple)}

    # predict with test set
    y_test_pred_multiple = lr_multiple.predict(X_test)

    # get metrics fro testing group
    r2_multiple_test = r2_score(y_test, y_test_pred_multiple)
    mae_multiple_test = mean_absolute_error(y_test, y_test_pred_multiple)

    model_performance['Regressao_Linear_Multipla_test'] = (lr_multiple, r2_multiple_test, mae_multiple_test)


    # Lasso Regression ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # regressão lasso
    lasso = Lasso(alpha=0.1)

    # treinar o modelo
    lasso.fit(X_train, y_train)

    # comparar com o grupo de validação
    y_val_pred_lasso = lasso.predict(X_val)

    #get metrics for validation group
    r2_lasso = r2_score(y_val, y_val_pred_lasso)
    mae_lasso = mean_absolute_error(y_val, y_val_pred_lasso)

    model_performance['Regressao_Lasso_val'] = (lasso, r2_lasso, mae_lasso)

    # predict using test data set
    y_test_pred_lasso = lasso.predict(X_test)

    # get metrics for test group
    r2_lasso_test = r2_score(y_test, y_test_pred_lasso)
    mae_lasso_test = mean_absolute_error(y_test, y_test_pred_lasso)

    model_performance['Regressao_Lasso_test'] = (lasso, r2_lasso_test, mae_lasso_test)

    # Ridge Regression ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ## Ridge regression
    ridge = Ridge(alpha=0.1)

    # training the model
    ridge.fit(X_train, y_train)

    # predict using validation set
    y_val_pred_ridge = ridge.predict(X_val)

    # metrics from validation set
    r2_ridge_val = r2_score(y_val, y_val_pred_ridge)
    mae_ridge_val = mean_absolute_error(y_val, y_val_pred_ridge)


    model_performance['Regressao_Ridge_val'] = (ridge, r2_ridge_val, mae_ridge_val)

    # predict using test set
    y_test_pred_ridge = ridge.predict(X_val)

    # metric from test set
    r2_ridge_test = r2_score(y_test, y_test_pred_ridge)
    mae_ridge_test = mean_absolute_error(y_test, y_test_pred_ridge)

    model_performance['Regressao_Ridge_test'] = (ridge, r2_ridge_test, mae_ridge_test)


    # Compare models performance
    for item in model_performance:
        value = model_performance[item]
        print(item, value)

    # select best model - greater r2
    # Selecionar o melhor modelo com base no R² de validação
    melhor_nome, (melhor_modelo, melhor_r2, melhor_mae) = max(model_performance.items(), key=lambda x: x[1][1])
    print(f"Melhor modelo baseado no R² de validação: {melhor_nome}")

    return (melhor_nome, melhor_modelo)