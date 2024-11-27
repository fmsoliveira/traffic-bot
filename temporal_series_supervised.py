# Exponential smoothing For time-series prediction (e.g., predicting congestion_level based on the timestamp)

from statsmodels.tsa.statespace.sarimax import SARIMAX

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# load data +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
df = pd.read_csv('dados_demanda_energia.csv', parse_dates=['Data'], index_col='Data')

# check data
print(df.head())
df.plot(figsize=(14, 6), title='Demanda de Energia')
plt.show()

# SARIMA  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# adjust the model
modelo_sarima = SARIMAX(df['Demanda'], order=(1, 1, 1), seasonal_order=(1, 1, 1,12)).fit()

previsao_sarima = modelo_sarima.forecast(steps=12)
# Visualizar previsão
df['Demanda'].plot(label='Observado')
previsao_sarima.plot(label='Previsão (SARIMA)', legend=True)
plt.show()



# Model comparison +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
mae_sarima = mean_absolute_error(df['Demanda'][-12:], previsao_sarima)
rmse_sarima = mean_squared_error(df['Demanda'][-12:], previsao_sarima, squared=False)
