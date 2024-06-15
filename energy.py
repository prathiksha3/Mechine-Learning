import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Example data (replace with actual data)
data = {
    'Date': pd.date_range(start='2022-01-01', periods=24, freq='M'),
    'Energy_Consumption': [300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450, 475,
                           490, 505, 520, 535, 550, 575, 590, 605, 620, 635, 650, 675]
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Visualize the data
df.plot(figsize=(10, 5))
plt.title('Historical Energy Consumption')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.show()

# Fit the ARIMA model
model = ARIMA(df['Energy_Consumption'], order=(1, 1, 1))  # You might need to adjust the order
model_fit = model.fit()

# Forecast future energy consumption (e.g., next 12 months)
forecast = model_fit.forecast(steps=12)
forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=12, freq='M')
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(df['Energy_Consumption'], label='Historical Energy Consumption')
plt.plot(forecast_series, label='Forecasted Energy Consumption', color='red')
plt.title('Energy Consumption Forecast')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.legend()
plt.show()
