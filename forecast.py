import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Example data (replace with actual data)
data = {
    'Date': pd.date_range(start='2020-01-01', periods=24, freq='M'),
    'Sales': [210, 220, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310,
              320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430]
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Visualize the data
df.plot(figsize=(10, 5))
plt.title('Historical Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Fit the ARIMA model
model = ARIMA(df['Sales'], order=(1, 1, 1))  # You might need to adjust the order
model_fit = model.fit()

# Forecast future sales (e.g., next 12 months)
forecast = model_fit.forecast(steps=12)
forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=12, freq='M')
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(df['Sales'], label='Historical Sales')
plt.plot(forecast_series, label='Forecasted Sales', color='red')
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
