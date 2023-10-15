import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the sunspot activity data
data = pd.read_csv('sunspot_activity.csv')

# Split the data into training and test sets
X_train = data['SunSpot Activity'][:int(0.8 * len(data))]
y_train = data['SunSpot Activity'][:int(0.8 * len(data))]
X_test = data['SunSpot Activity'][int(0.8 * len(data)):]
y_test = data['SunSpot Activity'][int(0.8 * len(data)):]

# Create a SARIMA model
model = SARIMAX(X_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# Train the model
model_fit = model.fit()

# Make predictions on the test set
y_pred = model_fit.predict(start=int(0.8 * len(data)), end=len(data))

# Evaluate the model's performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

# Plot the actual and predicted sunspot activity
import matplotlib.pyplot as plt

plt.plot(X_train, label='Training Data')
plt.plot(y_pred, label='Predicted Data')
plt.legend()
plt.show()
