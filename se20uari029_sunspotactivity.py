%matplotlib inline
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [16, 9]
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from timeit import default_timer as timers

df = pd.read_csv('/content/sunspot_data.csv', delimiter=',', na_values=['-1'])
df.dataframeName = 'sunspot_data.csv'
del(df['Unnamed: 0'])
df.columns = ['year', 'month', 'day', 'fraction','sunspots', 'sdt', 'obs','indicator']
df.head(-5)


df['time']=df[['year', 'month', 'day']].apply(lambda s: pd.datetime(*s),axis = 1)
df.index = df['time']
df['sunspots'].interpolate(method='linear', inplace=True)
ts = pd.Series(data=df.sunspots, index=df.index)
#ts = ts['1900-01-01':]
ts_month = ts.resample('MS').mean()
ts_quarter = ts.resample('Q').mean()
ts_quarter.plot()
plt.show()
plot_pacf(ts_quarter,lags=100,title='Sunspots')
plt.show()
plot_acf(ts_quarter,lags=100,title='Sunspots')
plt.show()
from statsmodels.tsa.stattools import adfuller
def printADFTest(serie):
    result = adfuller(serie, autolag='AIC')
    print("ADF Statistic %F" % (result[0]))
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')
    print('\n')
#d = 0
printADFTest(ts_quarter)
#d = 1
#printADFTest(ts_quarter.diff(1).dropna())
model = sm.tsa.statespace.SARIMAX(ts_quarter, trend='n', order=(3,0,10), seasonal_order=(1,1,0,43))
results = model.fit()
print(results.summary())
forecast = results.predict(start = ts_quarter.index[-2], end= ts_quarter.index[-2] + pd.DateOffset(months=240), dynamic= True)
ts_quarter.plot()
forecast.plot()
plt.show()
