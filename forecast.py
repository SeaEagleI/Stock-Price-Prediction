import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pylab import *  # 改变plot字体，适应中文
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from models import *
from myutils import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

parser = argparse.ArgumentParser(description='Time Series Forecasting')
parser.add_argument('--model',
                    default='linearRegression',
                    choices=[
                        'linearRegression', 'DeterministProcess',
                        'RelativeStrengthIndex', 'ARIMA', 'DecisionTree',
                        'KNN', 'LSTM', 'Prophet', 'SVM'
                    ])
args = parser.parse_args()

data_file = 'AMZN.csv'
if not os.path.exists(data_file):
    download_data()
if not os.path.exists('images'):
    os.mkdir('images')

data = pd.read_csv(data_file, index_col="Date", parse_dates=["Date"])
# data.plot(subplots=True, figsize=(10, 15))
# plt.savefig('images/data.png')

# # 按月频整合数据
# plt.clf()
# data["Close"].asfreq('M').interpolate().plot(legend=True)
# shifted = data["Close"].asfreq('M').interpolate().shift(10).plot(legend=True)
# plt.legend(['Close', 'Close_lagged'])
# plt.title(
#     'Closing price of Amazon over time (Monthly frequency) - raw and lagged')
# plt.savefig('images/data_by_month.png')
# plt.show()

# # Moving average plot to see the trend
# data["Close"].plot()
# moving_average = data["Close"].rolling(
#     window=360,  # 180-day window
#     center=True,  # puts the average at the center of the window
#     min_periods=180,  # choose about half the window size
# ).mean()  # compute the mean (could also do median, std, min, max, ...)
# moving_average.plot()
# plt.legend(["Close", "Moving average"])
# plt.savefig('images/moving_average.png')
# plt.show()

# # autocorrelation and partial autocorrelation
# plot_acf(data["Close"],
#          lags=20,
#          title="Autocorrelation chart: Amazon (Close price)")
# plt.savefig('images/autocorrelation.png')
# plt.show()
# _ = plot_pacf(
#     data["Close"],
#     lags=20,
#     title=
#     "Partial autocorrelation of Amazon (Close price) with 95% confidence intervals of no correlation."
# )
# plt.savefig('images/partial_auto.png')
# plt.show()

# # lag plot
# _ = plot_lags(data["Close"], lags=20, nrows=3)

# # Period of the series
# decomposed_oracle_close = seasonal_decompose(data["Close"], period=180)
# decomposed_oracle_close.plot()
# plt.savefig('images/decompose.png')
# plt.show()
# plot_periodogram(data.Close)

# # seasonal
# X = data.copy()
# # days within a week
# X["dayofweek"] = X.index.dayofweek  # the x-axis (freq)
# X["week"] = X.index.week  # the seasonal period (period)

# # days within a month
# X["dayofmonth"] = X.index.day
# X["month"] = X.index.month

# # days within a year
# X["dayofyear"] = X.index.dayofyear
# X["year"] = X.index.year

# fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(11, 6))
# seasonal_plot(X, y="Close", period="week", freq="dayofweek", ax=ax0)
# seasonal_plot(X, y="Close", period="month", freq="dayofmonth", ax=ax1)
# seasonal_plot(X, y="Close", period="year", freq="dayofyear", ax=ax2)
# plt.savefig('images/seasonal.png')

if args.model == 'linearRegression':
    # use lags as feature
    X = make_lags(data["Close"], lags=1)
    X = X.fillna(0.0)
    y = data["Close"].copy()
    linear_regression(X, y)

elif args.model == 'DeterministProcess':
    deterministic_process(data)

elif args.model == 'RelativeStrengthIndex':
    relative_strength(data, 14)

elif args.model == 'ARIMA':
    a = AutoARIMA_pridict(data)
    a.makePrediction(2500)

elif args.model == 'DecisionTree':
    a = DT_predict(data)
    a.make_predict(2500)

elif args.model == 'KNN':
    a = kNN_pridict(data[:1000])
    a.makePrediction(700)

elif args.model == 'LSTM':
    a = LSTM_Predict(data)
    a.makePrediction(2500)

elif args.model == 'Prophet':
    a = Prophet_Predict(data[-1000:])
    a.makePrediction(750)

elif args.model == 'SVM':
    a = SVM_Predict(data)
    a.makeSVMPrediction(0.8)
