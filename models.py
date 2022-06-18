import matplotlib.pyplot as plt
import pandas as pd
from fbprophet import Prophet
from keras.layers import LSTM, Dense
from keras.models import Sequential
from pmdarima import auto_arima
from sklearn import neighbors, preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

from myutils import *

plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)


def linear_regression(X, y, dp=False):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        shuffle=False)

    # Fit and predict
    model = LinearRegression(
    )  # `fit_intercept=True` since we didn't use DeterministicProcess
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_train), index=y_train.index)
    y_fore = pd.Series(model.predict(X_test), index=y_test.index)
    print(f'Model train accuracy: {model.score(X_train, y_train)*100:.3f}%')
    print(f'Model test accuracy: {model.score(X_test, y_test)*100:.3f}%')
    print(f'Model train MAE: {mae(y_pred,y_train):.3f}')
    print(f'Model train RMSE: {mse(y_pred,y_train, squared=False):.3f}')
    print(f'Model test MAE: {mae(y_fore,y_test):.3f}')
    print(f'Model test RMSE: {mse(y_fore,y_test, squared=False):.3f}')
    plt.clf()
    y_train.plot(**plot_params)
    y_test.plot(**plot_params)
    y_pred.plot()
    y_fore.plot()
    if dp:
        plt.savefig('images/deterministicProcess.png')
    else:
        plt.savefig('images/linearRegression.png')

    return model


def deterministic_process(data):
    fourier = CalendarFourier(freq="A", order=3)
    dp = DeterministicProcess(
        index=data.index,
        constant=True,  # dummy feature for bias (y-intercept)
        order=1,  # trend (order 1 means linear)
        #seasonal=True,              # seasonality (indicators).
        additional_terms=[fourier],  # annual seasonality (fourier)
        drop=True,  # drop terms to avoid collinearity
    )

    X_t_s = dp.in_sample()
    X_c = make_lags(data["Close"], lags=1)
    X_c = X_c.fillna(0.0)
    X = pd.concat([X_t_s, X_c], axis=1)
    y = data["Close"].copy()
    linear_regression(X, y, True)

    # trend
    y = data[["Close"]].copy()
    X = dp.in_sample()  # features for the training data
    idx_train, idx_test = train_test_split(y.index,
                                           test_size=0.2,
                                           shuffle=False)
    X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    # Fit trend model
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    # Make predictions
    y_fit = pd.DataFrame(
        model.predict(X_train),
        index=y_train.index,
        columns=y_train.columns,
    )
    y_pred = pd.DataFrame(
        model.predict(X_test),
        index=y_test.index,
        columns=y_test.columns,
    )

    # Plot
    axs = y_train.plot(color='0.25', subplots=True, sharex=True)
    axs = y_test.plot(color='0.25', subplots=True, sharex=True, ax=axs)
    axs = y_fit.plot(color='C0', subplots=True, sharex=True, ax=axs)
    axs = y_pred.plot(color='C3', subplots=True, sharex=True, ax=axs)
    for ax in axs:
        ax.legend([])
    _ = plt.suptitle("Trends")
    plt.savefig('images/trend.png')


def relative_strength(data, day):
    dataset = data['Close']
    RSI_set = []
    # 计算RSI值
    for i in range(0, len(data) - day):
        RSI = 0.0
        bigger_set = 0
        smaller_set = 0
        for j in range(0, 13):
            if dataset[i + j + 1] > dataset[i + j]:
                bigger_set += dataset[i + j + 1] - dataset[i + j]
            else:
                smaller_set += dataset[i + j] - dataset[i + j + 1]
        RSI = bigger_set / (bigger_set + smaller_set) * 100
        if i < 5:
            print(bigger_set)
            print(smaller_set)
            print(RSI)
        RSI_set.append(RSI)

    # 定义RSI表格
    dic = {
        '超买市场（RSI>=80）且实际下跌': 0,
        '超买市场（RSI>=80）但实际上涨': 0,
        '强势市场（50<=RSI<80）且实际下跌': 0,
        '强势市场（50<=RSI<80）但实际上涨': 0,
        '弱式市场（50>RSI>=20）且实际上涨': 0,
        '弱式市场（50>RSI>=20）但实际下跌': 0,
        '超卖市场（RSI<20）且实际上涨': 0,
        '超卖市场（RSI<20）但实际下跌': 0
    }

    for i in range(0, len(data) - 15):
        if (RSI_set[i] >= 80) & (dataset[i + 15] >= dataset[i + 14]):
            dic['超买市场（RSI>=80）但实际上涨'] += 1
        elif (RSI_set[i] >= 80) & (dataset[i + 15] < dataset[i + 14]):
            dic['超买市场（RSI>=80）且实际下跌'] += 1
        elif (RSI_set[i] < 80) & (RSI_set[i] >= 50) & (dataset[i + 15] >=
                                                       dataset[i + 14]):
            dic['强势市场（50<=RSI<80）但实际上涨'] += 1
        elif (RSI_set[i] < 80) & (RSI_set[i] >= 50) & (dataset[i + 15] <
                                                       dataset[i + 14]):
            dic['强势市场（50<=RSI<80）且实际下跌'] += 1
        elif (RSI_set[i] < 50) & (RSI_set[i] >= 20) & (dataset[i + 15] >=
                                                       dataset[i + 14]):
            dic['弱式市场（50>RSI>=20）且实际上涨'] += 1
        elif (RSI_set[i] < 50) & (RSI_set[i] >= 20) & (dataset[i + 15] <
                                                       dataset[i + 14]):
            dic['弱式市场（50>RSI>=20）但实际下跌'] += 1
        elif (RSI_set[i] < 20) & (dataset[i + 15] >= dataset[i + 14]):
            dic['超卖市场（RSI<20）且实际上涨'] += 1
        else:
            dic['超卖市场（RSI<20）但实际下跌'] += 1
    print(dic)


class AutoARIMA_pridict:
    stock_code = ''
    tsData = pd.DataFrame()

    def __init__(self, data):
        self.new_data = data

    def makePrediction(self, node):  # node为节点天数，在这之前为训练集、之后为测试集，
        new_data = self.new_data
        # 训练集和预测集
        train = new_data[:node]
        valid = new_data[node:]
        # 对收盘价进行测试
        training = train['Close']
        validation = valid['Close']
        # 拟合模型
        model = auto_arima(training,
                           start_p=1,
                           start_q=1,
                           max_p=6,
                           max_q=6,
                           m=12,
                           start_P=0,
                           seasonal=True,
                           d=1,
                           D=1,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True)  #
        model.fit(training)
        # 进行预测
        forecast = model.predict(n_periods=len(valid))
        forecast = pd.DataFrame(forecast,
                                index=valid.index,
                                columns=['Prediction'])
        y_fore = forecast['Prediction']
        y_test = valid['Close']
        print(f'Model test MAE: {mae(y_fore,y_test):.3f}')
        print(f'Model test RMSE: {mse(y_fore,y_test, squared=False):.3f}')
        # 画图
        valid['Predictions'] = forecast['Prediction']
        plt.clf()
        plt.plot(train['Close'], label='train')
        plt.plot(valid[['Close', 'Predictions']],
                 label=['ground truth', 'predictions'])
        plt.legend()
        plt.savefig('images/arima.png')
        plt.show()


class DT_predict:
    stock_code = ''
    tsData = pd.DataFrame()

    def __init__(self, data):
        self.tsData = data

    def make_predict(self, node):
        self.tsData['(t+1)-(t)'] = self.tsData['Close'].shift(
            1) - self.tsData['Close']
        self.tsData['label'] = 0
        # 构建对应表
        for i in range(1, len(self.tsData)):
            if self.tsData['(t+1)-(t)'][i] > 0:
                self.tsData['label'][i] = 1
            else:
                self.tsData['label'][i] = 0

        # 构建数据集
        test_data = self.tsData[:len(self.tsData) - node]
        train_data = self.tsData[len(self.tsData) - node:]
        train_X = train_data.iloc[:, 1:4].values
        train_y = train_data['label'].values
        test_X = test_data.iloc[:, 1:4].values
        test_y = test_data['label'].values

        # 进行预测
        clf = DecisionTreeClassifier(criterion='gini',
                                     max_depth=3,
                                     min_samples_leaf=6)
        clf.fit(train_X, train_y)
        y_pred = clf.predict(test_X)

        print('train accuracy: %f' %
              (accuracy_score(train_y, clf.predict(train_X))))
        print('test accuracy: %f' % (accuracy_score(test_y, y_pred)))
        print('roc: %f' % roc_auc_score(test_y, y_pred))  # 召回率


scaler = MinMaxScaler(feature_range=(0, 1))


class kNN_pridict:

    def __init__(self, data):
        self.new_data = data

    def makePrediction(self, node):  # node为节点天数，在这之前为训练集、之后为测试集
        new_data = self.new_data
        new_data = new_data.drop('Name', axis=1)
        # 训练集和预测集
        train = new_data[:node]
        valid = new_data[node:]
        x_train = train.drop('Close', axis=1)
        y_train = train['Close']
        x_valid = valid.drop('Close', axis=1)
        y_valid = valid['Close']
        # 缩放数据
        x_train_scaled = scaler.fit_transform(x_train)
        x_train = pd.DataFrame(x_train_scaled)
        x_valid_scaled = scaler.transform(x_valid)
        x_valid = pd.DataFrame(x_valid_scaled)
        # 使用gridsearch查找最佳参数
        params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
        knn = neighbors.KNeighborsRegressor()
        model = GridSearchCV(knn, params, cv=5)
        # 拟合模型并进行预测
        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        # 画图
        valid['Predictions'] = 0
        valid['Predictions'] = preds
        y_fore = valid['Predictions']
        y_test = valid['Close']
        print(f'Model test MAE: {mae(y_fore,y_test):.3f}')
        print(f'Model test RMSE: {mse(y_fore,y_test, squared=False):.3f}')
        plt.clf()
        plt.plot(valid[['Close', 'Predictions']],
                 label=['ground truth', 'prediction'])
        plt.plot(train['Close'], label='train')
        plt.legend()
        plt.savefig('images/knn.png')
        plt.show()


class LSTM_Predict:

    def __init__(self, data):
        self.new_data = data

    def makePrediction(self, node):  # node为节点天数，在这之前为训练集、之后为测试集
        new_data = self.new_data.loc[:, ['Close']]
        # 创建训练集和验证集
        dataset = new_data.values
        train = dataset[0:node, :]
        valid = dataset[node:, :]

        # 将数据集转换为x_train和y_train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        x_train, y_train = [], []
        for i in range(60, len(train)):
            x_train.append(scaled_data[i - 60:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # 创建和拟合LSTM网络
        model = Sequential()
        model.add(
            LSTM(units=50,
                 return_sequences=True,
                 input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

        # 使用过去值来预测
        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)
        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        # 作图
        train = new_data[:node]
        valid = new_data[node:]
        print('valid长度是：' + str(len(valid)))
        print(len(closing_price))
        valid['Predictions'] = closing_price
        y_fore = valid['Predictions']
        y_test = valid['Close']
        print(f'Model test MAE: {mae(y_fore,y_test):.3f}')
        print(f'Model test RMSE: {mse(y_fore,y_test, squared=False):.3f}')
        plt.clf()
        plt.plot(train['Close'], label='train')
        plt.plot(valid['Close'], label='ground truth')
        plt.plot(valid['Predictions'], label='predictions')
        plt.legend()
        plt.savefig('images/LSTM.png')
        plt.show()


class Prophet_Predict:

    def __init__(self, data):
        self.new_data = data

    def makePrediction(self, node):  # node为节点天数，在这之前为训练集、之后为测试集
        new_data = self.new_data.loc[:, ['Close']]
        new_data['Date'] = new_data.index
        new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
        forecast_valid = []
        # 训练集和预测集
        train = new_data[:node]
        valid = new_data[node:]
        # 拟合模型
        model = Prophet()
        model.fit(train)
        # 预测
        close_prices = model.make_future_dataframe(periods=len(valid))
        forecast = model.predict(close_prices)
        forecast_valid = forecast['yhat'][node:]
        # 画图
        valid['Predictions'] = forecast_valid.values
        y_fore = valid['Predictions']
        y_test = valid['y']
        print(f'Model test MAE: {mae(y_fore,y_test):.3f}')
        print(f'Model test RMSE: {mse(y_fore,y_test, squared=False):.3f}')
        plt.clf()
        fig = model.plot(forecast, xlabel='Date', ylabel='Close')
        plt.title("Amazon Stock Price Forecast", fontsize=16)
        plt.savefig('images/prophet1.png')
        plt.clf()
        model.plot_components(forecast)
        plt.savefig('images/prophet2.png')



class SVM_Predict:

    def __init__(self, data):
        self.data = data

    def makeSVMPrediction(self, rate):  # rate表示训练集和测试集的比例
        df_CB = self.data.drop('Name', axis=1)
        # value表示涨跌, =1为涨，=0为跌
        value = pd.Series(df_CB['Close'] - df_CB['Close'].shift(1), \
                          index=df_CB.index)
        value = value.bfill()
        value[value >= 0] = 1
        value[value < 0] = 0
        df_CB['Value'] = value
        # 后向填充空缺值
        df_CB = df_CB.fillna(method='bfill')
        df_CB = df_CB.astype('float64')

        L = len(df_CB)
        train = int(L * rate)
        total_predict_data = L - train

        # 对样本特征进行归一化处理
        df_CB_X = df_CB.drop(['Value'], axis=1)
        df_CB_X = preprocessing.scale(df_CB_X)

        # 开始循环预测，每次向前预测一个值
        correct = 0
        train_original = train
        while train < L:
            Data_train = df_CB_X[train - train_original:train]
            value_train = value[train - train_original:train]
            Data_predict = df_CB_X[train:train + 1]
            value_real = value[train:train + 1]

            # 核函数分别选取'ploy','linear','rbf'
            # classifier = svm.SVC(C=1.0, kernel='poly')
            # classifier = svm.SVC(kernel='linear')
            classifier = svm.SVC(C=1.0, kernel='rbf')
            classifier.fit(Data_train, value_train)
            value_predict = classifier.predict(Data_predict)
            #print("value_real=%d value_predict=%d" % (value_real[0], value_predict))
            # 计算测试集中的正确率
            if (value_real[0] == int(value_predict)):
                correct = correct + 1
            train = train + 1

        correct = correct * 100 / total_predict_data
        print("accuracy=%.2f%%" % correct)
