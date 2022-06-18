# 机器学习股价预测（使用多种模型实现）

## 文件说明
forecast.py 数据处理、可视化、模型调用
myutils.py 数据处理工具
models.py 模型定义
AMZN.csv 时间序列预测使用的数据集，2010.01-2022.06 Amazon 股价变化情况，来自https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks?select=sp500_stocks.csv
backtest.py 回测
draw_line.py 回测结果展示，backtest.py中自动调用
各月股票队列.xlsx 回测时使用的股票队列，2020.04-2020.12，不知道哪里搞来的
个股持有情况分析.xlsx 回测结果文件
持有期收益率情况图.html 回测结果图，需要pyecharts打开
每日资金情况图.html 回测结果图，需要pyecharts打开

## 环境配置
需要python3.8，其他依赖使用以下命令安装：
`pip install -r requirements.txt`

## 运行
1. 时间序列预测
`python forecast.py --model modelname`
其中`modelname`可选：`{linearRegression,DeterministProcess,RelativeStrengthIndex,ARIMA,DecisionTree,KNN,LSTM,Prophet,SVM}`

2. 回测
`python backtest.py`

## 参考项目
### 机器学习预测（不涉及因子）
- https://www.kaggle.com/code/nedahs/apple-stock-time-series-ml-models/notebook
- https://github.com/LightingFx/hs300_stock_predict
- https://github.com/moyuweiqing/A-stock-prediction-algorithm-based-on-machine-learning

### 多因子分析及预测
- https://github.com/phonegapX/alphasickle (x)
- https://github.com/JoshuaWu1997/EMD-ALSTM-Multi-Factor-Stock-Profit-Prediction (x)
