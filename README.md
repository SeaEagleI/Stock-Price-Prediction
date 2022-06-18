# 机器学习股价预测（使用多种模型实现）

## 文件说明
- forecast.py 数据处理、可视化、模型调用
- myutils.py 数据处理工具
- models.py 模型定义
- AMZN.csv 时间序列预测使用的数据集，为2010.01-2022.06的Amazon 股价变化情况，来自[Kaggle](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks?select=sp500_stocks.csv)
- backtest.py 回测
- draw_line.py 回测结果展示，backtest.py中自动调用
- 各月股票队列.xlsx 回测时使用的股票队列，2020.04-2020.12，不知道哪里搞来的
- 个股持有情况分析.xlsx 回测结果文件
- 持有期收益率情况图.html 回测结果图，需要pyecharts打开
- 每日资金情况图.html 回测结果图，需要pyecharts打开

## 环境配置
需要python3.8，其他依赖使用以下命令安装：
`pip install -r requirements.txt`

## 运行说明
- 时间序列预测  
运行命令```python forecast.py --model $modelname```  
运行时将`$modelname`替换为下面9种模型名称之一：
  1. `linearRegression`
  2. `DeterministProcess`
  3. `RelativeStrengthIndex`
  4. `ARIMA`
  5. `DecisionTree`
  6. `KNN`
  7. `LSTM`
  8. `Prophet`
  9. `SVM`

- 回测  
运行命令```python backtest.py```

## 参考项目

### 纯机器学习预测（不涉及因子建模）
- https://www.kaggle.com/code/nedahs/apple-stock-time-series-ml-models/notebook
- https://github.com/LightingFx/hs300_stock_predict
- https://github.com/moyuweiqing/A-stock-prediction-algorithm-based-on-machine-learning

### 多因子分析及预测（已放弃）
- https://github.com/phonegapX/alphasickle
- https://github.com/JoshuaWu1997/EMD-ALSTM-Multi-Factor-Stock-Profit-Prediction
