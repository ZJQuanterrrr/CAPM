import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import statsmodels.api as sm
# import getstock
import os

apikey = "809LQIK40YTDAU6O"
apikey_zzl = "xxx"
apikey_st = "xxx"

plt.style.use('seaborn')

from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import io
import requests
import os

from fredapi import Fred

fred = Fred(api_key='d56b593ad51b80ca0ca6b4bad48afe78')

# Get S&P 500 monthly adjusted returns data
data1 = fred.get_series('SP500')
data1 = data1.resample('M').ffill().pct_change()
data1 = data1.loc['2015-01-31':'2019-12-31']

# Print the first 5 rows of the data
print(data1.head())


import time
def getMonthlyStockPrices(symbol, apikey):
    import time
    # if(symbol=='^GSPC'):
    #     path = '../Data/stocks_monthly/' + symbol + '_Stock.csv'
    #     print('LocalData:')
    #     with open(path, mode='r', encoding='utf-8') as f:
    #         df = pd.read_csv(f)
    #         NEW_timeStamp = pd.to_datetime(df['timestamp'])
    #         df.index = NEW_timeStamp
    #         df = df.drop(columns = {'timestamp'})
    #         return df
    time.sleep(3)
    
    ts = TimeSeries(key=apikey, output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
    symbol_df = pd.DataFrame.from_dict( data, orient = 'index' )
    symbol_df = symbol_df.apply(pd.to_numeric)
    symbol_df.index = pd.to_datetime( symbol_df.index )
    symbol_df.columns = [ 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amt']
    symbol_df = symbol_df.sort_index( ascending=True )
    return symbol_df

ts = TimeSeries( key=apikey )
data, meta_data = ts.get_monthly_adjusted( '^GSPC' )

import openpyxl
def readExcelFile(filename):
    # 打开工作表
    workbook = openpyxl.load_workbook(filename)
    # 用索引取第一个工作薄
    booksheet = workbook.active
    Stocks_Name = []
    # 返回的结果集
    for i in range(1,506):
        Stocks_Name.append(booksheet.cell(row=i+1, column=1).value)
        # print(i,": ",booksheet.row_values(i)[0])
    return Stocks_Name
Stocks_Name = readExcelFile("SP_Stocks.xlsx")

def Exist(symbol):
    path = './stocks505/' + symbol + '_monthly.csv'
    if os.path.exists(path):
        return True
    else:
        return False
    return False
def getStocksLocally(Stocks_name):
    Stocks = {}
    for stock_symbol in Stocks_name:
#         if(stock_symbol=="CARR" or stock_symbol=="OTIS" or stock_symbol=="AMCR" or stock_symbol=="CBRE"):
#             continue
        path = './stocks505/' + stock_symbol + '_monthly.csv'
        if(Exist(stock_symbol)):
            # print(stock_symbol, end=" ")
            with open(path, mode='r', encoding='utf-8') as f:
                per = pd.read_csv(f)
                NEW_timeStamp = pd.to_datetime(per['Unnamed: 0'])
                per.index = NEW_timeStamp
                per.index.name = "timestamp"
                per = per.drop(columns = {'Unnamed: 0'})
                per = per.loc['2015':'2019']
                if(len(per)!=60):
                    continue
                Stocks[stock_symbol] = per
    return Stocks
def getIndex():
    columns = [ 'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amt']
    # get the market index of ^GSPC
    GSPC_index = getMonthlyStockPrices('MMM', apikey)
    GSPC_index.columns = columns
    GSPC_index = GSPC_index.sort_values('timestamp', ascending=True).loc['2015':'2019']
    return GSPC_index
Stocks = getStocksLocally(Stocks_Name)
GSPC_index = getIndex()
# GSPC_index.index.name = None

tbill_data = pd.read_excel('tbilldata.xlsx', skiprows=1) # skiprows=1 跳过第一行
tbill_data.index = pd.to_datetime(tbill_data['DATE'])                            # DATE column 转化成 index
tbill_data = tbill_data.drop('DATE', axis=1)                                     # 再删去DATE这个column
_15_19_byYear = tbill_data.loc['2015':'2019']['BANK DISCOUNT.2']
Sum = []
for year in range(2015,2020):
    Sum.append((_15_19_byYear.loc[str(year)]/100).mean()) 
avg_annual = np.mean(Sum)
print("avg_annual", avg_annual)
Monthly_risk_free = avg_annual/12
print("Monthly_risk_free", Monthly_risk_free)

def getInterceptSlope(stock_ret, index_ret):
    model_data = pd.concat([stock_ret, index_ret], axis=1)
    model_data.columns = ['stock_ret', 'index_ret']
#     display(model_data.head())
    # fit the data with ols model
    results = smf.ols('stock_ret ~ index_ret', data=model_data).fit()
    intercept = results.params.Intercept
    slope = results.params.index_ret
    rsquared = results.rsquared
    return intercept, slope, rsquared
def getRiskfree_1_beta(Rf, slope_name, Slopes):
    return Rf*(1-Slopes[slope_name])
def getSingleAlpha(Intercepts,Rf_1_beta):
    return Intercepts-Rf_1_beta
def getAlphas(stocks, index, Rf):
    Intercepts = {}
    Slopes = {}
    rsquareds = {}
    index_ret = index
    for stock in stocks:
        stock_ret = ((stocks[stock].close.diff() + stocks[stock].dividend_amt.shift())/stocks[stock].close.shift()).bfill()
        I_S = getInterceptSlope(stock_ret, index_ret)
        Intercepts[stock] = I_S[0]
        Slopes[stock] = I_S[1]
        rsquareds[stock] = I_S[2]
    Rf_1_beta = {}
    for slope_name in Slopes:
        Rf_1_beta[slope_name] = getRiskfree_1_beta(Rf, slope_name, Slopes)
    Alphas = {}
    for name in Intercepts:
        Alphas[name] = getSingleAlpha(Intercepts[name],Rf_1_beta[name])
    return Alphas, rsquareds
Alphas, rsquareds = getAlphas(Stocks, data1, Monthly_risk_free)

max_alpha = -1
max_alpha_name = 0
min_alpha = 1
min_alpha_name = 0
for alpha in Alphas:
    if(Alphas[alpha]>max_alpha):
        max_alpha = Alphas[alpha]
        max_alpha_name = alpha
    if(Alphas[alpha]<min_alpha):
        min_alpha = Alphas[alpha]
        min_alpha_name = alpha
print("min_alpha:", min_alpha_name, ":", min_alpha, "annual excess return:", (1+min_alpha)**12-1)
print("max_alpha:", max_alpha_name, ":", max_alpha, "annual excess return:", (1+max_alpha)**12-1)

max_rsquared = -1
max_rsquared_name = 0
min_rsquared = 1
min_rsquared_name = 0
for rsquared in rsquareds:
    if(rsquareds[rsquared]>max_rsquared):
        max_rsquared = rsquareds[rsquared]
        max_rsquared_name = rsquared
    if(rsquareds[rsquared]<min_rsquared):
        min_rsquared = rsquareds[rsquared]
        min_rsquared_name = rsquared
print("min_rsquared:",min_rsquared_name, ":", min_rsquared)
print("max_rsquared:",max_rsquared_name, ":", max_rsquared)

fig1, ax = plt.subplots(2, 2, figsize=(14,14))
# index_ret = (GSPC_index.close - GSPC_index.open + GSPC_index.dividend_amt)/GSPC_index.open
index_ret = data1

# stock with max_alpha
stock_ret = ((Stocks[max_alpha_name].close.diff() + Stocks[max_alpha_name].dividend_amt.shift()\
             )/Stocks[max_alpha_name].close.shift()).bfill()
ISR = getInterceptSlope(stock_ret, index_ret)
ax[0,0].scatter(index_ret, stock_ret)
ax[0,0].plot(index_ret, ISR[0]+ISR[1]*index_ret, "r")
ax[0,0].title.set_text("Return on "+max_alpha_name+"={:.4f}+{:.4f}*Return on Market".format(ISR[0],ISR[1])\
                       +"  α(max): {:.4f}".format(max_alpha))

# stock with min_alpha
stock_ret = ((Stocks[min_alpha_name].close.diff() + Stocks[min_alpha_name].dividend_amt.shift()\
             )/Stocks[min_alpha_name].close.shift()).bfill()
ISR = getInterceptSlope(stock_ret, index_ret)
ax[0,1].scatter(index_ret, stock_ret)
ax[0,1].plot(index_ret, ISR[0]+ISR[1]*index_ret, "r")
ax[0,1].title.set_text("Return on "+min_alpha_name+"={:.4f}+{:.4f}*Return on Market".format(ISR[0],ISR[1])\
                       +"  α(min): {:.4f}".format(min_alpha))

# stock with max_rsquared
stock_ret = ((Stocks[max_rsquared_name].close.diff() + Stocks[max_rsquared_name].dividend_amt.shift()\
             )/Stocks[max_rsquared_name].close.shift()).bfill()
ISR = getInterceptSlope(stock_ret, index_ret)
ax[1,0].scatter(index_ret, stock_ret)
ax[1,0].plot(index_ret, ISR[0]+ISR[1]*index_ret, "r")
ax[1,0].title.set_text("Return on "+max_rsquared_name+"={:.4f}+{:.4f}*Return on Market".format(ISR[0],ISR[1])\
                       +"  R²(max): {:.4f}".format(max_rsquared))

# stock with min_rsquared
stock_ret = ((Stocks[min_rsquared_name].close.diff() + Stocks[min_rsquared_name].dividend_amt.shift()\
             )/Stocks[min_rsquared_name].close.shift()).bfill()
ISR = getInterceptSlope(stock_ret, index_ret)
ax[1,1].scatter(index_ret, stock_ret)
ax[1,1].plot(index_ret, ISR[0]+ISR[1]*index_ret, "r")
ax[1,1].title.set_text("Return on "+min_rsquared_name+"={:.4f}+{:.4f}*Return on Market".format(ISR[0],ISR[1])\
                       +"  R²(min): {:.4f}".format(min_rsquared))

plt.show()

# # store in local catalog
# Stocks[max_alpha_name].to_csv(str(Stocks_Name[max_alpha_name])+"_max_alpha.csv")
# Stocks[min_alpha_name].to_csv(str(Stocks_Name[min_alpha_name])+"_min_alpha.csv")
# Stocks[max_rsquared_name].to_csv(str(Stocks_Name[max_rsquared_name])+"_max_rsquared.csv")
# Stocks[min_rsquared_name].to_csv(str(Stocks_Name[min_rsquared_name])+"_min_rsquared.csv")