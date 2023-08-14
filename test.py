import yfinance as yf
import os
import sys
import requests

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Tính toán chỉ số RSI
def compute_rsi(data, n=14):
    gain, loss = data[['symbol','change in price']].copy(), data[['symbol','change in price']].copy()

    gain.loc['change in price'] = gain.loc[(gain['change in price'] < 0), 'change in price'] = 0
    loss.loc['change in price'] = loss.loc[(loss['change in price'] > 0), 'change in price'] = 0
    loss['change in price'] = loss['change in price'].abs()

    gain_avg = gain.groupby('symbol')['change in price'].transform(lambda x: x.ewm(span = n).mean())
    loss_avg = loss.groupby('symbol')['change in price'].transform(lambda x: x.ewm(span = n).mean())

    rs = gain_avg / loss_avg
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Nhập dữ liệu vào data
    data['Down_days'] = loss['change in price']
    data['Up_days'] = gain['change in price']
    data['RSI'] = rsi
    
    return data
    

# Tính toán chỉ số SMA
def compute_sma(data, n=200):
    return data.rolling(n).mean()

# Tính toán chỉ số Stochastic Oscillator
def compute_SO(data, n=14):
    low, high = data[['Low', 'symbol']].copy(), data[['High', 'symbol']].copy()
    
    low = low.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = n).min())
    high = high.groupby('symbol')['High'].transform(lambda x: x.rolling(window = n).max())
    
    #Tính Stochastic Oscillator
    k_percent = 100*((data['Close'] - low) / (high - low))
    
    #Nhập dữ liệu vào data
    data['Low_14'] = low
    data['High_14'] = high
    data['K_percent'] = k_percent
    
    return data
    
    
# Tính toán chỉ số Williams %R
def compute_R(data, n=14):
    low, high = data[['Low', 'symbol']].copy(), data[['High', 'symbol']].copy()
    
    low = low.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = n).min())
    high = high.groupby('symbol')['High'].transform(lambda x: x.rolling(window = n).max())
    
    #Tính Williams %R
    r_percent = ((high - data['Close']) / (high - low)) * (-100)
    
    #Nhập dữ liệu vào data
    data['R_percent'] = r_percent
    
    return data
    

#Tính toán chỉ số MACD
def compute_MACD(data):
    ema_26 = data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span = 26).mean())
    ema_12 = data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span = 12).mean())
    macd = ema_12 - ema_26
    
    ema_9_macd = macd.ewm(span = 9).mean()
    
    data['MACD'] = macd
    data['MACD_EMA'] = ema_9_macd
    
    return data
    
#Tính toán chỉ số Price Rate Of Change
def compute_PROC(data, n=9):
    data['Price_Rate_Of_Change'] = data.groupby('symbol')['Close'].transform(lambda x: x.pct_change(periods = n))
    
    return data
    
#Tính toán chỉ số Lượng Cân Bằng (On Balance Volume)
def compute_OBV(data):
    volume = data['Volume']
    change = data['Close'].diff()
    
    prev_obv = 0
    obv_values = []
    
    for i,j in zip(change, volume):
        if i>0:
            cur_obv = prev_obv + j
        elif i<0:
            cur_obv = prev_obv - j
        else:
            cur_obv = prev_obv
        
        prev_obv = cur_obv
        obv_values.append(cur_obv)
    
    return pd.Series(obv_values, index= data.index)

# Lấy dữ liệu lịch sử của cổ phiếu
def grab_price_data():
    tickers = ["AAPL", "AMZN", "GOOG", "COST", "TSLA", "EVA", "BN", "ABBV", "CO", "PL", "CD", "AMJ"]
    full_price_history = []
    for ticker in tickers:
        data = yf.download(ticker, period='2y', interval='1d')
        
        data['symbol'] = ticker
        full_price_history.append(data)

    price_data = pd.concat(full_price_history)
    
    price_data.reset_index('Date', inplace=True)
    
    price_data.to_csv('price_data.csv', index=False)

if os.path.exists('price_data.csv'):
    price_data = pd.read_csv('price_data.csv')
else:
    #grab data
    grab_price_data()
    #load data
    price_data = pd.read_csv('price_data.csv', index_col= 0)
    
price_data = price_data[['symbol','Date','Close','High','Low','Open','Volume']]

price_data.sort_values(by = ['symbol','Date'], inplace = True)

#calculate the change in price
price_data['change in price'] = price_data['Close'].diff()

mask = price_data['symbol'] != price_data['symbol'].shift(1)

price_data['change in price'] = np.where(mask == True, np.nan, price_data['change in price'])

price_data[price_data.isna().any(axis= 1)]   

#smothing
# define the number of days out you want to predict
days_out = 30

# Group by symbol, then apply the rolling function and grab the Min and Max.
price_data_smoothed = price_data.groupby(['symbol'])[['Close','Low','High','Open','Volume']].transform(lambda x: x.ewm(span = days_out).mean())

# Join the smoothed columns with the symbol and datetime column from the old data frame.
smoothed_df = pd.concat([price_data[['symbol','Date']], price_data_smoothed], axis=1, sort=False)

smoothed_df 

#signal flag
# define the number of days out you want to predict
days_out = 30

# create a new column that will house the flag, and for each group calculate the diff compared to 30 days ago. Then use Numpy to define the sign.
smoothed_df['Signal_Flag'] = smoothed_df.groupby('symbol')['Close'].transform(lambda x : np.sign(x.diff(days_out)))

#tính rsi (>70 --> quá mua nên bán, <30 --> quá bán, nên mua)
compute_rsi(price_data)

#tính Stochastic Oscillator (>80 --> quá mua nên bán, <20 --> quá bán nên mua)
compute_SO(price_data)

#tính Williams %R (-80-->-50 --> đang tăng giá, -20-->-50 -->đang giảm giá)
compute_R(price_data)

#tính MACD()
compute_MACD(price_data)

#tính PROC
compute_PROC(price_data)

#tính OBV
obv_groups = price_data.groupby('symbol').apply(compute_OBV)

price_data['On_Balance_Volume'] = obv_groups.reset_index(level= 0, drop= True)

##Build model
close_groups = price_data.groupby('symbol')['Close']

# Apply the lambda function which will return -1.0 for down, 1.0 for up and 0.0 for no change.
close_groups = close_groups.transform(lambda x : np.sign(x.diff()))

# add the data to the main dataframe.
price_data['Prediction'] = close_groups

# for simplicity in later sections I'm going to make a change to our prediction column. To keep this as a binary classifier I'll change flat days and consider them up days.
price_data.loc[price_data['Prediction'] == 0.0] = 1.0

#Drop NaN values
print('Before Drop: {} x {}'.format(price_data.shape[0], price_data.shape[1]))

# Any row that has a `NaN` value will be dropped.
price_data = price_data.dropna()

# Display how much we have left now.
print('After Drop: {} x {}'.format(price_data.shape[0], price_data.shape[1]))

# Print the head.
price_data.head()

#Chuẩn bị dữ liệu cho mô hình
print('Values : RSI, K_percent(Stochastic Oscillator), R_percent(William %R), PROC, MACD')

# Grab our X & Y Columns.
X_Cols = price_data[['RSI','K_percent','R_percent','Price_Rate_Of_Change','MACD','On_Balance_Volume']]
Y_Cols = price_data['Prediction']

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state = 0)

# Create a Random Forest Classifier
rf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 0)

# Fit the data to the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Print the Accuracy of our Model.
print('Correct Prediction (%): ', accuracy_score(y_test, rf.predict(X_test), normalize = True) * 100.0)

#Target
target_names = ['Down Day', 'Up Day']

# Build a classifcation report
report = classification_report(y_true = y_test, y_pred = y_pred, target_names = target_names, output_dict = True)

# Add it to a data frame, transpose it for readability.
report_df = pd.DataFrame(report).transpose()
print(report_df)

#Ma trận nhầm lẫn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

rf_matrix = confusion_matrix(y_test, y_pred)

TN = rf_matrix[0][0]
FN = rf_matrix[1][0]
TP = rf_matrix[1][1]
FP = rf_matrix[0][1]

accuracy = (TN + TP) / (TP + TN + FP + FN)
presicion = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)

print('Accuracy : {}'.format(float(accuracy)))
print('Presicion : {}'.format(float(presicion)))
print('Recall : {}'.format(float(recall)))
print('Specificity : {}'.format(float(specificity)))

disp = ConfusionMatrixDisplay(confusion_matrix=rf_matrix, display_labels=['Down Day', 'Up Day'])
disp.plot(cmap=plt.cm.Blues, values_format='.2f')
disp.ax_.set_title('Confsion Matrix - N')
plt.show()

print('Random Forest Out-Of-Bag Error Score: {}'.format(rf.oob_score_))