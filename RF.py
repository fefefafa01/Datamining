import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_recall_curve

# Tính toán chỉ số RSI
def compute_rsi(data, n=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Tính toán chỉ số SMA
def compute_sma(data, n=200):
    return data.rolling(n).mean()

# Lấy dữ liệu lịch sử của cổ phiếu
tickers = ["AAPL", "AMZN", "GOOG", "COST", "TSLA", "EVA", "BN", "ABBV", "CO", "PL", "CD", "AMJ"]
data = yf.download(tickers, period='5y', group_by='ticker')

# Tính toán các chỉ số kỹ thuật
for ticker in tickers:
    data[ticker + '_RSI'] = compute_rsi(data[ticker])
    data[ticker + '_SMA_200'] = compute_sma(data[ticker]['Close'])

# Xóa bỏ các dòng không có đầy đủ thông tin về các chỉ số
data.dropna(inplace=True)

# Chuẩn bị dữ liệu cho mô hình
X_cols = [ticker + '_RSI' for ticker in tickers] + [ticker + '_SMA_200' for ticker in tickers]
X = data[X_cols]
y = (data["AAPL"]["Close"].shift(-1) > data["AAPL"]["Close"]).astype(int)

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình phân loại và cross-validation
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean score: {cv_scores.mean()}")

# Huấn luyện mô hình
rf.fit(X_train, y_train)

# Dự đoán trên cổ phiếu được chọn để test
test_ticker = "AAPL"
test_data = data.tail(1)[X_cols]
prediction = rf.predict(test_data)

# Xuất classification report
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

