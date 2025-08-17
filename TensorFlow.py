import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv(r"C:\Users\User\Downloads\googl_us_d.csv")  # replace with your file
print(df.head())

# Assume dataset has "Date" and "Close" columns
# If "Date" is not needed, drop it
if "Date" in df.columns:
    df = df.drop(columns=["Date"])

# =========================
# 2. Preprocess Data
# =========================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

# Sequence length
seq_length = 60
X, y = [], []

for i in range(seq_length, len(scaled_data)):
    X.append(scaled_data[i - seq_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# =========================
# 3. Build LSTM Model
# =========================
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, batch_size=32, epochs=5)

# =========================
# 4. Predictions
# =========================
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Inverse transform actual y_test for comparison
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# =========================
# 5. Plot Results
# =========================
plt.figure(figsize=(12,6))
plt.plot(actual_prices, color="green", label="Actual Prices")
plt.plot(predictions, color="red", label="Predicted Prices")
plt.title("Stock Price Prediction on Test Data")
plt.xlabel("Time Interval")  # just intervals, not actual dates
plt.ylabel("Price")
plt.legend()

# Show fewer x-axis ticks for readability
plt.xticks(np.linspace(0, len(actual_prices)-1, 10, dtype=int))

plt.show()