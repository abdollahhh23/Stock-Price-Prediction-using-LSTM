# Stock-Price-Prediction-using-LSTM
This project applies Long Short-Term Memory (LSTM) neural networks to forecast stock closing prices. LSTM is particularly suited for sequential data such as financial time series because it can capture long-term dependencies better than traditional models.

This project applies Long Short-Term Memory (LSTM) neural networks to forecast stock closing prices. LSTM is particularly suited for sequential data such as financial time series because it can capture long-term dependencies better than traditional models.

2. Dataset

Source: Historical stock market data (CSV).

Feature Used: Closing Price (Close).

Preprocessing:

Normalized prices using MinMaxScaler (0–1).

Created 60-day sliding windows to predict the next day.

Data split into 80% training, 20% testing.

3. Model Architecture

LSTM(50 units, return_sequences=True)

LSTM(50 units)

Dense(25 units)

Dense(1 unit) – Output layer

Optimizer: Adam

Loss Function: Mean Squared Error (MSE)

Epochs: 5

4. Results

Evaluation Metric: Root Mean Squared Error (RMSE)

Dataset	RMSE
Training	~20.31
Testing	~43.58

Visualization: The predicted prices (red) track actual prices (green), capturing the trend but with slight deviations in volatile regions.

5. Conclusion

LSTM is effective for short-term stock trend prediction.

Achieved low RMSE on training and reasonable performance on testing data.

Performance can be further improved with additional features (volume, open, high/low) and hyperparameter tuning.

6. Future Enhancements

Add technical indicators (EMA, RSI, MACD).

Explore GRU and Transformer-based models.

Implement multi-step forecasting.
