# Step 0: Install required libraries (run this in a separate cell if needed)
# Uncomment these lines if you don't have the libraries installed yet.
# !pip install yfinance
# !pip install tensorflow  
# !pip install matplotlib
# !pip install scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from itertools import product

# Step 1: Fetch Historical Data
stock_symbol = "AAL"  # Stock symbol
start_date = "2017-01-01"
end_date = "2021-12-31"  # Updated end date to include data until 2021

# Download historical stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)
if data.empty:
    print("No data fetched. Please check the stock symbol and date range.")
else:
    data = data[['Adj Close']]  # Use only the Adjusted Close column

    # Step 2: Data Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create training data
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):  # Create look-back sequences
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    # Convert to numpy arrays and reshape
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Step 3: Define a function to create and train the LSTM model
    def create_lstm_model(units, dropout_rate):
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=False, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(rate=dropout_rate))  # Add dropout
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Step 4: Grid Search for Hyperparameter Tuning
    units_list = [100]  # Number of units in the LSTM layer
    dropout_rates = [0.2]  # Dropout rates
    batch_sizes = [16]  # Batch sizes
    epochs_list = [20]  # Number of epochs

    # Create a grid of hyperparameter combinations
    param_grid = list(product(units_list, dropout_rates, batch_sizes, epochs_list))

    # Initialize variables to store the best parameters and their performance
    best_model = None
    best_params = None
    best_mse = float('inf')  # Start with a very high MSE

    # Perform grid search
    for units, dropout_rate, batch_size, epochs in param_grid:
        print(f"Training with units={units}, dropout_rate={dropout_rate}, batch_size={batch_size}, epochs={epochs}")
        model = create_lstm_model(units, dropout_rate)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)  # Train the model silently

        # Evaluate model on the training data
        train_predictions = model.predict(X_train, verbose=0)
        mse = mean_squared_error(y_train, train_predictions)

        print(f"Model MSE: {mse:.4f}")

        # Update the best model if this one is better
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_params = (units, dropout_rate, batch_size, epochs)

    print("\nBest Hyperparameters:")
    print(f"Units: {best_params[0]}, Dropout Rate: {best_params[1]}, Batch Size: {best_params[2]}, Epochs: {best_params[3]}")
    print(f"Best MSE: {best_mse:.4f}")

    # Use the best model for predictions
    model = best_model

    # Step 5: Predict Stock Prices for the Entire Dataset
    full_predictions = []
    input_sequence = scaled_data[:60]  # Start with the first 60 days

    # Store the first 60 actual prices
    full_predictions.extend(scaled_data[:60, 0])  # Add actual values for the first 60 days

    for i in range(len(scaled_data) - 60):  # Predict for the entire dataset
        input_sequence = np.reshape(input_sequence, (1, input_sequence.shape[0], 1))
        predicted_price = model.predict(input_sequence, verbose=0)
        full_predictions.append(predicted_price[0, 0])
        input_sequence = np.append(input_sequence[0, 1:], scaled_data[60 + i, 0])  # Shift input

    # Inverse transform predictions
    full_predictions = scaler.inverse_transform(np.array(full_predictions).reshape(-1, 1))

    # Step 6: Calculate and Display Metrics
    predicted_prices = full_predictions[60:]  # Skip the first 60 predicted prices
    actual_prices = data['Adj Close'][60:]

    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    accuracy_percentage = 100 - mape

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")

    # Step 7: Plot Net Gains and Losses
    data['Daily Return'] = data['Adj Close'].pct_change()
    data['Gain'] = data['Daily Return'].apply(lambda x: x if x > 0 else 0)
    data['Loss'] = data['Daily Return'].apply(lambda x: -x if x < 0 else 0)

    plt.figure(figsize=(14, 7))
    plt.bar(data.index, data['Gain'], color='green', label='Net Gain')
    plt.title("Net Gains")
    plt.xlabel("Date")
    plt.ylabel("Daily Gains")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.bar(data.index, data['Loss'], color='red', label='Net Loss')
    plt.title("Net Losses")
    plt.xlabel("Date")
    plt.ylabel("Daily Losses")
    plt.legend()
    plt.grid()
    plt.show()

    # Step 8: Future Predictions
    future_days = 10
    last_60_days = scaled_data[-60:]
    future_predictions = []
    input_sequence = last_60_days.copy()

    for _ in range(future_days):
        input_sequence = np.reshape(input_sequence, (1, input_sequence.shape[0], 1))
        predicted_price = model.predict(input_sequence, verbose=0)
        future_predictions.append(predicted_price[0, 0])
        input_sequence = np.append(input_sequence[0, 1:], predicted_price[0, 0])

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1, freq='B')[1:]

    # Step 9: Plot Full Data with Future Predictions
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['Adj Close'], label='Historical Prices', color='blue')
    plt.plot(data.index, full_predictions, label='Predicted Prices', color='orange')
    plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='dashed', color='green')
    plt.title(f"{stock_symbol} Stock Price Prediction", fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()
