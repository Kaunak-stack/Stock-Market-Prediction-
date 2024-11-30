# Stock Price Prediction with LSTM (American Airlines - AAL)

This project demonstrates how to predict stock prices for **American Airlines Group Inc. (AAL)** using a Long Short-Term Memory (LSTM) neural network. The model is trained on historical stock price data and evaluated based on prediction accuracy. The project also visualizes predictions and calculates metrics to measure performance.

---

## Features

1. **Fetch Historical Data**:
   - Automatically downloads historical stock prices for AAL from Yahoo Finance using the `yfinance` library.
   - Uses the **Adjusted Close** price for training and predictions.
2. **Data Preprocessing**:
   - Scales prices to the range `[0, 1]` using `MinMaxScaler`.
   - Creates input sequences using a 60-day look-back period.
3. **LSTM Model**:
   - Builds an LSTM model optimized for time-series forecasting.
   - Performs grid search to find the best hyperparameters.
4. **Evaluation Metrics**:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Accuracy Percentage
5. **Visualization**:
   - Plots Historical vs. Predicted Prices.
   - Displays daily gains and losses as bar charts.
   - Predicts and visualizes future prices for the next 10 business days.

---

## Prerequisites

You need the following Python libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `tensorflow`
- `scikit-learn`

Install them with:
```bash
pip install numpy pandas matplotlib yfinance tensorflow scikit-learn
```

---

## How to Run

### 1. Clone or Download the Repository
Save the script as `stock_prediction.py`.

### 2. Run the Script
Execute the script using:
```bash
python stock_prediction.py
```

---

## How the Code Works

### Step 1: Fetch Historical Data
The script fetches **Adjusted Close** prices for AAL stock from Yahoo Finance. By default, it retrieves data from **January 1, 2017**, to **December 31, 2021**. Example of fetched data:
```
         Date         Adj Close
    2017-01-03       49.726276
    2017-01-04       49.954285
    2017-01-05       49.591915
```

### Step 2: Data Preprocessing
The data is scaled using `MinMaxScaler` to the range `[0, 1]`. A sliding window of 60 days is used to create sequences for training:
- Input: Prices for the past 60 days.
- Output: Price for the next day.

### Step 3: Build and Train the LSTM Model
The script builds an LSTM model and optimizes hyperparameters like:
- Number of LSTM units
- Dropout rate
- Batch size
- Number of epochs

The best configuration is chosen based on the lowest Mean Squared Error (MSE).

### Step 4: Predict and Evaluate
- The model predicts stock prices for the entire dataset.
- Metrics such as MSE, MAE, and Accuracy Percentage are calculated to evaluate the model.

### Step 5: Visualize Results
- **Historical vs. Predicted Prices**: Plots show how closely the predictions match actual prices.
- **Daily Gains and Losses**: Bar charts display the percentage changes in prices.
- **Future Predictions**: Predicts stock prices for the next 10 business days.

---

## Outputs

### Metrics
Example evaluation metrics:
```
Best Hyperparameters:
Units: 100, Dropout Rate: 0.2, Batch Size: 16, Epochs: 20
Best MSE: 0.0012

Model Performance:
MSE: 5.2345
MAE: 2.1345
Accuracy Percentage: 95.34%
```

### Plots
1. **Historical vs. Predicted Prices**:
   - A line plot comparing actual prices to predicted prices.

2. **Daily Gains and Losses**:
   - Green bars represent gains, and red bars represent losses.

3. **Future Price Predictions**:
   - A graph showing predictions for the next 10 business days.

---

## Customization

1. **Stock Symbol**:
   - Change the `stock_symbol` variable to analyze other stocks (e.g., "AAPL", "MSFT").
2. **Date Range**:
   - Modify `start_date` and `end_date` for a different time period.
3. **Future Days**:
   - Adjust the `future_days` variable to predict more or fewer future days.

---

## Example Visualizations

### Historical vs. Predicted Prices
A plot showing how closely the model's predictions align with actual prices.

### Daily Gains and Losses
Bar charts visualizing daily price changes, split into gains and losses.

### Future Price Predictions
A graph showing predictions for the next 10 business days.

---

## Limitations

1. **Single Feature**:
   - The model only uses past prices for predictions. External factors like news or market sentiment are not considered.
2. **Overfitting**:
   - The model might overfit the training data and struggle with unseen data.
3. **External Factors**:
   - Does not account for macroeconomic events or industry-specific news.

---

## Future Enhancements

1. **Additional Models**:
   - Incorporate GRU, ARIMA, or Prophet for comparison.
2. **Sentiment Analysis**:
   - Use news articles and social media sentiment as additional features.
3. **Web Interface**:
   - Build an interactive dashboard using Streamlit or Flask.

---


![AAL STOCK PREDICITION](https://github.com/user-attachments/assets/5ab046dd-25c9-403b-83bd-b2e94585a3b6)

![NET LOSS GRAPH](https://github.com/user-attachments/assets/a327b2ae-c504-4146-827c-bb6885410627)

![NET GAIN GRAPH](https://github.com/user-attachments/assets/915c1879-8f50-49fc-a35c-2e5fba2516ff)




