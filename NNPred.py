import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Use 'ggplot' style for matplotlib
plt.style.use('ggplot')

# Suppress specific pandas warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
plt.style.use('ggplot')

class TrainingLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\nEnd of epoch {epoch + 1}")
        print(f"Training Loss: {logs.get('loss')}")
        print(f"Validation Loss: {logs.get('val_loss')}")
        print(f"Training MAE: {logs.get('mae')}")
        print(f"Validation MAE: {logs.get('val_mae')}")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def add_lags(df, num_days_pred):
    target = 'Close'
    df = df.copy()
    for i in range(1, 13):  # Create 12 lags
        df.loc[:, f'lag{i}'] = df.loc[:, target].shift(num_days_pred * i)
    return df

def create_features(df):
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def prepare_data_for_nn(df, num_days_pred):
    df = create_features(df)
    df = add_lags(df, num_days_pred)
    df.dropna(inplace=True)

    X = df.drop(columns='Close')
    y = df['Close'].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y, X.index

def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def main():
    stock_ticker = input("Enter the stock ticker symbol: ")
    num_days_pred = int(input("Enter the number of days to predict into the future: "))
    years_of_data = int(input("Enter the number of years of historical data to use: "))

    stock_data = yf.download(stock_ticker)
    slice_point = int(len(stock_data) - 365 * years_of_data)
    stock_data = stock_data.iloc[slice_point:]
    stock_data.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)

    X_scaled, y_scaled, scaler_X, scaler_y, indices = prepare_data_for_nn(stock_data, num_days_pred)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X_scaled, y_scaled, indices, test_size=0.3, random_state=42
    )

    model = build_model(X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping, TrainingLoggingCallback()], batch_size=32, verbose=2)
    y_pred_test_nn = model.predict(X_test)
    y_test_inverse = scaler_y.inverse_transform(y_test)
    y_pred_test_nn_inverse = scaler_y.inverse_transform(y_pred_test_nn)

    nn_loss = mean_absolute_percentage_error(y_test_inverse, y_pred_test_nn_inverse)
    print(f"ERROR PERCENT = {nn_loss}% ")

    plt.figure(figsize=(12, 8))
    plt.scatter(indices_test, y_test_inverse, color='blue', label='Actual (Test Set)')
    plt.scatter(indices_test, y_pred_test_nn_inverse, color='red', label='Predicted (Test Set)')

    # Predicting future data
    last_date = stock_data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=num_days_pred + 1)[1:]
    future_data = pd.DataFrame(index=future_dates)
    future_data = create_features(future_data)

    # Make sure future_data has all the columns needed for prediction
    # This assumes the last observed 'Close' value is carried forward
    for i in range(1, 13):
        future_data[f'lag{i}'] = stock_data['Close'].iloc[-(num_days_pred * i) if (num_days_pred * i) < len(stock_data) else -1]

    future_data_scaled = scaler_X.transform(future_data)
    future_predictions = model.predict(future_data_scaled)
    future_predictions_inverse = scaler_y.inverse_transform(future_predictions)
    print("Future Predictions:")
    for date, prediction in zip(future_dates, future_predictions_inverse.flatten()):
        print(f"Date: {date.strftime('%Y-%m-%d')}, Predicted Stock Value: {prediction:.2f}")


    plt.plot(future_dates.to_numpy(), future_predictions_inverse, color='green', linestyle='--', marker='o', label='Future Predicted')
    plt.title(f'Actual vs Predicted Stock Values and Future Predictions for {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Stock Value')
    plt.legend()
    plt.show()

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"{stock_ticker}_{current_date}_predicted_vs_actual_NN.png"
    plt.savefig(filename)

if __name__ == "__main__":
    main()
