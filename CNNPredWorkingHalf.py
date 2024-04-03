from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

plt.style.use('ggplot')

def create_features(df):
    df.loc[:, 'dayofweek'] = df.index.dayofweek
    df.loc[:, 'quarter'] = df.index.quarter
    df.loc[:, 'month'] = df.index.month
    df.loc[:, 'year'] = df.index.year
    df.loc[:, 'dayofyear'] = df.index.dayofyear
    df.loc[:, 'dayofmonth'] = df.index.day
    df.loc[:, 'weekofyear'] = df.index.isocalendar().week
    return df

def add_lags(df, n_lags):
    for i in range(1, n_lags + 1):
        df.loc[:, f'lag_{i}'] = df['Close'].shift(i)
    return df.dropna()

def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def mse_loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def forward_pass(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        return Z2, Z1, A1, Z2

    def backward_pass(self, X, Y, Z1, A1, Z2, output, lr):
        m = Y.shape[0]
        dZ2 = output - Y
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, Y, X_val, Y_val, initial_lr=0.1, factor=0.5, patience=10, epochs=100):
        lr = initial_lr
        best_val_loss = np.inf
        wait = 0

        for epoch in range(epochs):
            output, Z1, A1, Z2 = self.forward_pass(X)
            self.backward_pass(X, Y, Z1, A1, Z2, output, lr)

            train_loss = self.mse_loss(Y, output)
            val_output, _, _, _ = self.forward_pass(X_val)
            val_loss = self.mse_loss(Y_val, val_output)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                lr *= factor
                wait = 0

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

    def predict(self, X):
        output, _, _, _ = self.forward_pass(X)
        return output
def main():
    stock_ticker = input("Enter the stock ticker symbol: ")
    years_of_data = int(input("Enter the number of years of historical data to use: "))

    # Fetch stock data
    stock_data = yf.download(stock_ticker)
    stock_data = stock_data[-365*years_of_data:]  # Get last 'years_of_data' years
    stock_data = create_features(stock_data)
    stock_data = add_lags(stock_data, 12)  # Add 12 lags of the 'Close' feature

    # Prepare input X and target Y
    X = stock_data.drop(columns='Close').values
    Y = stock_data[['Close']].values

    scaler_X, scaler_Y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    split_point = int(len(X_scaled) * 0.8)
    X_train, Y_train = X_scaled[:split_point], Y_scaled[:split_point]
    X_val, Y_val = X_scaled[split_point:], Y_scaled[split_point:]

    nn = SimpleNN(input_size=X_train.shape[1], hidden_size=50, output_size=1)
    nn.train(X_train, Y_train, X_val, Y_val, initial_lr=0.1, epochs=100)

    predictions_scaled = nn.predict(X_val)
    predictions = scaler_Y.inverse_transform(predictions_scaled)
    actual = scaler_Y.inverse_transform(Y_val)

    # Correcting the indices for the test set for plotting
    test_indices = stock_data.index[split_point:split_point + len(predictions)].to_numpy()

    plt.figure(figsize=(12, 8))
    plt.plot(test_indices, actual, color='blue', label='Actual Close Values')
    plt.plot(test_indices, predictions, color='red', linestyle='--', label='Predicted Close Values')
    plt.title(f'Actual vs Predicted Stock Close Values for {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Stock Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
