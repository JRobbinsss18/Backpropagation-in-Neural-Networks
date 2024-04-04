# Imports
import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Adding features for potential better estimations
def create_features(df):
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# In essence, when making estimations, this tells us how many previous values to consider
def add_lags(df, n_lags):
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    return df.dropna()


class SimpleNN:
    # Initialize weights and biases
    def __init__(self, input_size, hidden_size, output_size, lambd=0.01):
        # Initialize weights and biases from input layer to hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        # Initialize weights and biases from hidden layer to output layer
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        # Set lambd (regularized parameter
        self.lambd = lambd
    # Define ReLU activation function
    def relu(self, Z):
        # Sets all negative values in Z to 0
        return np.maximum(0, Z)
    # Define derivative of ReLU for backpropagation
    def relu_derivative(self, Z):
        # Derivative is 1 for all positive values, 0 otherwise
        return Z > 0
    # Define mean squared error loss function
    def mse_loss(self, y_true, y_pred):
        # Average of squared difference between true and predicted
        return np.mean((y_true - y_pred) ** 2)
    

    # ALTERNATE LOSS FUNCTION
    def huber_loss(self, y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error**2
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.where(is_small_error, squared_loss, linear_loss).mean()

    # Define forward pass, returning output for given input 'X'
    def forward_pass(self, X):
        # Calculate Pre activation and Post activation for the first hidden layer
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        # Calculate Pre activation for output layer
        Z2 = np.dot(A1, self.W2) + self.b2
        return Z2, Z1, A1, Z2
    # Define backwards pass, updating weights and biases based on the loss gradient
    def backward_pass(self, X, Y, Z1, A1, Z2, output, lr, lambd):
        # Get number of examples to normalize gradient
        m = Y.shape[0]
        # Calculate gradient of the loss with respect to Z2
        dZ2 = output - Y
        # Calculate gradients for weights and biases of output layer, including regularization for weights
        dW2 = (1 / m) * np.dot(A1.T, dZ2) + (lambd / m) * self.W2
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        # Backpropagate errors from output to hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        # Apply derivative of ReLU to get gradient of loss with respect to pre-activation of first layer
        dZ1 = dA1 * self.relu_derivative(Z1)
        # Compute gradient for weights and biases of first layer
        dW1 = (1 / m) * np.dot(X.T, dZ1) + (lambd / m) * self.W1
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        grad_clip_size_neg = -.5
        grad_clip_size_pos = abs(grad_clip_size_neg)
        # Employ gradient clipping to prevent exploding gradients
        dW1 = np.clip(dW1, grad_clip_size_neg, grad_clip_size_pos)
        db1 = np.clip(db1, grad_clip_size_neg, grad_clip_size_pos)
        dW2 = np.clip(dW2, grad_clip_size_neg, grad_clip_size_pos)
        db2 = np.clip(db2, grad_clip_size_neg, grad_clip_size_pos)

        # Print gradients
        print("dW1:", dW1)
        print("db1:", db1)
        print("dW2:", dW2)
        print("db2:", db2)
        
        # Update weights and biases
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, Y, X_val, Y_val, initial_lr=0.001, lr_increase_factor=1.02, lr_decrease_factor=0.98,
              smoothness_factor=0.5, patience=10, min_delta=0.0001, min_lr=1e-6, max_lr=0.1, epochs=100):
        # Set values to track best validation loss and adjustment for learning rate
        lr = initial_lr
        best_val_loss = np.inf
        wait = 0
        last_val_output = None
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            output, Z1, A1, Z2 = self.forward_pass(X)
            # Backwards pass
            self.backward_pass(X, Y, Z1, A1, Z2, output, lr, self.lambd)
            # Training loss computation
            # ---> train_loss = self.mse_loss(Y, output)
            train_loss = self.huber_loss(Y, output)
            # Forward pass performed once more on the validation set to compute validation loss
            val_output, _, _, _ = self.forward_pass(X_val)
            # ---> val_loss = self.mse_loss(Y_val, val_output)
            val_loss = self.huber_loss(Y_val, val_output)
            # Smoothing logic, if last epoch exist, validation change is computed,and smoothness loss is added to validation loss, attempting to penalize super large changes and encourage stability
            if last_val_output is not None:
                change_rate = val_output - last_val_output
                smoothness_loss = smoothness_factor * np.mean(np.diff(change_rate, n=2) ** 2)
                val_loss += smoothness_loss
            # Learning rate adjustments
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                wait = 0
                lr *= lr_decrease_factor
            else:
                wait += 1
                if wait >= patience:
                    wait = 0
                    lr *= lr_increase_factor
            lr = max(min(lr, max_lr), min_lr)
            last_val_output = val_output
            print(f"Epoch {epoch}, LR: {lr:.6f}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    # Predict, do a forward pass without validation
    def predict(self, X):
        output, _, _, _ = self.forward_pass(X)
        return output
# Function to help me find best lambda
def hyperparameter_tuning(X_train, Y_train, X_val, Y_val, lambdas, epochs=100):
    performance_records = []
    for lambd in lambdas:
        nn = SimpleNN(input_size=X_train.shape[1], hidden_size=50, output_size=1, lambd=lambd)
        nn.train(X_train, Y_train, X_val, Y_val, epochs=epochs)
        val_output, _, _, _ = nn.forward_pass(X_val)
        val_loss = nn.mse_loss(Y_val, val_output)
        performance_records.append((lambd, val_loss))
        
    # Sort based on validation loss
    performance_records.sort(key=lambda x: x[1])
    best_lambda, best_loss = performance_records[0]
    print(f"Best lambda: {best_lambda} with validation loss: {best_loss}")
    return best_lambda

def main():
    
    # User Inputs
    stock_ticker = input("Enter the stock ticker symbol (e.g., AAPL): ")
    years_of_data = int(input("Enter the number of years of historical data to use: "))
    days_to_predict = int(input("Enter the number of days to predict into the future: "))
    # Pull from API
    stock_data = yf.download(stock_ticker, period=f"{years_of_data}y")
    # Handle data and prepare for NN
    stock_data = create_features(stock_data)
    stock_data = add_lags(stock_data, 10)
    stock_data.dropna(inplace=True)

    X = stock_data.drop(['Close'], axis=1).values
    Y = stock_data['Close'].values.reshape(-1, 1)
    scaler_X, scaler_Y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)
    split = int(0.8 * len(X_scaled))
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    Y_train, Y_val = Y_scaled[:split], Y_scaled[split:]
    
    #lambdas = [0.001, 0.01, 0.1, 1, 10,100,1000]
    #best_lambda = hyperparameter_tuning(X_train, Y_train, X_val, Y_val, lambdas)
    
    # Initialize the NN
    nn = SimpleNN(input_size=X_train.shape[1], hidden_size=50, output_size=1, lambd=10)

    # Train the NN
    nn.train(X_train, Y_train, X_val, Y_val, initial_lr=0.05, epochs=200)

    # Predict
    predictions_scaled = nn.predict(X_val)
    predictions = scaler_Y.inverse_transform(predictions_scaled)
    actual = scaler_Y.inverse_transform(Y_val)

    future_X = X_scaled[-days_to_predict:]
    future_predictions_scaled = nn.predict(future_X)
    future_predictions = scaler_Y.inverse_transform(future_predictions_scaled)
    
    # Print predictions
    print("\nFuture predictions:")
    for i, prediction in enumerate(future_predictions, 1):
        print(f"Day {i}: {prediction[0]}")

    
    # Prepare to plot
    dates = stock_data.index[split:].to_numpy()
    last_known_date = dates[-1]
    future_dates = pd.bdate_range(start=last_known_date + pd.Timedelta(days=1), periods=days_to_predict).to_numpy()

    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(dates, actual, label='Actual Close Values', color='orange')
    plt.plot(dates, predictions, label='Predicted Close Values', linestyle='--', color='blue')
    plt.plot(future_dates, future_predictions, label='Future Predicted Close Values', color='green', linestyle='--')
    plt.title(f'Actual vs Predicted Stock Close Values for {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Stock Value')
    plt.legend()
    #plt.show()
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"{stock_ticker}_{current_date}_predicted_vs_actual_ManualNN.png"
    plt.savefig(filename)
    print(stock_data)
    print(f"{filename} saved to current directory!")
# Run main
if __name__ == "__main__":
    main()

