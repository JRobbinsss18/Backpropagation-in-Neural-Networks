import datetime
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yfinance as yf

import pandas_datareader as pdr
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from pmdarima import auto_arima

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def add_lags(df,num_days_pred):
    target = 'Close'
    df['lag1'] = df[target].shift(num_days_pred)  
    df['lag2'] = df[target].shift(num_days_pred*2)    
    df['lag3'] = df[target].shift(num_days_pred*3)    
    df['lag4'] = df[target].shift(num_days_pred*4)    
    df['lag5'] = df[target].shift(num_days_pred*5)
    df['lag6'] = df[target].shift(num_days_pred*6)
    df['lag7'] = df[target].shift(num_days_pred*7)
    df['lag8'] = df[target].shift(num_days_pred*8)
    df['lag9'] = df[target].shift(num_days_pred*9)
    df['lag10'] = df[target].shift(num_days_pred*10)
    df['lag11'] = df[target].shift(num_days_pred*11)
    df['lag12'] = df[target].shift(num_days_pred*12)



    return df

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df
def xgboostmodel(df_xgb,create_features,num_days_pred):

    df_xgb = create_features(df_xgb)
    df_xgb = add_lags(df_xgb,num_days_pred)
    
    X = df_xgb.drop(columns='Close')
    y = df_xgb['Close']
    return X,y

def objective(trial, X_train, y_train, X_test, y_test):
    # Define hyperparameters to search
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'verbosity': 0,
        #'tree_method': 'gpu_hist',
    }
    
    # Initialize XGBoost regressor with the suggested parameters
    xgb = XGBRegressor(**param)
    
    # Fit the model on training data
    xgb.fit(X_train, y_train)
    
    # Predict on the validation set
    y_pred = xgb.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return rmse
import datetime  # Ensure this is imported at the beginning of your script

def main():
    stock_ticker = input("Enter the stock ticker symbol (e.g., AAPL): ")
    num_days_pred = int(input("Enter the number of days to predict into the future: "))
    years_of_data = int(input("Enter the number of years of historical data to use: "))
    
    # Download stock data
    stock_data = yf.download(stock_ticker)
    slice_point = int(len(stock_data) - 365 * years_of_data)
    stock_data = stock_data.iloc[slice_point:]
    stock_data.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
    
    # Prepare data for XGBoost model
    df_xgb = stock_data.copy()
    X, y = xgboostmodel(df_xgb, create_features, num_days_pred)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Perform hyperparameter optimization using Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)
    
    # Print the best trial and parameters found
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Use the best parameters to train the final model
    best_params = best_trial.params
    xgb_best = XGBRegressor(**best_params)
    xgb_best.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred_test_xgb = xgb_best.predict(X_test)
    
    # Calculate and print the error percentage
    xgb_loss = mean_absolute_percentage_error(y_test, y_pred_test_xgb)
    print(f"ERROR PERCENT = {xgb_loss}% ")

    # Plotting actual vs predicted values and future predictions
    plt.figure(figsize=(12, 8))
    plt.scatter(X_test.index, y_test, color='blue', label='Actual (Test Set)')
    plt.scatter(X_test.index, y_pred_test_xgb, color='red', label='Predicted (Test Set)')

    # Prepare future data for prediction
    start = df_xgb.index.max()
    end = start + pd.Timedelta(days=num_days_pred)
    future = pd.date_range(start=start, end=end, freq='1d')
    future_df = pd.DataFrame(index=future)
    future_df['isFuture'] = True
    df_xgb['isFuture'] = False
    df_and_future = pd.concat([df_xgb, future_df])
    df_and_future = create_features(df_and_future)
    df_and_future = add_lags(df_and_future, num_days_pred)
    future_w_features = df_and_future.query('isFuture').copy()
    future_w_features['pred'] = xgb_best.predict(future_w_features.drop(columns=['Close', 'isFuture']))
    prediction_xgb = pd.DataFrame(future_w_features['pred'])

    # Plot the future predicted values
    plt.plot(prediction_xgb.index.to_numpy(), prediction_xgb['pred'].to_numpy(), color='green', linestyle='--', marker='o', label='Future Predicted')
    # Setting the title and labels
    plt.title(f'Actual vs Predicted Stock Values and Future Predictions for {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Stock Value')
    plt.legend()
    plt.gcf().autofmt_xdate()  # Auto-format date labels

    # Save the plot
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"{stock_ticker}_{current_date}_predicted_vs_actual_XBoost.png"
    plt.savefig(filename)
    # plt.show()  # Uncomment if you want to display the plot in Jupyter Notebook

    # Print future predictions
    print(prediction_xgb)

if __name__ == "__main__":
    main()

