import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Use 'ggplot' style for matplotlib
plt.style.use('ggplot')

def main():
    stock_ticker = input("Enter the stock ticker symbol: ")
    years_of_data = int(input("Enter the number of years of historical data to use: "))

    # Fetch stock data
    stock_data = yf.download(stock_ticker)

    # Filter data to include only the last specified years
    slice_point = int(len(stock_data) - 365 * years_of_data)
    stock_data = stock_data.iloc[slice_point:]

    # Keep only the 'Close' values
    stock_data = stock_data[['Close']]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(stock_data.index.to_numpy(), stock_data['Close'].to_numpy(), color='blue', label='Actual Close Values')
    plt.title(f'Actual Stock Close Values for {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Stock Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

