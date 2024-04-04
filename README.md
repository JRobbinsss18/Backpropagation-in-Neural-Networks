# About this Project

This project showcases 3 different machine learning models, each unique, with the main goal to show the working under the hood of a neural network.

## XGBoostPred.py

This program implements an algorithm called XGBoost for stock price prediction, it is a algorithm within machine learning that does NOT use backpropagation

## DNNPred.py

This program implements a Dense Neural Network VIA TensorFlow libraries, it uses backpropagation, and it is what we model our third code after.

## NNPred.py

This program creates a neural network by hand, implementing various features to achieve a better price prediction. it uses backpropagation.

# Getting Started

Here we will discuss how to get our codes up and running


## First, make sure you have python, this was developed on Python 3.10.12

# Install prerequisites

### XGBoostPred.py

run ``` pip install matplotlib numpy optuna pandas yfinance pandas_datareader prophet scikit-learn statsmodels xgboost pmdarim ```

## DNNPred.py

run ``` pip install matplotlib numpy pandas yfinance scikit-learn tensorflow ```

## ManualNNPred.py
run ``` pip install matplotlib numpy pandas yfinance scikit-learn tensorflow ```

# Run the program

Simply get to the directory VIA terminal, run 'python *filename*

Running this code will give you predictions at the command line but also save a file to your current directory for the chart


# Example outputs
## XGBoost
![XGBoost Version](Examples/AAPL_20240324_predicted_vs_actual_XBoost.png "XGBoost Version")

## DNN
![DNN Version]()

# ManualNN
![Manual NN Version]()
