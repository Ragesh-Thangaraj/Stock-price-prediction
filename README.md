# Stock-price-prediction
Stock Price Prediction Using Simple RNN
This project implements a Recurrent Neural Network (RNN) to predict future stock prices based on historical stock price data using Keras and TensorFlow.

# Overview
The goal of this project is to predict the next stock price of Meta (formerly Facebook) using an RNN model, trained on historical price data. The model is designed to take a sequence of stock prices (a lookback window of 25 days) and predict the stock price for the next day.

# Project Workflow
## Data Preparation:

The dataset used in this project contains stock prices for Meta (FB), with columns:
Date: The date of the stock price
Price: The stock price for the day
Data is loaded from a CSV file, and the prices are normalized using StandardScaler.

## Creating an RNN Model:

A simple RNN model is built using Keras. It consists of:
A SimpleRNN layer with 32 units.
A Dense layer to output a single predicted price.
The model is trained to minimize the mean squared error (MSE) between the actual and predicted prices.

## Training and Testing:

The dataset is split into training and testing sets. The training set is used to fit the model, and the test set is used to evaluate its performance.
The model is trained over 5 epochs with a batch size of 1 for improved performance on the small dataset.
Making Predictions:

After training, the model is used to predict the stock prices on the test set.
The predicted prices are scaled back to their original form using inverse_transform from StandardScaler for interpretability.

## Prediction Visualization:

A plot is generated comparing the predicted prices against the actual prices for the test set, showing the performance of the RNN model.
Predicting Future Prices:

The model is also used to predict the next stock price based on a new sequence of past stock prices.

## Requirements
The following Python libraries are needed for the project:

numpy
pandas
matplotlib
tensorflow
keras
scikit-learn

##You can install the dependencies using:

pip install -r requirements.txt
Or install them individually:

pip install numpy pandas matplotlib tensorflow keras scikit-learn

# How to Run

1) Clone the repository:
    git clone <repository_url>
    cd <repository_directory>
2) Ensure you have the required libraries installed.
3) Run the Python script to train the model and make predictions:
     python Filename
4) The script will display a plot showing the original test stock prices and the model's predicted prices.
