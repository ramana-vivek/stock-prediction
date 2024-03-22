import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf

# Load the pre-trained model
model = load_model('keras_model2.h5')

def predict_stock_prices(stock_name, days, ahead):
    stock = yf.Ticker(stock_name)
    hist = stock.history(period="5y")

    df = hist.reset_index()  # Reset the index to create a new 'Date' column
    df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime

    n = int(hist.shape[0] * 0.8)
    training_set = df.iloc[:n, 1:2].values
    test_set = df.iloc[n:, 1:2].values

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(days, n - ahead):
        X_train.append(training_set_scaled[i - days:i, 0])
        y_train.append(training_set_scaled[i + ahead, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    dataset_train = df.iloc[:n, 1:2]
    dataset_test = df.iloc[n:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - days:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(days, inputs.shape[0]):
        X_test.append(inputs[i - days:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    return df, dataset_test.values, predicted_stock_price

def main():
    st.title("Stock Price Prediction")

    stock_name = st.text_input("Enter Stock Ticker Symbol", value='SBIN.NS')
    days = st.number_input("Enter Number of Days", min_value=1, value=1000)
    ahead = st.number_input("Enter Number of Days Ahead", min_value=1, value=1)

    if st.button("Predict"):
        df, actual_prices, predicted_prices = predict_stock_prices(stock_name, days, ahead)
        n = int(df.shape[0] * 0.8)

        # Plot actual and predicted prices
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.loc[n:, 'Date'], actual_prices, color='red', label='Actual Price')
        ax.plot(df.loc[n:, 'Date'], predicted_prices, color='blue', label='Predicted Price')
        ax.set_title(f'Stock Price Prediction for {stock_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Display the predicted price for the next day
        last_date = df.loc[n:, 'Date'].iloc[-1]
        last_price = actual_prices[-1].item()
        next_day_price = predicted_prices[-1].item()
        st.write(f"Last Date: {last_date.strftime('%Y-%m-%d')}")
        st.write(f"Last Price: {last_price:.2f}")
        st.write(f"Predicted Price for Next Day: {next_day_price:.2f}")

if __name__ == "__main__":
    main()
