import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import yfinance as yf

def predict_stock_price(stock_name, ahead, d):
    stock = yf.Ticker(stock_name)
    hist = stock.history(period="5y")
    df = hist

    n = int(hist.shape[0] * 0.8)
    training_set = df.iloc[:n, 1:2].values
    test_set = df.iloc[n:, 1:2].values

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(d, n - ahead):
        X_train.append(training_set_scaled[i - d:i, 0])
        y_train.append(training_set_scaled[i + ahead, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    dataset_train = df.iloc[:n, 1:2]
    dataset_test = df.iloc[n:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis=0)

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - d:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(d, inputs.shape[0]):
        X_test.append(inputs[i - d:i, 0])
    X_test = np.array(X_test)

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = predicted_stock_price.reshape(-1, 1)  # Reshape predicted_stock_price to 2D
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    df['Date'] = df.index
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots()
    ax.plot(df.loc[n:, 'Date'], dataset_test.values, color='red', label='Actual Price')
    ax.plot(df.loc[n:, 'Date'], predicted_stock_price, color='blue', label='Predicted Price')
    ax.set_title('Stock Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    plt.xticks(rotation=90)

    return fig

def main():
    st.title("Stock Price Prediction")
    stock_name = st.text_input("Enter stock name (e.g., AAPL, GOOG, TSLA)")
    ahead = st.number_input("Enter number of days ahead to predict, you can predict upto 1 month", min_value=1, value=30, step=1)
    d = st.number_input("Enter number of previous days to consider, you can take stock upto 1 year", min_value=1, value=365, step=1)

    if st.button("Predict"):
        if stock_name:
            fig = predict_stock_price(stock_name, ahead, d)
            st.pyplot(fig)
        else:
            st.warning("Please enter a stock name.")

if __name__ == "__main__":
    main()
