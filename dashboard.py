import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
from io import StringIO

# Set the title of the dashboard
st.title('USD/EUR Exchange Rate Dashboard')
with st.expander("About this dashboard"):
    st.write("""
    This dashboard provides insights into the USD/EUR exchange rate, including historical trends, technical indicators, and machine learning-based forecasts.
    """)

# Load the cleaned exchange rate data
@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_usd_eur_exchange_rate.csv', parse_dates=['timestamp'], index_col='timestamp')
    return data

exchange_rate_data = load_data()

# Feature Engineering
exchange_rate_data['Lag_1'] = exchange_rate_data['Close'].shift(1)  # Lagged value (1 day)
exchange_rate_data['Lag_2'] = exchange_rate_data['Close'].shift(2)  # Lagged value (2 days)
exchange_rate_data.dropna(inplace=True)  # Drop rows with missing values

# Train models
@st.cache_data
def train_models(X_train, y_train):
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    return lr_model, rf_model

# Prepare features and target
X = exchange_rate_data[['Lag_1', 'Lag_2']]  # Features (lagged values)
y = exchange_rate_data['Close']  # Target (exchange rate)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr_model, rf_model = train_models(X_train, y_train)

# Display the dataset
st.write("### Dataset Preview")
with st.expander("What is this?"):
    st.write("""
    This section shows the first few rows of the dataset, including the exchange rate and engineered features (lagged values).
    """)
st.write(exchange_rate_data.head())

# Add a section for the exchange rate over time
st.write("### Exchange Rate Over Time")
with st.expander("What is this?"):
    st.write("""
    This plot shows the historical trend of the USD/EUR exchange rate over time.
    """)

plt.figure(figsize=(12, 6))  # Increase figure size for better spacing
plt.plot(exchange_rate_data.index, exchange_rate_data['Close'], label='USD/EUR Exchange Rate', color='blue')
plt.title('USD/EUR Exchange Rate Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Exchange Rate', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# Rotate x-axis labels and adjust spacing
plt.xticks(rotation=45, ha='right')  # Rotate labels by 45 degrees and align them to the right
plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding

st.pyplot(plt)


# Add a candlestick chart
st.write("### Candlestick Chart")
with st.expander("What is this?"):
    st.write("""
    This candlestick chart visualizes the open, high, low, and close prices of the USD/EUR exchange rate.
    """)
candlestick_data = exchange_rate_data[['Open', 'High', 'Low', 'Close']]

fig = go.Figure(data=[go.Candlestick(
    x=candlestick_data.index,
    open=candlestick_data['Open'],
    high=candlestick_data['High'],
    low=candlestick_data['Low'],
    close=candlestick_data['Close'],
    name='USD/EUR'
)])

fig.update_layout(
    title='USD/EUR Candlestick Chart',
    xaxis_title='Date',
    yaxis_title='Exchange Rate',
    template='plotly_dark',
    xaxis_rangeslider_visible=True
)

st.plotly_chart(fig, use_container_width=True)

# Add a date range filter
st.write("### Filter by Date Range")
with st.expander("What is this?"):
    st.write("""
    Use the date pickers below to filter the dataset and visualizations for a specific time period.
    """)
start_date = st.date_input('Start Date', exchange_rate_data.index.min())
end_date = st.date_input('End Date', exchange_rate_data.index.max())

# Filter the data based on the selected date range
filtered_data = exchange_rate_data[(exchange_rate_data.index >= pd.to_datetime(start_date)) & 
                                  (exchange_rate_data.index <= pd.to_datetime(end_date))]

# Display the filtered exchange rate over time
st.write("### Filtered Exchange Rate Over Time")
with st.expander("What is this?"):
    st.write("""
    This plot shows the USD/EUR exchange rate for the selected date range.
    """)
plt.figure(figsize=(10, 6))
plt.plot(filtered_data.index, filtered_data['Close'], label='USD/EUR Exchange Rate', color='blue')
plt.title('Filtered USD/EUR Exchange Rate Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Exchange Rate', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
st.pyplot(plt)

# Add a section for Bollinger Bands
st.write("### Bollinger Bands")
with st.expander("What is this?"):
    st.write("""
    Bollinger Bands are a technical indicator that shows the volatility and potential price levels of the exchange rate.
    """)
window = st.slider('Select Rolling Window for Bollinger Bands', min_value=10, max_value=50, value=20)

# Calculate Bollinger Bands
filtered_data['SMA'] = filtered_data['Close'].rolling(window=window).mean()
filtered_data['Std'] = filtered_data['Close'].rolling(window=window).std()
filtered_data['Upper_Band'] = filtered_data['SMA'] + 2 * filtered_data['Std']
filtered_data['Lower_Band'] = filtered_data['SMA'] - 2 * filtered_data['Std']

# Plot Bollinger Bands
plt.figure(figsize=(10, 6))
plt.plot(filtered_data.index, filtered_data['Close'], label='USD/EUR Exchange Rate', color='blue')
plt.plot(filtered_data.index, filtered_data['SMA'], label=f'{window}-Day SMA', color='orange')
plt.plot(filtered_data.index, filtered_data['Upper_Band'], label='Upper Band', color='green', linestyle='--')
plt.plot(filtered_data.index, filtered_data['Lower_Band'], label='Lower Band', color='red', linestyle='--')
plt.fill_between(filtered_data.index, filtered_data['Lower_Band'], filtered_data['Upper_Band'], color='gray', alpha=0.2)
plt.title('Bollinger Bands for USD/EUR Exchange Rate', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Exchange Rate', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
st.pyplot(plt)

# Add a heatmap of daily returns
st.write("### Heatmap of Daily Returns")
with st.expander("What is this?"):
    st.write("""
    This heatmap shows the average daily returns of the USD/EUR exchange rate by month and year.
    """)
filtered_data['Daily_Return'] = filtered_data['Close'].pct_change()

# Scale daily returns to percentages
filtered_data['Daily_Return'] = filtered_data['Daily_Return'] * 100

# Resample data to monthly frequency and calculate average daily returns
monthly_returns = filtered_data['Daily_Return'].resample('ME').mean()

# Create a pivot table for the heatmap
heatmap_data = pd.DataFrame({
    'Year': monthly_returns.index.year,
    'Month': monthly_returns.index.month_name(),
    'Returns': monthly_returns.values
})

heatmap_pivot = heatmap_data.pivot_table(index='Year', columns='Month', values='Returns', aggfunc='mean')

# Ensure the months are in the correct order
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]
month_order = [month for month in month_order if month in heatmap_pivot.columns]
heatmap_pivot = heatmap_pivot[month_order]

# Plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(
    heatmap_pivot, 
    cmap='coolwarm', 
    annot=True, 
    fmt=".2f",  # Display 2 decimal places for percentages
    linewidths=0.5, 
    cbar_kws={'label': 'Average Daily Return (%)'}
)
plt.title('Monthly Average Daily Returns Heatmap', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Year', fontsize=12)
st.pyplot(plt)

# Add a rolling autocorrelation plot
st.write("### Rolling Autocorrelation Plot")
with st.expander("What is this?"):
    st.write("""
    This plot shows the rolling autocorrelation of the USD/EUR exchange rate, which helps identify patterns or trends.
    """)
autocorr_window = st.slider('Select Rolling Window for Autocorrelation', min_value=5, max_value=50, value=30)

# Calculate rolling autocorrelation
autocorr = filtered_data['Close'].rolling(window=autocorr_window).apply(lambda x: x.autocorr(), raw=False)

# Plot the rolling autocorrelation
plt.figure(figsize=(10, 6))
plt.plot(autocorr.index, autocorr, label=f'{autocorr_window}-Day Rolling Autocorrelation', color='purple')
plt.title('Rolling Autocorrelation of USD/EUR Exchange Rate', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Autocorrelation', fontsize=12)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
st.pyplot(plt)

# Add a section for exchange rate forecasting
st.write("### Exchange Rate Forecasting")
with st.expander("What is this?"):
    st.write("""
    This section allows you to predict future USD/EUR exchange rates using machine learning models.
    """)

# Model selection
model_option = st.selectbox('Select Model', ['Linear Regression', 'Random Forest'])
model = lr_model if model_option == 'Linear Regression' else rf_model

# Historical accuracy
y_pred_backtest = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_backtest)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_backtest))
st.write("### Model Performance")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Input for the number of days to forecast
forecast_days = st.slider('Select Number of Days to Forecast', min_value=1, max_value=30, value=7)

# Generate predictions
if st.button('Predict'):
    last_known = exchange_rate_data['Close'].iloc[-1]  # Last known exchange rate
    predictions = []
    confidence_intervals = []

    for i in range(forecast_days):
        # Predict the next day's exchange rate
        prediction = model.predict([[last_known, exchange_rate_data['Close'].iloc[-2]]])
        predictions.append(prediction[0])

        # Calculate confidence interval
        residuals = y_test - model.predict(X_test)
        std_error = np.std(residuals)
        confidence_intervals.append(1.96 * std_error)  # 1.96 for 95% confidence

        last_known = prediction[0]  # Update the last known value

    # Display the predictions
    st.write("### Predicted Exchange Rates with 95% Confidence Intervals")
    for i, (pred, ci) in enumerate(zip(predictions, confidence_intervals)):
        st.write(f"Day {i+1}: {pred:.4f} ± {ci:.4f}")

    # Plot the predictions
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, forecast_days + 1), predictions, label='Predicted Exchange Rate', color='green', marker='o')
    plt.fill_between(
        range(1, forecast_days + 1),
        [pred - ci for pred, ci in zip(predictions, confidence_intervals)],
        [pred + ci for pred, ci in zip(predictions, confidence_intervals)],
        color='lightgreen', alpha=0.3, label='95% Confidence Interval'
    )
    plt.title('Predicted USD/EUR Exchange Rates', fontsize=16)
    plt.xlabel('Days Ahead', fontsize=12)
    plt.ylabel('Exchange Rate', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(plt)

    # Download predictions as CSV
    predictions_df = pd.DataFrame({
        'Day': range(1, forecast_days + 1),
        'Predicted Exchange Rate': predictions,
        'Confidence Interval': confidence_intervals
    })
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='exchange_rate_predictions.csv',
        mime='text/csv'
    )

   