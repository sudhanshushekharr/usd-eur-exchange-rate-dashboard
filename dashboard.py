import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Set the title of the dashboard
st.title('USD/EUR Exchange Rate Dashboard')

# Load the cleaned exchange rate data
exchange_rate_data = pd.read_csv('cleaned_usd_eur_exchange_rate.csv', parse_dates=['timestamp'], index_col='timestamp')

# Display the dataset
st.write("### Dataset Preview")
st.write(exchange_rate_data.head())

# Add a section for the exchange rate over time
st.write("### Exchange Rate Over Time")
plt.figure(figsize=(10, 6))
plt.plot(exchange_rate_data.index, exchange_rate_data['Close'], label='USD/EUR Exchange Rate', color='blue')
plt.title('USD/EUR Exchange Rate Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Exchange Rate', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
st.pyplot(plt)

# Add a candlestick chart
st.write("### Candlestick Chart")
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
start_date = st.date_input('Start Date', exchange_rate_data.index.min())
end_date = st.date_input('End Date', exchange_rate_data.index.max())

# Filter the data based on the selected date range
filtered_data = exchange_rate_data[(exchange_rate_data.index >= pd.to_datetime(start_date)) & 
                                  (exchange_rate_data.index <= pd.to_datetime(end_date))]

# Display the filtered exchange rate over time
st.write("### Filtered Exchange Rate Over Time")
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
autocorr_window = st.slider('Select Rolling Window for Autocorrelation', min_value=10, max_value=100, value=30)

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