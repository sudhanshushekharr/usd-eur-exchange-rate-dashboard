import pandas as pd
import plotly.graph_objects as go

# Load the cleaned exchange rate data
exchange_rate_data = pd.read_csv('cleaned_usd_eur_exchange_rate.csv', parse_dates=['timestamp'], index_col='timestamp')

# Ensure the data has 'Open', 'High', 'Low', and 'Close' columns
if not all(col in exchange_rate_data.columns for col in ['Open', 'High', 'Low', 'Close']):
    raise ValueError("The dataset must contain 'Open', 'High', 'Low', and 'Close' columns.")

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=exchange_rate_data.index,  # Date
    open=exchange_rate_data['Open'],  # Opening price
    high=exchange_rate_data['High'],  # Highest price
    low=exchange_rate_data['Low'],  # Lowest price
    close=exchange_rate_data['Close'],  # Closing price
    name='USD/EUR'
)])

# Customize the layout
fig.update_layout(
    title='USD/EUR Exchange Rate Candlestick Chart',
    xaxis_title='Date',
    yaxis_title='Exchange Rate',
    template='plotly_dark',  # Use a dark theme for better visuals
    xaxis_rangeslider_visible=True  # Add a range slider for zooming
)

# Show the chart
fig.show()