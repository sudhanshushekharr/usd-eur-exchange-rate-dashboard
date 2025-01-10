import pandas as pd

# Load the raw exchange rate data
exchange_rate_data = pd.read_csv('usd_eur_exchange_rate.csv')

# Convert 'timestamp' to datetime
exchange_rate_data['timestamp'] = pd.to_datetime(exchange_rate_data['timestamp'])

# Set 'timestamp' as the index
exchange_rate_data.set_index('timestamp', inplace=True)

# Rename columns for clarity
exchange_rate_data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close'
}, inplace=True)

# Display the cleaned data
print(exchange_rate_data.head())

# (Optional) Save the cleaned data to a new CSV file
exchange_rate_data.to_csv('cleaned_usd_eur_exchange_rate.csv')