import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned exchange rate data for USD/EUR
exchange_rate_data = pd.read_csv('cleaned_usd_eur_exchange_rate.csv', parse_dates=['timestamp'], index_col='timestamp')

# Step 1: Calculate Bollinger Bands
window = 20  # 20-day rolling window
exchange_rate_data['SMA'] = exchange_rate_data['Close'].rolling(window=window).mean()  # Middle Band (SMA)
exchange_rate_data['Std'] = exchange_rate_data['Close'].rolling(window=window).std()   # Standard Deviation
exchange_rate_data['Upper_Band'] = exchange_rate_data['SMA'] + 2 * exchange_rate_data['Std']  # Upper Band
exchange_rate_data['Lower_Band'] = exchange_rate_data['SMA'] - 2 * exchange_rate_data['Std']  # Lower Band

# Step 2: Plot Bollinger Bands
plt.figure(figsize=(14, 8))
plt.plot(exchange_rate_data.index, exchange_rate_data['Close'], label='USD/EUR Exchange Rate', color='blue', linewidth=1.5)
plt.plot(exchange_rate_data.index, exchange_rate_data['SMA'], label=f'{window}-Day SMA', color='orange', linewidth=1.5)
plt.plot(exchange_rate_data.index, exchange_rate_data['Upper_Band'], label='Upper Band', color='green', linestyle='--', linewidth=1.5)
plt.plot(exchange_rate_data.index, exchange_rate_data['Lower_Band'], label='Lower Band', color='red', linestyle='--', linewidth=1.5)
plt.fill_between(exchange_rate_data.index, exchange_rate_data['Lower_Band'], exchange_rate_data['Upper_Band'], color='gray', alpha=0.2)
plt.title('Bollinger Bands for USD/EUR Exchange Rate', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Exchange Rate', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()