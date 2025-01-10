import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned exchange rate data for USD/EUR
exchange_rate_data = pd.read_csv('cleaned_usd_eur_exchange_rate.csv', parse_dates=['timestamp'], index_col='timestamp')

# Step 1: Calculate Autocorrelation
# Use a 30-day rolling window to calculate autocorrelation
autocorr = exchange_rate_data['Close'].rolling(window=30).apply(lambda x: x.autocorr(), raw=False)

# Step 2: Plot the Autocorrelation
plt.figure(figsize=(12, 6))
plt.plot(autocorr.index, autocorr, label='30-Day Rolling Autocorrelation (USD/EUR)', color='blue', linewidth=2)
plt.title('Rolling Autocorrelation of USD/EUR Exchange Rate', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Autocorrelation', fontsize=12)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Add a horizontal line at 0
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()