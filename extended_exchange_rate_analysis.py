import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned exchange rate data
exchange_rate_data = pd.read_csv('cleaned_usd_eur_exchange_rate.csv', parse_dates=['timestamp'], index_col='timestamp')

# Step 1: Calculate Moving Averages
exchange_rate_data['SMA_7'] = exchange_rate_data['Close'].rolling(window=7).mean()  # 7-day simple moving average
exchange_rate_data['SMA_30'] = exchange_rate_data['Close'].rolling(window=30).mean()  # 30-day simple moving average

# Step 2: Calculate Daily Returns and Volatility
exchange_rate_data['Daily_Return'] = exchange_rate_data['Close'].pct_change()  # Daily percentage change
exchange_rate_data['Volatility'] = exchange_rate_data['Daily_Return'].rolling(window=30).std()  # 30-day rolling volatility

# Step 3: Create the Plot
plt.figure(figsize=(14, 10))

# Subplot 1: Exchange Rate and Moving Averages
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
plt.plot(exchange_rate_data.index, exchange_rate_data['Close'], label='USD/EUR Exchange Rate', color='blue', linewidth=1.5)
plt.plot(exchange_rate_data.index, exchange_rate_data['SMA_7'], label='7-Day SMA', color='orange', linestyle='--', linewidth=1)
plt.plot(exchange_rate_data.index, exchange_rate_data['SMA_30'], label='30-Day SMA', color='green', linestyle='--', linewidth=1)
plt.title('USD/EUR Exchange Rate with Moving Averages', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Exchange Rate', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# Step 4: Annotate Key Events
# Define event dates
brexit_date = pd.to_datetime('2016-06-23')  # Brexit date
covid_date = pd.to_datetime('2020-03-01')   # COVID-19 date

# Find the closest dates in the dataset
brexit_closest_date = exchange_rate_data.index[exchange_rate_data.index.get_loc(brexit_date, method='nearest')]
covid_closest_date = exchange_rate_data.index[exchange_rate_data.index.get_loc(covid_date, method='nearest')]

# Get the exchange rate values for the closest dates
brexit_rate = exchange_rate_data.loc[brexit_closest_date, 'Close']
covid_rate = exchange_rate_data.loc[covid_closest_date, 'Close']

# Annotate Brexit
plt.annotate(
    'Brexit Vote', 
    xy=(brexit_closest_date, brexit_rate),  # Point to annotate
    xytext=(brexit_closest_date, brexit_rate - 0.02),  # Text position
    arrowprops=dict(facecolor='red', shrink=0.05),  # Arrow properties
    fontsize=10, 
    color='red'
)

# Annotate COVID-19
plt.annotate(
    'COVID-19 Pandemic', 
    xy=(covid_closest_date, covid_rate),  # Point to annotate
    xytext=(covid_closest_date, covid_rate - 0.02),  # Text position
    arrowprops=dict(facecolor='red', shrink=0.05),  # Arrow properties
    fontsize=10, 
    color='red'
)

# Subplot 2: Volatility
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
plt.plot(exchange_rate_data.index, exchange_rate_data['Volatility'], label='30-Day Volatility', color='purple', linewidth=1.5)
plt.title('USD/EUR Exchange Rate Volatility', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()