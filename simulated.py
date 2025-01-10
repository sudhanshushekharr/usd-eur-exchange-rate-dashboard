import pandas as pd
import numpy as np

# Load the cleaned exchange rate data for USD/EUR
exchange_rate_data = pd.read_csv('cleaned_usd_eur_exchange_rate.csv', parse_dates=['timestamp'], index_col='timestamp')

# Simulate macroeconomic data
np.random.seed(42)  # For reproducibility
exchange_rate_data['Interest_Rate'] = np.random.normal(loc=2.0, scale=0.5, size=len(exchange_rate_data))  # Interest Rate (%)
exchange_rate_data['GDP_Growth'] = np.random.normal(loc=3.0, scale=1.0, size=len(exchange_rate_data))      # GDP Growth (%)
exchange_rate_data['Inflation'] = np.random.normal(loc=1.5, scale=0.3, size=len(exchange_rate_data))       # Inflation (%)

# Display the first few rows of the dataset
print(exchange_rate_data.head())