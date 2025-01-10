import requests
import pandas as pd

# Replace with your Alpha Vantage API key
API_KEY = '66V1B99PR95WW1O9'

# API endpoint for USD/EUR exchange rate (daily)
url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=EUR&apikey={API_KEY}&datatype=csv'

# Make the API request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Save the data to a CSV file
    with open('usd_eur_exchange_rate.csv', 'wb') as file:
        file.write(response.content)
    print("Data downloaded successfully!")
else:
    print(f"Failed to download data. Status code: {response.status_code}")

# Load the CSV file into a pandas DataFrame
exchange_rate_data = pd.read_csv('usd_eur_exchange_rate.csv')

# Display the first few rows of the data
print(exchange_rate_data.head())