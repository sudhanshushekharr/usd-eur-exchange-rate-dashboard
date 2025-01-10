import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned exchange rate data
exchange_rate_data = pd.read_csv('cleaned_usd_eur_exchange_rate.csv', parse_dates=['timestamp'], index_col='timestamp')

# Step 1: Calculate Daily Returns
exchange_rate_data['Daily_Return'] = exchange_rate_data['Close'].pct_change()

# Step 2: Resample Data to Monthly Frequency
# Calculate the average daily return for each month
monthly_returns = exchange_rate_data['Daily_Return'].resample('M').mean()

# Step 3: Prepare Data for Heatmap
# Create a DataFrame with Year, Month, and Average Daily Returns
heatmap_data = pd.DataFrame({
    'Year': monthly_returns.index.year,
    'Month': monthly_returns.index.month_name(),
    'Returns': monthly_returns.values
})

# Pivot the data to create a 2D table for the heatmap
heatmap_pivot = heatmap_data.pivot_table(index='Year', columns='Month', values='Returns', aggfunc='mean')

# Ensure the months are in the correct order
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]
heatmap_pivot = heatmap_pivot[month_order]

# Step 4: Plot the Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_pivot, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Average Daily Return'})
plt.title('Monthly Average Daily Returns Heatmap', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.tight_layout()
plt.show()