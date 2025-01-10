# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load the Cleaned Exchange Rate Data
# Ensure the cleaned data file is in the same directory as this script, or provide the full path.
exchange_rate_data = pd.read_csv('cleaned_usd_eur_exchange_rate.csv', parse_dates=['timestamp'], index_col='timestamp')

# Step 3: Verify the Data
# Display the first few rows of the data to ensure it's loaded correctly.
print("First few rows of the data:")
print(exchange_rate_data.head())

# Step 4: Set Up the Plot
# Create a figure and axis with a specific size (10x6 inches).
plt.figure(figsize=(10, 6))

# Step 5: Plot the Closing Exchange Rate
# Use the 'Close' column for the exchange rate values.
plt.plot(
    exchange_rate_data.index,  # X-axis: Dates (index of the DataFrame)
    exchange_rate_data['Close'],  # Y-axis: Closing exchange rate values
    label='USD/EUR Exchange Rate',  # Label for the legend
    color='blue',  # Line color
    linewidth=2  # Line thickness
)

# Step 6: Customize the Plot
# Add a title and axis labels.
plt.title('USD/EUR Exchange Rate Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Exchange Rate', fontsize=12)

# Add a legend to explain the line.
plt.legend(loc='upper right', fontsize=12)

# Add grid lines for better readability.
plt.grid(True, linestyle='--', alpha=0.6)

# Step 7: Display the Plot
plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()

# Step 8: (Optional) Save the Plot as an Image
# Uncomment the following line to save the plot as a PNG file.
# plt.savefig('usd_eur_exchange_rate_plot.png', dpi=300, bbox_inches='tight')