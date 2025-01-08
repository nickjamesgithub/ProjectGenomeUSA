import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates

matplotlib.use('TkAgg')


# Global parameters
window = 90
regime_l1_percentile = 20
regime_l2_percentile = 80

# Import mapping data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\company_list_asx200_.csv")
# Define the directory containing the CSV files
directory = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Capiq_data\share_price"
# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(directory, "*.csv"))
# Initialize an empty list to store DataFrames
dataframes = []

# Iterate over the list of CSV files and read each one into a DataFrame
for file in csv_files:
    # Extract the ticker from the filename (assuming the structure is 'prefix_ticker_suffix.csv')
    filename = os.path.basename(file)
    ticker = filename.split('_')[1]
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    # Add a new column with the ticker
    df['Ticker'] = ticker
    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all the DataFrames into a single DataFrame
combined_price_df = pd.concat(dataframes, ignore_index=True)
# Convert 'Date' to datetime for proper sorting
combined_price_df['Date'] = pd.to_datetime(combined_price_df['Date'])
# Extract the unique dates across all dataframes
unique_dates = pd.DataFrame(combined_price_df['Date'].unique(), columns=['Date'])
# Initialize an empty list to store reindexed DataFrames
reindexed_dataframes = []

# Iterate over each ticker's DataFrame to reindex on the unique date axis and backfill
for ticker in combined_price_df['Ticker'].unique():
    ticker_df = combined_price_df[combined_price_df['Ticker'] == ticker]
    ticker_df = unique_dates.merge(ticker_df, on='Date', how='left').sort_values('Date')
    ticker_df = ticker_df.ffill().bfill()  # Fill forward then backward to ensure no missing data
    reindexed_dataframes.append(ticker_df)

# Concatenate all the reindexed DataFrames into a single DataFrame
aligned_price_df = pd.concat(reindexed_dataframes, ignore_index=True)

# Merge the aligned DataFrame with the mapping data to add the sector information
aligned_price_df = pd.merge(aligned_price_df, mapping_data[['Ticker', 'Sector_new']], on='Ticker', how='left')

# Pivot the DataFrame to have prices running down the rows and tickers across the columns
pivot_df = aligned_price_df.pivot(index='Date', columns='Ticker', values='Price')
log_returns_market = (np.log(pivot_df) - np.log(pivot_df.shift(1))).fillna(0)
avg_correlation_market_list = []
for j in range(window, len(log_returns_market)):
    print("Market Iteration ", log_returns_market.index[j])  # Updated to log_returns_market.index
    returns_slice = log_returns_market.iloc[j - window:j]
    # Drop company if it has insufficient variability/trading data in that window
    returns_slice_clean = returns_slice.loc[:, (returns_slice != 0.0).any(axis=0)]
    # Compute correlation matrix
    correlation_matrix_j = returns_slice_clean.corr()
    avg_correlation = np.diag(correlation_matrix_j, k=1).mean()
    avg_correlation_market_list.append(avg_correlation)

# Create a list of dates for the x-axis
dates_list = log_returns_market.index[window:]

# Create the plot
fig, ax = plt.subplots()

# Plot average correlation by sector over time
ax.plot(dates_list, avg_correlation_market_list, color='black')

# Shade regions based on y-values
ax.fill_between(dates_list, 0, np.percentile(avg_correlation_market_list, regime_l1_percentile), color='green', alpha=0.2)
ax.fill_between(dates_list, np.percentile(avg_correlation_market_list, regime_l1_percentile), np.percentile(avg_correlation_market_list, regime_l2_percentile), color='blue', alpha=0.2)
ax.fill_between(dates_list, np.percentile(avg_correlation_market_list, regime_l2_percentile), max(avg_correlation_market_list), color='red', alpha=0.2)

# Add the horizontal lines for reference
ax.axhline(y=np.percentile(avg_correlation_market_list, regime_l1_percentile), color='gray', linestyle='--')
ax.axhline(y=np.percentile(avg_correlation_market_list, regime_l2_percentile), color='gray', linestyle='--')

# Set labels and title
ax.set_ylabel("Avg correlation")
ax.set_xlabel("Date")
ax.set_title("Market correlation over time")

# Format the x-axis to display dates
fig.autofmt_xdate()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.show()

# Calculate log returns for each sector
sectors_list = []
dates_list = []
avg_correlation_vector_list = []
regime_classification_vector_list = []
log_returns_sector_vector_list = []

# Loop over sectors
for sector in aligned_price_df["Sector_new"].dropna().unique():
    # Append sector label to sectors_list
    sectors_list.append(sector)
    dates_list.append(pivot_df.index[window:])

    # Slice sector and get log returns for the specific sector
    sector_df = aligned_price_df[aligned_price_df["Sector_new"] == sector]
    pivot_df_sector = sector_df.pivot(index='Date', columns='Ticker', values='Price')
    log_returns_sector = (np.log(pivot_df_sector) - np.log(pivot_df_sector.shift(1))).fillna(0)

    avg_correlation_sector_list = []
    regime_classification_sector_list = []

    for j in range(window, len(log_returns_sector)):
        print("Sector " + sector + " Iteration ", aligned_price_df.index[j])
        returns_slice = log_returns_sector.iloc[j - window:j]
        # Compute average log returns over the trailing window
        avg_log_returns = returns_slice.iloc[:,:].mean().mean()
        # Drop company if it has insufficient variability/trading data in that window
        returns_slice_clean = returns_slice.loc[:, (returns_slice != 0.0).any(axis=0)]
        # Compute correlation matrix
        correlation_matrix_j = returns_slice_clean.corr()
        avg_correlation = np.diag(correlation_matrix_j, k=1).mean()
        # Append average correlation, regime classification and average returns
        avg_correlation_sector_list.append(avg_correlation)

    # Append vector of average correlations to master list
    avg_correlation_vector_list.append(avg_correlation_sector_list)
    regime_classification_vector_list.append(regime_classification_sector_list)

# Plot average correlation by sector over time, skipping empty lists
for i in range(len(avg_correlation_vector_list)):
    if avg_correlation_vector_list[i]:  # Check if the list is not empty
        plt.plot(dates_list[i], avg_correlation_vector_list[i], label=sectors_list[i])

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Average Correlation')
plt.title('Average Correlation by Sector Over Time')
# Format the x-axis to display dates
plt.gcf().autofmt_xdate()
plt.legend()
plt.show()