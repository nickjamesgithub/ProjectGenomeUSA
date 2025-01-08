import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Global parameters
window = 250

# Define autocorrelation function
def autocorrelation_function(returns, window):
    up_days = len([x for x in returns[-window:] if x > 0])
    down_days = len([x for x in returns[-window:] if x <= 0])
    return down_days, up_days

def downside_risk(returns, risk_free=0):
    adj_returns = returns - risk_free
    sqr_downside = np.square(np.clip(adj_returns, np.NINF, 0))
    return np.sqrt(np.nanmean(sqr_downside) * 252)

def sortino(returns, risk_free=0):
    adj_returns = returns - risk_free
    drisk = downside_risk(adj_returns)

    if drisk == 0:
        return np.nan

    return (np.nanmean(adj_returns) * np.sqrt(252)) \
        / drisk

def generate_genome_classification_df(df):
    # Conditions EP/FE
    conditions_genome = [
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"] < 0),
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"] >= 0.20),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"] < 0),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"] >= 0.20)
    ]

    # Values to display
    values_genome = ["UNTENABLE", "TRAPPED", "BRAVE", "FEARLESS", "CHALLENGED", "VIRTUOUS", "FAMOUS", "LEGENDARY"]

    df["Genome_classification"] = np.select(conditions_genome, values_genome)

    return df

def merge_csv_files_with_ticker(directory_path):
    # List to hold the DataFrames
    dataframes = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a CSV
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Extract the ticker from the filename (everything after and before the two "_")
            parts = filename.split('_')
            if len(parts) > 2:
                ticker = parts[1]  # Assuming the ticker is always the second part
            else:
                ticker = parts[0].replace('.csv', '')  # Fallback in case the format is not as expected
            # Add the ticker as a new column
            df['Ticker'] = ticker
            # Append the DataFrame to the list
            dataframes.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def merge_csv_files(directory_path):
    # List to hold the DataFrames
    dataframes = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a CSV
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Append the DataFrame to the list
            dataframes.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# Import mapping data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")
# full_ticker_list = ticker_list = mapping_data["Full Ticker"].values

# Example usage
directory_path_prices = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\share_price"
directory_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data"

# Prices dataframe merged with tickers
prices_df = merge_csv_files_with_ticker(directory_path_prices)
df = merge_csv_files(directory_path)

# Append Genome data to financial dataframe
df = generate_genome_classification_df(df)

# Loop over tickers and find tickers that satisfy criteria
unique_tickers = prices_df["Ticker"].unique()
features_list = []
for i in range(len(unique_tickers)):
    # Iteration i
    print("Iteration ", i, " ", unique_tickers[i])
    try:
        # Slice financial data for company i
        df["Ticker"] = df["Ticker"].astype(str)
        company_i = df.loc[df["Ticker"]==unique_tickers[i]]
        genome_class_i_trailing_2 = company_i.loc[company_i["Year"] == 2022]["Genome_classification"].iloc[0]
        genome_class_i_trailing = company_i.loc[company_i["Year"] == 2023]["Genome_classification"].iloc[0]
        genome_class_i_current = company_i.loc[company_i["Year"]==2024]["Genome_classification"].iloc[0]
        company_name_i = company_i.loc[company_i["Year"]==2024]["Company_name"].iloc[0]

        # Get the last 1 year of trading data
        slice_i = prices_df.loc[prices_df["Ticker"]==unique_tickers[i]].iloc[-window:,1:]
        slice_i["Log_returns"] = np.log(slice_i["Price"]) - np.log(slice_i["Price"].shift(1))
        log_returns_i = slice_i["Log_returns"]

        # Compute autocorrelation function on log returns
        downs, ups = autocorrelation_function(log_returns_i, window)
        down_ratio = np.nan_to_num(downs / ups)

        # Compute total returns
        total_returns_window = np.nan_to_num(np.sum(log_returns_i))
        # Compute volatility
        volatility_window = np.nan_to_num(np.std(log_returns_i))
        # Compute Sharpe Ratio
        sharpe = np.nan_to_num(total_returns_window / volatility_window)
        # Sortino Ratio
        sortino_r = sortino(log_returns_i, 0)
        # Append features to feature list
        features_list.append([company_name_i, unique_tickers[i], genome_class_i_current, genome_class_i_trailing, genome_class_i_trailing_2,
                              total_returns_window, down_ratio, sharpe, sortino_r])

    except:
        print("Error with company ", unique_tickers[i])

# Make a dataframe and append columns
features_df = pd.DataFrame(features_list)
features_df.columns = ["Company_name", "Ticker", "Genome_segment_24", "Genome_segment_23", "Genome_segment_22", "Total_returns", "Down_ratio", "Sharpe_ratio", "Sortino_ratio"]
features_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Market_updates\USA_distressed_flag.csv")

