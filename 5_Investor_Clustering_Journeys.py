import numpy as np
import pandas as pd
from Utilities import compute_percentiles, firefly_plot, geometric_return
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from itertools import combinations, product

matplotlib.use('TkAgg')

# Read the data
df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

# Desired sectors and date range
country_list = ["USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'] # "USA", 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
unique_sectors = df_full["Sector"].unique()
desired_sectors = unique_sectors
company_ticker = "ASX:SGH"
start_year = 2017
end_year = 2024

# 1. Filter by Country & Sector
df_slice = df_full[(df_full['Country'].isin(country_list)) & (df_full["Sector"]).isin(unique_sectors)]
tickers = df_slice["Ticker_full"].unique()

# Define Genome encoding
genome_encoding = {
    "UNTENABLE": (-1, -1),
    "CHALLENGED": (-1, 1),
    "TRAPPED": (0, -1),
    "VIRTUOUS": (0, 1),
    "BRAVE": (1, -1),
    "FAMOUS": (1, 1),
    "FEARLESS": (2, -1),
    "LEGENDARY": (2, 1)
}

# Encode X and Y Values
df_slice["Genome_encoding_x"], df_slice["Genome_encoding_y"] = zip(*df_slice["Genome_classification_bespoke"].map(lambda x: genome_encoding.get(x, (np.nan, np.nan))))

# Company slice
target_genome_slice = df_slice.loc[(df_slice["Year"]>=start_year) & (df_slice["Year"]<=end_year) & (df_slice["Ticker_full"]==company_ticker)][["Genome_encoding_x", "Genome_encoding_y"]]


# Function to compute L1 distance
def l1_distance(df1, df2):
    return np.nansum(np.abs(df1.values - df2.values))


# Initialize list to store results
distance_results = []

# Define the fixed anchor window length
time_window_length = end_year - start_year + 1  # Inclusive range

# Get anchor company's available years
anchor_years = df_slice[df_slice["Ticker_full"] == company_ticker]["Year"].unique()
anchor_years.sort()

# Get anchor company name
anchor_company_name = df_slice[df_slice["Ticker_full"] == company_ticker]["Company_name"].iloc[0]

# Get anchor company data for the fixed anchor window
anchor_df = df_slice[(df_slice["Ticker_full"] == company_ticker) &
                     (df_slice["Year"].between(start_year, end_year))][
    ["Genome_encoding_x", "Genome_encoding_y", "Genome_classification_bespoke"]]

anchor_genome_seq = list(anchor_df["Genome_classification_bespoke"].values)

# Loop through all tickers for comparison
for ticker in tickers:
    if ticker != company_ticker:  # Avoid self-comparison
        company_years = df_slice[df_slice["Ticker_full"] == ticker]["Year"].unique()
        company_years.sort()

        # Iterate over all valid sliding windows of `time_window_length`
        for i in range(len(company_years) - time_window_length + 1):
            compare_start = company_years[i]
            compare_end = compare_start + time_window_length - 1  # Ensure fixed-length window

            # Get data for this sliding window
            compare_df = df_slice[(df_slice["Ticker_full"] == ticker) &
                                  (df_slice["Year"].between(compare_start, compare_end))][
                ["Genome_encoding_x", "Genome_encoding_y", "Genome_classification_bespoke"]]

            # **Skip the computation entirely if there's an "UNKNOWN"**
            try:
                if "UNKNOWN" in anchor_genome_seq or "UNKNOWN" in compare_df["Genome_classification_bespoke"].values:
                    raise ValueError("Skipping due to UNKNOWN value")

                # Ensure valid comparison (both have `time_window_length` data points)
                if len(anchor_df) == len(compare_df) == time_window_length:
                    distance = l1_distance(anchor_df[["Genome_encoding_x", "Genome_encoding_y"]],
                                           compare_df[["Genome_encoding_x", "Genome_encoding_y"]])

                    compare_company_name = df_slice[df_slice["Ticker_full"] == ticker]["Company_name"].iloc[0]
                    compare_genome_seq = list(compare_df["Genome_classification_bespoke"].values)

                    # Store results
                    distance_results.append([
                        company_ticker, anchor_company_name, start_year, end_year,  # Anchor details
                        ticker, compare_company_name, compare_start, compare_end,  # Comparison details
                        distance, anchor_genome_seq, compare_genome_seq
                    ])

                    print(f"Anchor: {company_ticker} ({anchor_company_name}) [{start_year}-{end_year}] | "
                          f"Comparing: {ticker} ({compare_company_name}) [{compare_start}-{compare_end}] | "
                          f"L1 Distance: {distance}")

            except ValueError:
                print(f"Skipping comparison {ticker} due to UNKNOWN in sequence.")

# Convert to DataFrame
distance_df = pd.DataFrame(distance_results, columns=[
    "Anchor_Ticker", "Anchor_Company", "Anchor_Start_Year", "Anchor_End_Year",
    "Comparison_Ticker", "Comparison_Company", "Comparison_Start_Year", "Comparison_End_Year",
    "L1_Distance", "Anchor_Genome_Sequence", "Comparison_Genome_Sequence"
])

x=1
y=2

