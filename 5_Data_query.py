import numpy as np
import pandas as pd
from Utilities import compute_percentiles, firefly_plot, geometric_return
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math
import glob
import os

make_plots = True

# Apply Genome Filter
genome_filtering = False
top_100_companies = False
asx_100 = True
asx_200 = False

# Market capitalisation threshold
mcap_threshold = 500

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

matplotlib.use('TkAgg')

# Initialise years
beginning_year = 2011
end_year = 2023
# Generate grid of years
year_grid = np.linspace(beginning_year, end_year, end_year-beginning_year+1)
rolling_window = 3

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\company_list_asx200_.csv")

# Get unique tickers
unique_tickers = mapping_data["Ticker"].unique()

# Choose sectors to include
sector = mapping_data["Sector_new"].values

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

if asx_200:
    dfs_list = []
    for i in range(len(tickers_)):
        company_i = tickers_[i]
        try:
            df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Empirical_analysis\ASX100_data\_" + company_i + ".csv")
            # df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Capiq_data\_" + company_i + ".csv")
            dfs_list.append(df)
            print("Company data ", company_i)
        except:
            print("Error with company ", company_i)

if asx_100:
    # Directory containing the CSV files
    directory_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Empirical_analysis\ASX100_data"
    # List to store the dataframes
    dfs_list = []
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    # Read each CSV file and append to dfs_list
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs_list.append(df)
            # print(f"Successfully read {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"Error reading {os.path.basename(csv_file)}: {e}")

    # Optionally, print the list of successfully read dataframes
    print(f"Total files read: {len(dfs_list)}")

# Merge dataframes
df_concat = pd.concat(dfs_list)
df_merge = generate_genome_classification_df(df_concat)
# Create feature for Price-to-book
df_merge["Price_to_Book"] = df_merge["Market_Capitalisation"]/df_merge["Book_Value_Equity"]

if top_100_companies:
    # Assuming df is your DataFrame
    df_merge = df_merge.groupby('Company_name').filter(lambda x: x['Market_Capitalisation'].min() >= mcap_threshold)
    df_unique_companies = pd.DataFrame(df_merge["Company_name"].unique())
    # Write to csv file
    df_merge.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\CSL\Filtered_data\ASX_filtered_data.csv")
    df_unique_companies.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\CSL\Filtered_data\ASX_filtered_peers.csv")


# Desired classifications
desired_classifications = ["VIRTUOUS", "FAMOUS", "LEGENDARY"]

# Loop over unique tickers
results_list = []
for i in range(len(unique_tickers)):
    try:
        # Slice on unique ticker
        df_i = df_merge.loc[df_merge["Ticker"]==unique_tickers[i]]
        company_name_i = df_i["Company_name"].unique()[0]
        genome_classifications = df_i["Genome_classification"].iloc[3:]

        # Compute the percentage of desired classifications
        percentage = genome_classifications.isin(desired_classifications).mean() * 100

        # Append results to the list
        results_list.append({
            'Ticker': unique_tickers[i],
            'Company_name': company_name_i,
            'Percentage_in_Desired_Classifications': percentage
        })
        print("Iteration ", company_name_i)
    except:
        print("Issue with data for ", company_name_i)

# Convert results list to a DataFrame for better readability
results_df = pd.DataFrame(results_list)
results_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Survivors_top_segments.csv")
# Print the results DataFrame
print(results_df)

results_list_consecutive = []

for ticker in unique_tickers:
    try:
        df_i = df_merge.loc[df_merge["Ticker"] == ticker]
        company_name_i = df_i["Company_name"].unique()[0]

        is_desired_classification = df_i["Genome_classification"].isin(desired_classifications).values
        years = df_i["Year"].values
        classifications = df_i["Genome_classification"].values

        max_consecutive_years = 0
        current_consecutive_years = 0
        start_year = None
        end_year = None
        current_start_year = None
        next_classification_after_fallout = None

        for i, classification in enumerate(is_desired_classification):
            if classification:
                current_consecutive_years += 1
                if current_start_year is None:
                    current_start_year = years[i]
                if current_consecutive_years > max_consecutive_years:
                    max_consecutive_years = current_consecutive_years
                    start_year = current_start_year
                    end_year = years[i]
            else:
                if current_consecutive_years > 0:
                    next_classification_after_fallout = classifications[i]
                current_consecutive_years = 0
                current_start_year = None

        classification_after_fallout_category = None
        if next_classification_after_fallout in ["UNTENABLE", "TRAPPED", "BRAVE", "FEARLESS"]:
            classification_after_fallout_category = "BELOW THE LINE"
        elif next_classification_after_fallout in ["CHALLENGED", "VIRTUOUS", "FAMOUS", "LEGENDARY"]:
            classification_after_fallout_category = "ABOVE THE LINE"

        if max_consecutive_years >= 3:
            results_list_consecutive.append({
                'Ticker': ticker,
                'Company_name': company_name_i,
                'Max_Consecutive_Years': max_consecutive_years,
                'Start_Year': start_year,
                'End_Year': end_year,
                'Next_Classification_After_Fallout': next_classification_after_fallout,
                'Classification_After_Fallout_Category': classification_after_fallout_category
            })
    except:
        print("Issue with data for ", ticker)

results_list_consecutive_df = pd.DataFrame(results_list_consecutive)
results_list_consecutive_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Survivors_top_segments_consecutive.csv")
print(results_list_consecutive_df)