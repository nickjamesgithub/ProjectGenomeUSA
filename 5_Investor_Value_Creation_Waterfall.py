import numpy as np
import pandas as pd
from Utilities import compute_percentiles, firefly_plot, geometric_return
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
import numpy as np
from Utilities import value_creation_waterfall
import glob
import os

make_plots = True

# Apply Genome Filter
genome_filtering = True
sp_500 = True

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
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")

# Get unique tickers
unique_tickers = mapping_data["Ticker"].unique()

# Choose sectors to include
sector = mapping_data["Sector_new"].values

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

if sp_500:
    dfs_list = []
    for i in range(len(tickers_)):
        company_i = tickers_[i]
        try:
            df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_" + company_i + ".csv")
            dfs_list.append(df)
            print("Company data ", company_i)
        except:
            print("Error with company ", company_i)


# Merge dataframes
df_concat = pd.concat(dfs_list)
df_merge = generate_genome_classification_df(df_concat)
# Create feature for Price-to-book
df_merge["Price_to_Book"] = df_merge["Market_Capitalisation"]/df_merge["Book_Value_Equity"]


# Initial: Year, Genome Segment, Sectors
year_init = 2017
year_final = 2023
genome_classification_init = ["CHALLENGED"] # df_merge["Genome_classification"].unique()
genome_classification_final = ["VIRTUOUS", "FAMOUS"] # df_merge["Genome_classification"].unique() # ["FAMOUS"]
sector = df_merge["Sector"].unique() # ["Technology"]

# Filter the initial dataframe
df_init = df_merge.loc[(df_merge["Year"] == year_init) &
                       (df_merge["Genome_classification"].isin(genome_classification_init)) &
                       (df_merge["Sector"].isin(sector))]

# Filter the final dataframe
df_final = df_merge.loc[(df_merge["Year"] == year_final) &
                        (df_merge["Genome_classification"].isin(genome_classification_final)) &
                        (df_merge["Sector"].isin(sector))]

# Find the common companies in both filtered dataframes
common_companies = df_init[df_init["Company_name"].isin(df_final["Company_name"])]

# Filter the initial and final dataframes to include only the common companies
df_init_filtered = df_init[df_init["Company_name"].isin(common_companies["Company_name"])]
df_final_filtered = df_final[df_final["Company_name"].isin(common_companies["Company_name"])]

# Compute value creation waterfall
value_creation_waterfall(df_init_filtered, df_final_filtered, year_init, year_final)
# Calculate metrics
df_init_filtered["Revenue_init"] = df_init_filtered["Revenue"]
df_final_filtered["Revenue_final"] = df_final_filtered["Revenue"]

df_init_filtered[f"NPAT_margin_{year_init}"] = df_init_filtered["NPAT"] / df_init_filtered["Revenue"]
df_final_filtered[f"NPAT_margin_{year_final}"] = df_final_filtered["NPAT"] / df_final_filtered["Revenue"]

# Handle inf values by replacing them with NaN
df_init_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)
df_final_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)

# Select and rename the relevant columns
df_init_filtered = df_init_filtered[["Company_name", "Revenue", f"NPAT_margin_{year_init}", "PE_Implied"]]
df_final_filtered = df_final_filtered[["Company_name", "Revenue", f"NPAT_margin_{year_final}", "PE_Implied"]]

# Rename columns to avoid conflicts during merge
df_init_filtered = df_init_filtered.rename(columns={"Revenue": f"Revenue_{year_init}", "PE_Implied": f"PE_Implied_{year_init}"})
df_final_filtered = df_final_filtered.rename(columns={"Revenue": f"Revenue_{year_final}", "PE_Implied": f"PE_Implied_{year_final}"})

# Merge the initial and final filtered dataframes on Company_name
df_combined = pd.merge(df_init_filtered, df_final_filtered, on="Company_name")

# Calculate additional metrics
df_combined["Revenue_growth"] = (df_combined[f"Revenue_{year_final}"] - df_combined[f"Revenue_{year_init}"]) / df_combined[f"Revenue_{year_init}"]
df_combined["Profit_margin_diff"] = df_combined[f"NPAT_margin_{year_final}"] - df_combined[f"NPAT_margin_{year_init}"]
df_combined["PE_diff"] = df_combined[f"PE_Implied_{year_final}"] - df_combined[f"PE_Implied_{year_init}"]

# Create the final table
df_summary = df_combined[["Company_name",
                          f"Revenue_{year_init}", f"Revenue_{year_final}",
                          "Revenue_growth",
                          f"NPAT_margin_{year_init}", f"NPAT_margin_{year_final}",
                          "Profit_margin_diff",
                          f"PE_Implied_{year_init}", f"PE_Implied_{year_final}",
                          "PE_diff"]]

# Calculate averages, ignoring NaN values
averages = df_summary.mean(numeric_only=True, skipna=True).to_frame().T
averages["Company_name"] = "Average"

# Append the averages to the bottom of the dataframe
df_summary = pd.concat([df_summary, averages], ignore_index=True)

# Display the table
print(df_summary)

# Optional: Display the table in a more readable format
print(df_summary.to_string(index=False))