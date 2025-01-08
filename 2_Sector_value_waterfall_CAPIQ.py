import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestRegressor
import shap
from Utilities import compute_percentiles
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot
matplotlib.use('TkAgg')

"""
This is a script to compute the waterfall for an entire sector, or collection of sectors
"""

# Choose sector
sector = ["Technology"]
return_method = "Geometric" # Geometric or Arithmetic

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

dfs_list = []
for i in range(len(tickers_)):
    company_i = tickers_[i]
    try:
        df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_"+company_i+".csv")
        dfs_list.append(df)
        print("Company data ", company_i)
    except:
        print("Error with company ", company_i)

# Merge dataframes
df_merge = pd.concat(dfs_list)

# Drop rows with missing values in key columns
features = ["Year", "Market_Capitalisation",  "Book_Value_Equity", "Shares_outstanding", "Dividends_Paid", "Stock_repurchased",  "Dividend_Yield", "Dividend_Buyback_Yield", "ROTE", "PE", "Diluted_EPS",
            "NPAT", "Stock_Price"]
df_slice_ = df_merge[features]
df_slice = df_slice_.dropna()
df_slice["BVE_per_share"] = df_slice["Book_Value_Equity"] / df_slice["Shares_outstanding"]

# Get initial & final years
beginning_year = 2016
end_year = 2023

# Compute index levels for a market-cap based index
index_list_values = []
for i in range(beginning_year, end_year+1, 1):
    year_i = df_slice.loc[df_slice["Year"] == i]
    num_stocks = len(year_i)
    index_level = np.sum(year_i["Market_Capitalisation"].values)  # Corrected line
    index_list_values.append([i, index_level])

# Index values Dataframe
index_values_df = pd.DataFrame(index_list_values)
index_values_df.columns = ["Year", "Index"]

### BVE/share per annum (entire sector) ###
start_bv = df_slice.loc[df_slice["Year"]==beginning_year]["Book_Value_Equity"].values.sum()
start_shares_os = df_slice.loc[df_slice["Year"]==beginning_year]["Shares_outstanding"].values.sum()
start_bve_sector = start_bv/start_shares_os

end_bv = df_slice.loc[df_slice["Year"]==end_year]["Book_Value_Equity"].values.sum()
end_shares_os = df_slice.loc[df_slice["Year"]==end_year]["Shares_outstanding"].values.sum()
end_bve_sector = end_bv/end_shares_os
if return_method == "Arithmetic":
    bve_change_sector = (end_bve_sector/start_bve_sector - 1)/(end_year-beginning_year + 1)
if return_method == "Geometric":
    bve_change_sector = (end_bve_sector/start_bve_sector)**(1/(end_year-beginning_year+1)) - 1

### Total dividends & buybacks per share divided by all companies ###
if return_method == "Arithmetic" or return_method == "Geometric":
    # df_slice["Dividend_Buyback_Yield"].replace([np.inf, -np.inf], 0, inplace=True)
    total_dividends = abs(df_slice[(df_slice['Year'] >= beginning_year) & (df_slice['Year'] <= end_year)]["Dividends_Paid"]).sum()
    total_stock_repurchased = abs(df_slice[(df_slice['Year'] >= beginning_year) & (df_slice['Year'] <= end_year)]["Stock_repurchased"]).sum()
    total_index_values = abs(index_values_df[(index_values_df['Year'] >= beginning_year) & (index_values_df['Year'] <= end_year)]["Index"]).sum()

    # Compute dividend / buyback yield
    dividend_buyback_yield = (total_dividends + total_stock_repurchased)/total_index_values

### Change in ROE per annum (entire sector) ###
start_npat = df_slice.loc[df_slice["Year"]==beginning_year]["NPAT"].values.sum()
start_bve = df_slice.loc[df_slice["Year"]==beginning_year]["Book_Value_Equity"].values.sum()
start_roe_sector = start_npat/start_bve

end_npat = df_slice.loc[df_slice["Year"]==end_year]["NPAT"].values.sum()
end_bve = df_slice.loc[df_slice["Year"]==end_year]["Book_Value_Equity"].values.sum()
end_roe_sector = end_npat/end_bve
if return_method == "Arithmetic":
    roe_change_sector = ((end_roe_sector-start_roe_sector)/start_roe_sector)/(end_year-beginning_year + 1)
if return_method == "Geometric":
    roe_change_sector = (end_roe_sector/start_roe_sector)**(1/(end_year-beginning_year+1)) - 1

### Change in P/E per annum (entire sector) ###
start_price = df_slice.loc[df_slice["Year"]==beginning_year]["Stock_Price"].values.sum()
start_EPS = df_slice.loc[df_slice["Year"]==beginning_year]["Diluted_EPS"].values.sum()
start_pe_sector = start_price/start_EPS

end_price = df_slice.loc[df_slice["Year"]==end_year]["Stock_Price"].values.sum()
end_EPS = df_slice.loc[df_slice["Year"]==end_year]["Diluted_EPS"].values.sum()
end_pe_sector = end_price/end_EPS
if return_method == "Arithmetic":
    pe_change_sector = (end_pe_sector/start_pe_sector - 1)/(end_year-beginning_year + 1)
if return_method == "Geometric":
    pe_change_sector = (end_pe_sector/start_pe_sector)**(1/(end_year-beginning_year+1)) - 1

value_creation_sector = [bve_change_sector,  roe_change_sector, pe_change_sector, dividend_buyback_yield]
value_creation_labels = ["BVE Change", "ROE Change", "P/E Change", "Dividend & Buyback Yield"]

# Dataframe
df_plot = pd.DataFrame(value_creation_sector)
df_plot.index = value_creation_labels
df_plot.columns = ["Features"]

# Figure Size
fig, ax = plt.subplots(figsize=(16, 9))

# Horizontal Bar Plot
ax.barh(df_plot.index, df_plot["Features"])
plt.xlabel("Annualised growth "+ return_method)
plt.title(sector[0] + " TSR drivers")
plt.savefig(sector[0] + "_TSR_Decomposition "+str(beginning_year)+" - "+str(end_year))
plt.show()

# Decomposition of TSR
transition = pd.Series(list(value_creation_sector), index=value_creation_labels)
title_sector_year_margin = str(sector[0]) + " " + str(beginning_year) + "-" + str(end_year)

# Waterfall plots
waterfall_value_plot(transition, title_sector_year_margin)