import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
import os

# grid = np.linspace(2011,2020,10)
#
# # Store quartile values over time
# quartile_75_list = []
# quartile_50_list = []
# quartile_25_list = []
#
# for i in range(len(grid)):

# Year range
beginning_year = 2019
end_year = 2024
tsr_method = "capital_iq" # bain or capital_iq
make_plots = False

# Apply Genome Filter
genome_filtering = False
sp500 = True

matplotlib.use('TkAgg')

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")
# Choose sectors to include
sector = mapping_data["Sector_new"].values

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

if sp500:
    dfs_list = []
    for i in range(len(tickers_)):
        company_i = tickers_[i]
        try:
            df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_" + company_i + ".csv")
            # df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Capiq_data\_" + company_i + ".csv")
            dfs_list.append(df)
            print("Company data ", company_i)
        except:
            print("Error with company ", company_i)


# Merge dataframes
df_merge = pd.concat(dfs_list)
# Create feature for Price-to-book
df_merge["Price_to_Book"] = df_merge["Market_Capitalisation"]/df_merge["Book_Value_Equity"]


# Get unique tickers
unique_tickers = df_merge["Ticker"].unique()

# Loop over unique tickers
dfs_concise_list = []
problem_ticker_list = []
for i in range(len(unique_tickers)):
    # Get unique ticker i
    ticker_i = unique_tickers[i]

    try:
        # Slice ticker_i within date range
        df_slice = df_merge.loc[(df_merge["Ticker"]==ticker_i) &
            (df_merge["Year"] >= beginning_year) & (df_merge["Year"] <= end_year)
                                ][["Year", "TSR", "Company_name", "Cost of Equity", "Stock_Price", "Adjusted_Stock_Price", "DPS", "BBPS", "DBBPS",
                                                                       "EP/FE", "Revenue_growth_3_f", "Price_to_Book",  "Dividend_Yield", "Buyback_Yield"]]

        # Get Cumulative TSR components + Average yields
        company_name_i = df_slice["Company_name"].unique()[0]

        # Get Final year EP/FE and Revenue growth
        ep_fe_final = df_slice.loc[df_slice["Year"]==end_year]["EP/FE"].iloc[0]
        revenue_growth_final = df_slice.loc[df_slice["Year"] == end_year]["Revenue_growth_3_f"].iloc[0]
        price_to_book_final = df_slice.loc[df_slice["Year"] == end_year]["Price_to_Book"].iloc[0]

        # Get unadjusted stock prices
        beginning_price = df_slice.loc[df_slice["Year"]==beginning_year]["Stock_Price"].values[0]
        final_price = df_slice.loc[df_slice["Year"]==end_year]["Stock_Price"].values[0]

        # Get adjusted stock prices
        beginning_price_adjusted = df_slice.loc[df_slice["Year"]==beginning_year]["Adjusted_Stock_Price"].values[0]
        final_price_adjusted = df_slice.loc[df_slice["Year"]==end_year]["Adjusted_Stock_Price"].values[0]

        # Cumulative dividends and buybacks per share
        cumulative_dps = np.sum(np.nan_to_num(df_slice.loc[(df_slice["Year"]>beginning_year) &
                                      (df_slice["Year"]<=end_year)]["DPS"]))
        cumulative_bbps = np.sum(np.nan_to_num(df_slice.loc[(df_slice["Year"]>beginning_year) &
                                      (df_slice["Year"]<=end_year)]["BBPS"]))

        # Average dividend and buyback yield
        avg_dividend_yield = df_slice.loc[(df_slice["Year"]>beginning_year) &
                                      (df_slice["Year"]<=end_year)]["Dividend_Yield"].mean()
        avg_buyback_yield = df_slice.loc[(df_slice["Year"]>beginning_year) &
                                      (df_slice["Year"]<=end_year)]["Buyback_Yield"].mean()

        # Compute annualized TSR with Bain method
        cumulative_tsr_bain = (final_price - beginning_price + cumulative_bbps + cumulative_dps)/beginning_price
        n = len(df_slice)-1
        annualized_tsr_bain = (1+cumulative_tsr_bain)**(1/n) - 1

        # Compute annualized TSR with Capital IQ method
        cumulative_tsr_ciq = final_price_adjusted/beginning_price_adjusted - 1
        annualized_tsr_ciq = (1 + cumulative_tsr_ciq)**(1/n) - 1

        # Print unique ticker & annualized TSR
        print("Ticker is ", unique_tickers[i] + " with annualized Bain TSR of " + str(annualized_tsr_bain))

        # Create a DataFrame for each row
        df_concise = pd.DataFrame({
            "Company": [company_name_i],
            "Ticker": [ticker_i],
            "Annualized_TSR_Bain": [annualized_tsr_bain],
            "Annualized_TSR_CIQ": [annualized_tsr_ciq],
            "Avg_dividend_yield": [avg_dividend_yield],
            "Avg_buyback_yield": [avg_buyback_yield],
            "EP/FE_final": [ep_fe_final],
            "Revenue_growth_final": [revenue_growth_final],
            "Price_to_book_final": [price_to_book_final]
        })

        # Append dictionary to storing list
        dfs_concise_list.append(df_concise)
        print("Unique ticker ", unique_tickers[i])
    except:
        print("Issue with company ", ticker_i)
        problem_ticker_list.append(ticker_i)

# Collapse dataframes into flat file
df_flat = pd.concat(dfs_concise_list)
df_flat.loc[:, ["Avg_dividend_yield", "Avg_buyback_yield"]] = df_flat.loc[:, ["Avg_dividend_yield", "Avg_buyback_yield"]].replace([np.inf, -np.inf], np.nan)

# Remove N/A values
df_flat.loc[df_flat['Avg_dividend_yield'] > 1, 'Avg_dividend_yield'] = np.nan
df_flat.loc[df_flat['Avg_buyback_yield'] > 1, 'Avg_buyback_yield'] = np.nan

# Apply standard Genome Filtering
if genome_filtering:
    df_flat = df_flat.loc[(df_flat["EP/FE_final"] >= -.3) & (df_flat["EP/FE_final"] <= .5) &
                                     (df_flat["Revenue_growth_final"] >= -.3) & (df_flat["Revenue_growth_final"] <= 1.5) &
                                     (df_flat["Annualized_TSR_CIQ"] >= -.4) & (df_flat["Annualized_TSR_CIQ"] <= 1) &
                                     (df_flat["Price_to_book_final"] > -200)]

if tsr_method == "capital_iq":
    # Remove all entries where TSR = -1
    df_flat = df_flat[df_flat["Annualized_TSR_CIQ"] != -1]
    # Sample data
    sorted_tsr = df_flat["Annualized_TSR_CIQ"].sort_values(ascending=False)

# Filter out non-finite values
sorted_tsr = sorted_tsr[np.isfinite(sorted_tsr)]

# Calculate percentiles
q_75 = np.percentile(sorted_tsr, 75)
q_50 = np.percentile(sorted_tsr, 50)
q_25 = np.percentile(sorted_tsr, 25)

# Plot ordered data as a line plot
plt.plot(sorted_tsr.values, np.arange(len(sorted_tsr)), marker='o')

# Annotate percentiles with vertical lines
plt.axvline(x=q_75, color='red', linestyle='--', label=f'75th percentile = {q_75 * 100:.2f}%')
plt.axvline(x=q_50, color='green', linestyle='--', label=f'50th percentile = {q_50 * 100:.2f}%')
plt.axvline(x=q_25, color='purple', linestyle='--', label=f'25th percentile = {q_25 * 100:.2f}%')

# Add legend
plt.legend()

# Add labels and title
plt.xlabel("Annualized TSR")
plt.ylabel("Company ranking")
plt.title(str(beginning_year) + "-" + str(end_year) + " Market TSR Quantiles")

# Show plot
plt.savefig("Market_tsr_quantiles_"+tsr_method+"_"+str(beginning_year)+"-"+str(end_year))
plt.show()
