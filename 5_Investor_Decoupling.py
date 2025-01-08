import numpy as np
import pandas as pd
from Utilities import compute_percentiles, firefly_plot
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
from scipy.stats import linregress

matplotlib.use('TkAgg')

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\company_list_asx200.csv")

# Get unique tickers
unique_tickers = mapping_data["Ticker"].unique()

# Choose sectors to include
sector = mapping_data["Sector_new"].values

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

dfs_list = []
for i in range(len(tickers_)):
    company_i = tickers_[i]
    try:
        df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Capiq_data\_" + company_i + ".csv")
        dfs_list.append(df)
        print("Company data ", company_i)
    except:
        print("Error with company ", company_i)

# Merge dataframes
df_merge = pd.concat(dfs_list)

# Initialise Master DF
master_df_list = []
for ticker in range(len(tickers_)):
    # Global parameters
    company_ticker = unique_tickers[ticker]
    return_metric = "EP/FE"
    growth_metric = "Revenue_growth_3_f"
    print("Ticker iteration ", company_ticker)

    try:
        # Slice based on metric and company s
        sector_i = df_merge.loc[df_merge["Ticker"]==company_ticker]["Sector"].unique()[0]
        company_name_i = df_merge.loc[df_merge["Ticker"]==company_ticker]["Company_name"].unique()[0]

        # Get sector return and growth
        df_sector = df_merge.loc[df_merge["Sector"]==sector_i]
        df_company = df_merge.loc[df_merge["Ticker"]==company_ticker]

        # Growth & Return average
        company_i_return = df_company[return_metric].values
        company_i_growth = df_company[growth_metric].values

        # Sector
        sector_return_avg = df_sector.groupby("Year")[return_metric].mean().values
        sector_growth_avg = df_sector.groupby("Year")[growth_metric].mean().values

        # Market
        market_return_avg = df_merge.groupby("Year")[return_metric].mean().values
        market_growth_avg = df_merge.groupby("Year")[growth_metric].mean().values

        # Remove NaN values from Company
        company_i_return = company_i_return[~np.isnan(company_i_return)]
        company_i_growth = company_i_growth[~np.isnan(company_i_growth)]

        # Remove NaN values from Sector
        sector_return_avg = sector_return_avg[~np.isnan(sector_return_avg)]
        sector_growth_avg = sector_growth_avg[~np.isnan(sector_growth_avg)]

        # Remove NaN values from Market
        market_return_avg = market_return_avg[~np.isnan(market_return_avg)]
        market_growth_avg = market_growth_avg[~np.isnan(market_growth_avg)]

        def company_decoupling(company_name_i, company_vector, peer_vector, peer_group, metric):
            # Remove NaN values
            company_vector = company_vector[~np.isnan(company_vector)]
            peer_vector = peer_vector[~np.isnan(peer_vector)]

            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(peer_vector, company_vector)

            # Create scatter plot
            plt.scatter(peer_vector, company_vector, label='Data')

            # Plot the line of best fit
            plt.plot(peer_vector, intercept + slope * peer_vector, color='red', label='Regression')

            # Add gray line at x=0 and y=0
            plt.axhline(0, color='gray', alpha=0.3)
            plt.axvline(0, color='gray', alpha=0.3)

            # Add labels and legend
            plt.xlabel(peer_group)
            plt.ylabel(company_name_i)
            plt.title("Decoupling " + metric + " vs " + peer_group + " " + company_name_i)
            legend_text = f'Regression\nBeta: {slope:.2f}, Alpha: {intercept:.2f}'
            plt.legend([legend_text])
            plt.savefig("Decoupling " + company_name_i + " " + metric + " " + peer_group)

            # # Show plot
            plt.show()

            # Print intercept and slope
            print(metric + " Alpha:", intercept)
            print(metric + " Beta:", slope)

            return intercept, slope

        # Compute alpha & beta comparing Growth/Return to Sector/Market
        alpha_growth_sector, beta_growth_sector = company_decoupling(company_name_i, company_i_growth, sector_growth_avg, sector_i, "Growth")
        alpha_return_sector, beta_return_sector = company_decoupling(company_name_i, company_i_return, sector_return_avg, sector_i, "Return")
        # alpha_growth_market, beta_growth_market = company_decoupling(company_name_i, company_i_growth, market_growth_avg, "Market", "Growth")
        # alpha_return_market, beta_return_market = company_decoupling(company_name_i, company_i_return, market_return_avg, "Market", "Return")

        # Append to master dataframe
        master_df_list.append([company_name_i, company_ticker, sector_i, alpha_return_sector, beta_return_sector, alpha_growth_sector, beta_growth_sector])

    except:
        print("Issue with company ", company_ticker)

# Write to CSV file
master_df = pd.DataFrame(master_df_list)
master_df.columns = ["Company_name", "Ticker", "Sector", "Return_alpha", "Return_beta", "Growth_alpha", "Growth_beta"]
master_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Decoupling_script.csv")


