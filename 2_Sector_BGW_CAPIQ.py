import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")
# Choose sectors to include
# sector = mapping_data["Sector_new"].values
sector = ["Technology"]

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values
# tickers_ = ["CSL:ASX"]
plot_label = "USA_technology"

# Store Black/Grey/White values
bgw_values_list = []
for i in range(len(tickers_)):

    # Company ticker
    company_i = tickers_[i]

    try:
        df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_"+company_i+".csv")
        price_df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\share_price\_"+company_i+"_price.csv")
        current_price = price_df["Price"].iloc[-1]

        print("Company data ", company_i)

        # EPS, T-2
        eps_t_2 = df.loc[df["Year"]==2022]["Diluted_EPS"].values[0]
        # EPS, T-1
        trailing_eps = df.loc[df["Year"]==2023]["Diluted_EPS"].values[0]
        # Inferred EPS Growth Rate
        eps_growth = trailing_eps/eps_t_2 - 1
        # EPS, T+1
        forward_eps = trailing_eps * (1+eps_growth)
        # Trailing P/E
        trailing_pe = df.loc[df["Year"] == 2023]["PE_Implied"].values[0]
        # Forward P/E
        forward_pe = (df.loc[df["Year"]==2023]["Stock_Price"] / forward_eps).values[0]

        # Compute Cost of equity
        cost_of_equity = df["Cost of Equity"].iloc[-1]

        # Compute black space based on current earnings
        black_space = trailing_eps / cost_of_equity

        # Grey Space is computed based on Analyst estimates -- take median analyst estimate of target price
        eps1 = forward_eps
        eps2 = forward_eps * (1 + eps_growth)
        eps3 = eps2 * (1 + eps_growth)

        # Compute present value of EPS
        pv_eps_1 = (forward_eps - trailing_eps) / (1 + cost_of_equity)
        pv_eps_2 = (eps2 - eps1) / (1 + cost_of_equity) ** 2 * cost_of_equity
        pv_eps_3 = (eps3 - eps2) / (1 + cost_of_equity) ** 3 * cost_of_equity

        # Compute Grey space price
        grey_space = pv_eps_1 + pv_eps_2 + pv_eps_3

        # Compute White space
        white_space = current_price - grey_space - black_space

        # Final value
        total = black_space + white_space + grey_space

        # Append black/grey/white values to the master list
        bgw_values_list.append([black_space, grey_space, white_space])

    except:
        print("Error with company ", company_i)


def bgw_percentage_calculator(bgw_values):
    black = bgw_values[0]
    grey = bgw_values[1]
    white = bgw_values[2]

    # Black adjustment
    if black > 0:
        black_adj = black
    else:
        black_adj = 0

    # Grey adjustment
    if grey > 0:
        grey_adj = grey
    else:
        grey_adj = 0

    # White adjustment
    if white > 0:
        white_adj = white
    else:
        white_adj = 0

    bgw_values_adjusted = [black_adj, grey_adj, white_adj]

    # Percentages for the figure
    bgw_values_percentages = np.round((bgw_values_adjusted/np.sum(bgw_values_adjusted)), 3)

    return list(bgw_values_percentages)

# Store B/G/W values in a list
bgw_percentages_list = []
for j in range(len(bgw_values_list)):
    print("B/G/W Percentage adjustment")
    bgw_percentages = bgw_percentage_calculator(bgw_values_list[j])
    bgw_percentages_list.append(bgw_percentages)

# Black/Grey/White dataframe
bgw_percentages_df = pd.DataFrame(bgw_percentages_list)
bgw_percentages_df.columns = ["Black", "Grey", "White"]
bgw_percentages_df.index = tickers_

# Plot
bgw_percentages_df.plot.bar(stacked=True, rot=0, color = {'Black':'Black', "Grey":'dimgrey', "White":'lightgrey'})
plt.title(plot_label + " - Black/Grey/White space")
plt.savefig("BGW_"+plot_label)
plt.show()
