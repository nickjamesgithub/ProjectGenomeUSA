import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import matplotlib
matplotlib.use('TkAgg')

"""
This is a TSR Driver analysis where you can choose the sector and identify the key drivers
"""

# Choose sector
sector = ["Banking"]
plot_title = "Banking Rolling P/E"

# Import data
# mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Sector_mapping.csv")
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\company_list_asx200.csv")

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

pe_list = []
dates_list = []
for i in range(len(tickers_)):
    try:
        print("Iteration ", tickers_[i])
        company_i = tickers_[i]
        # Standard data
        df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Capiq_data\_"+company_i+".csv")
        # Price data
        price_df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Capiq_data\share_price\_"+company_i+"_price.csv")
        price_df["Year"] = pd.DatetimeIndex(price_df["Date"]).year

        # Set conditions to conditionally populate P/E
        conditions = [
            (price_df['Year'] == 2019),
            (price_df['Year'] == 2020),
            (price_df['Year'] == 2021),
            (price_df['Year'] == 2022),
            (price_df['Year'] == 2023),
        ]

        # create a list of the values we want to assign for each condition
        values = [df.loc[df["Year"]==2019]["Diluted_EPS"],
                  df.loc[df["Year"]==2020]["Diluted_EPS"],
                  df.loc[df["Year"]==2021]["Diluted_EPS"],
                  df.loc[df["Year"]==2022]["Diluted_EPS"],
                  df.loc[df["Year"]==2023]["Diluted_EPS"]]

        # create a new column and use np.select to assign values to it using our lists as arguments
        price_df["Diluted_EPS_Year"] = np.select(conditions, values)

        # Add a column for rolling P/E based on rolling Price/EPS
        price_df["PE_rolling"] = price_df["Price"] / price_df["Diluted_EPS_Year"]

        # Remove all N/A and inf values based on no EPS data
        price_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        price_df.dropna(inplace=True)

        # Append to lists
        pe_list.append(price_df["PE_rolling"])
        dates_list.append(price_df["Date"])
    except:
        print("Issue with company ", tickers_[i])

fig, ax = plt.subplots()
for i in range(len(pe_list)):
    date_grid = pd.date_range(min(dates_list[i]), max(dates_list[i]), len(pe_list[i]))
    plt.plot(date_grid, pe_list[i], label = tickers_[i])
    plt.title(plot_title)
    company_flat = plot_title.translate(str.maketrans('', '', string.punctuation))
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
plt.legend()
plt.savefig("PE_Insurance_rolling")
plt.show()

