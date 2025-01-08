import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
pd.set_option('compute.use_numexpr', True)

"""
This is a TSR Driver analysis where you can choose the sector and identify the key drivers
"""

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")
# Choose sectors to include
sector = mapping_data["Sector_new"].values
plot_label = "Market"
# sector = ["Banking", "Industrials", "Consumer Discretionary", "Gaming"]

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

# List of year
year_lb = 2011
year_ub = 2023
year_grid = np.linspace(year_lb, year_ub, year_ub-year_lb+1)

tsr_list = []
year_list = []
for i in range(len(tickers_)):
    try:
        print("Iteration ", tickers_[i])
        company_i = tickers_[i]
        # Standard data
        df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Capiq_data\_"+company_i+".csv")
        # Append TSR
        tsr = df["TSR_CIQ_no_buybacks"]
        tsr_list.append(tsr)
    except:
        print("Issue with company data for ", tickers_[i])

# Create TSR dataframe
tsr_df = pd.DataFrame(tsr_list)
tsr_df.replace([np.nan, np.inf], 0, inplace=True)
tsr_df.columns = year_grid

# Generate Boxplot for market
fig, ax = plt.subplots()
bp = ax.boxplot(tsr_df.iloc[:,1:], showfliers=False)
ax.set_xticklabels(year_grid.astype(int)[1:])
plt.ylabel("Average TSR")
plt.xlabel("Year")
plt.title(plot_label + " TSR Quartiles")
plt.savefig(plot_label + " TSR Quartiles")
plt.show()


