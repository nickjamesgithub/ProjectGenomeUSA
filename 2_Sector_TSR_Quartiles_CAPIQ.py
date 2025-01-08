import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestRegressor
import shap
from Utilities import compute_percentiles
import matplotlib
matplotlib.use('TkAgg')

"""
This is a TSR Driver analysis where you can choose the sector and identify the key drivers
"""

# Choose sector
sector = ["Insurance"]
plot_title = "Insurance Rolling P/E"

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

# List of year
year_lb = 2011
year_ub = 2024
year_grid = np.linspace(year_lb, year_ub, year_ub-year_lb+1)

tsr_list = []
year_list = []
for i in range(len(tickers_)):
    print("Iteration ", tickers_[i])
    company_i = tickers_[i]
    # Standard data
    df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_"+company_i+".csv")
    # Append TSR
    tsr = df["TSR_CIQ_no_buybacks"]
    tsr_list.append(tsr)

# Create TSR dataframe
tsr_df = pd.DataFrame(tsr_list)
tsr_df.columns = year_grid

# Generate Boxplot for market
fig, ax = plt.subplots()
bp = ax.boxplot(tsr_df, showfliers=False)
ax.set_xticklabels(year_grid.astype(int))
plt.ylabel("Average TSR")
plt.xlabel("Year")
plt.title(sector[0] + " TSR Quartiles")
plt.savefig(sector[0] + " TSR Quartiles")
plt.show()

