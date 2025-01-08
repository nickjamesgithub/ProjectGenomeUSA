import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib
matplotlib.use('TkAgg')

"""
This is a TSR Driver analysis where you can choose the sector and identify the key drivers
"""

# Choose sector
sector = ["Technology"]

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

dfs_list = []
for i in range(len(tickers_)):
    try:
        company_i = tickers_[i]
        df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_" + company_i + ".csv")
        dfs_list.append(df)
        print("Company data ", company_i)
    except:
        print("error with company ", company_i)

# Merge dataframes
df_merge = pd.concat(dfs_list)
df_merge_clean = df_merge.dropna()

# Random Forest Fit
X_drivers = ["Revenue_growth_1_f", "NAV_1_f", "Economic_profit_1_f", "EP/FE_1_f", "profit_margin_1_f",
"ROTE", "ROTE_above_Cost_of_equity", "roa_1_f", "Revenue_growth_2_f", "NAV_growth_2_f", "EP_growth_2_f", "EP/FE_growth_2_f",
"profit_margin_growth_2_f", "roa_growth_2_f", "Revenue_growth_3_f", "NAV_growth_3_f",
"EP_growth_3_f", "EP/FE_growth_3_f", "profit_margin_growth_3_f", "roa_growth_3_f", "CROTE_TE", "BVE_per_share_1_f", "BVE_per_share_growth_2_f", "BVE_per_share_growth_3_f"]

# Fill na values with 0
X = df_merge_clean[X_drivers]
# Clean y
y = df_merge_clean["TSR"]

# Fit random forest regression model
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X, y)

# Run Shapely values
explainer = shap.Explainer(rf, X)
shap_values = explainer(X)
fig = shap.summary_plot(shap_values, X, max_display=10, show=False)
plt.savefig("_Capiq" + sector[0] + "_" + "_Shap.png")
plt.show()
