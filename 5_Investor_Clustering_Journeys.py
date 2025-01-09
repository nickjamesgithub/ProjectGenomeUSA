import pandas as pd
from Utilities import compute_percentiles, firefly_plot
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np

matplotlib.use('TkAgg')

# Global parameters
make_plots = True
num_clusters = 6
outlier_tickers = [] # James Hardie, Neuren Pharma, De Grey

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")
# Choose sectors to include
sector = mapping_data["Sector_new"].values

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values
tickers_full = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Full Ticker"].values

# Drop outlier tickers from tickers_
tickers_ = np.setdiff1d(tickers_, outlier_tickers)

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
df_merge.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_data_compressed.csv")
# Engineer individual columns
labels = np.array(df_merge["Year"])
x_axis = np.array(df_merge["Revenue_growth_3_f"])
y_axis = np.array(df_merge["EP/FE"])

# Loop over all companies and compute distance between points on grid
firefly_dfs = []
company_list = []
company_labels_list = []
tickers_list_survive = []
for i in range(len(tickers_)):
    # Get dimensions for company i in range 2017-2023
    tickers = df_merge.loc[(df_merge["Ticker_full"]==tickers_full[i]) & (df_merge["Year"]>=2017) & (df_merge["Year"]<=2023)]
    firefly_attributes = tickers[["Company_name", "Ticker_full", "Year", "EP/FE", "Revenue_growth_3_f", "TSR", "Dividend_Buyback_Yield", "PE_Implied", "WACC_Damodaran", "ROTE", "ROA", "ROFE", "ROCE"]]
    ep_fe = tickers["EP/FE"]
    rev_growth = tickers["Revenue_growth_3_f"]
    if ep_fe.isna().sum() == 0 and rev_growth.isna().sum() == 0 and len(ep_fe) == 7 and len(rev_growth) == 7:
        firefly_dfs.append(firefly_attributes)
        company_list.append([tickers["Company_name"].unique()[0], ep_fe.values, rev_growth.values])
        company_labels_list.append(tickers["Company_name"].unique()[0])
        tickers_list_survive.append(tickers_[i])
    else:
        print("Issue with company ", tickers["Company_name"].unique())

# Make company attributes list into a DF
firefly_df = pd.concat(firefly_dfs)

# Compute distance b
distance_matrix_ep_fe = np.zeros(((len(company_list), len(company_list))))
distance_matrix_revenue = np.zeros(((len(company_list), len(company_list))))
for i in range(len(company_list)):
    for j in range(len(company_list)):
        # Company i
        ep_fe_i = company_list[i][1]
        rev_growth_i = company_list[i][2]

        # Company j
        ep_fe_j = company_list[j][1]
        rev_growth_j = company_list[j][2]

        # EP/FE distances
        ep_fe_distance = np.sum(np.abs(ep_fe_i - ep_fe_j))
        rev_growth_distance = np.sum(np.abs(rev_growth_i - rev_growth_j))

        # Journey distance
        distance_matrix_ep_fe[i,j] = ep_fe_distance
        distance_matrix_revenue[i,j] = rev_growth_distance

    print("Company iteration ", i)

# Convert to affinity matrices
affinity_ep_fe = 1 - distance_matrix_ep_fe/np.max(distance_matrix_ep_fe)
affinity_revenue = 1 - distance_matrix_revenue/np.max(distance_matrix_revenue)
affinity_master = affinity_ep_fe + affinity_revenue

if make_plots:
    # Journey distance heatmap
    plt.matshow(affinity_master)
    plt.show()

    # Dendrogram plot
    dendrogram_plot(affinity_master, "L1", "Journey", np.array(company_labels_list))

# Apply clustering to algorithm
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(affinity_master)
cluster_labels = kmeans.labels_

# cluster approximation
cluster_labels_even = get_even_clusters(affinity_master, (len(affinity_master)/num_clusters)+1)

# Add labels to Firefly dataframe
firefly_cluster_df_lists = []
for i in range(len(tickers_list_survive)):
    company_i = firefly_df.loc[firefly_df["Ticker_full"]==tickers_list_survive[i]]
    company_i["Cluster_label"] = cluster_labels_even[i]
    firefly_cluster_df_lists.append(company_i)

# Collapse into one Dataframe
firefly_cluster_df_merge = pd.concat(firefly_cluster_df_lists)
# Replace 'inf' values with np.nan
firefly_cluster_df_merge.replace([np.inf, -np.inf], np.nan, inplace=True)
# Write to CSV file
firefly_cluster_df_merge.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_Journey_clusters.csv")

# for i in range(0,num_clusters):
#     # Select cluster of interest
#     cluster_i = firefly_cluster_df_merge.loc[firefly_cluster_df_merge["Cluster_label"]==i]
#
#     # Group the DataFrame by 'Year' and compute the mean for each metric excluding non-numeric columns
#     numeric_columns = cluster_i.select_dtypes(include=[np.number]).columns
#     grouped_df = cluster_i.groupby('Year')[numeric_columns].median()
#
#     # Firefly plot
#     firefly_plot(grouped_df["Year"], grouped_df["Revenue_growth_3_f"], grouped_df["EP/FE"], "cluster "+str(i+1))
#     print("Cluster ", str(i), " Revenue growth ", grouped_df["Revenue_growth_3_f"].median(), " EP/FE ",  grouped_df["EP/FE"].median())

