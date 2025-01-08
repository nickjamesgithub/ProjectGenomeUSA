import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Year range
beginning_year = 2014
end_year = 2023
feature_1 = "Revenue_growth_3_f" # x-axis
feature_2 = "EP/FE" # y-axis

matplotlib.use('TkAgg')

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")
# Choose sectors to include
sector = mapping_data["Sector_new"].values

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

# Get unique tickers
unique_tickers = df_merge["Ticker"].unique()

def generate_quintiles(dataframe, feature):
    # Get feature of data
    feature_slice = dataframe[feature].replace([np.nan, -np.inf], 0)
    # feature_slice = np.nan_to_num(dataframe[feature])

    # Calculate quintiles for each column
    q1 = np.percentile(feature_slice, 10)
    q2 = np.percentile(feature_slice, 20)
    q3 = np.percentile(feature_slice, 30)
    q4 = np.percentile(feature_slice, 40)
    q5 = np.percentile(feature_slice, 50)
    q6 = np.percentile(feature_slice, 60)
    q7 = np.percentile(feature_slice, 70)
    q8 = np.percentile(feature_slice, 80)
    q9 = np.percentile(feature_slice, 90)

    # Define conditions for partitioning the data into quintiles
    conditions = [
        (feature_slice <= q1),
        (feature_slice > q1) & (feature_slice <= q2),
        (feature_slice > q2) & (feature_slice <= q3),
        (feature_slice > q3) & (feature_slice <= q4),
        (feature_slice > q4) & (feature_slice <= q5),
        (feature_slice > q5) & (feature_slice <= q6),
        (feature_slice > q6) & (feature_slice <= q7),
        (feature_slice > q7) & (feature_slice <= q8),
        (feature_slice > q8) & (feature_slice <= q9),
        (feature_slice > q9)
    ]

    # Define labels for the quintiles
    labels = [1,2,3,4,5,6,7,8,9,10]

    # Partition the data into quintiles using np.select
    dataframe[feature+'_Decile'] = np.select(conditions, labels)

# Generate quantiles
generate_quintiles(df_merge, feature_1)
generate_quintiles(df_merge, feature_2)

# Remove NaNs, infs, and values greater than 1000
df_merge = df_merge.replace([np.inf, -np.inf], np.nan)
df_merge = df_merge.dropna()
df_merge = df_merge[(df_merge["TSR"] <= 1000)]

# Now calculate average TSR based on size & leverage deciles
average_tsr = df_merge.groupby([feature_1+"_Decile", feature_2+"_Decile"])["TSR"].mean()

# If you want to reset the index and convert the result to a DataFrame
average_tsr_df = average_tsr.reset_index()

# If you want to pivot the result for better visualization
tsr_manifold = average_tsr_df.pivot(index=feature_1+"_Decile", columns=feature_2+"_Decile", values="TSR")

# Fill NaN values in the pivot table with 0
tsr_manifold_filled = tsr_manifold.fillna(0)

# Get the unique values for debt percentage and market capitalization
f1_values = np.sort(df_merge[feature_1].unique())
f2_values = np.sort(df_merge[feature_2].unique())

# Transpose the filled pivot table to align with the correct axis labels
tsr_manifold_filled_transposed = tsr_manifold_filled.T

# Convert the index and columns to arrays for the surface plot
X = tsr_manifold_filled.columns.values
Y = tsr_manifold_filled.index.values
X, Y = np.meshgrid(X, Y)

# Convert the table values to a 2D array for the surface plot
Z = tsr_manifold_filled.values
Z_percentage = Z * 100

# Set actual decile cutoff values as tick labels
# Calculate the deciles
f1_deciles = [np.percentile(f1_values, i) for i in range(10, 101, 10)]
f2_deciles = [np.percentile(f2_values, i) for i in range(10, 101, 10)]

# Format the deciles
f1_deciles_formatted = [f'{val:.2f}' for val in f1_deciles]
f2_deciles_formatted = [f'{val:.2f}' for val in f2_deciles]

# Plotting the contour plot with Z_percentage and a specified colormap
plt.figure()
contour = plt.contourf(X, Y, Z_percentage, cmap='RdYlGn')

# Add colorbar
plt.colorbar(contour, label='Average TSR (%)')

# Set labels and title
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title('Average TSR: ' + feature_1 + " vs " + feature_2)

# Set xticks and yticks labels
plt.xticks(np.arange(1, len(f1_deciles_formatted)+1), f1_deciles_formatted, rotation=45)
plt.yticks(np.arange(1, len(f2_deciles_formatted)+1), f2_deciles_formatted)

# Show the plot
plt.tight_layout()
plt.savefig("TSR_Surface_EPFE_Revenue_3")
plt.show()

