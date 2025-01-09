import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

matplotlib.use('TkAgg')

""" 
This is a tool to compute en evolutionary Firefly at the sector level (bespoke)
"""

# Choose sector
sector_list = ["Industrials"]
plot_title = "USA_Industrials_Firefly"

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector_list)]["Ticker"].values

# Create lists for Market cap / Rolling Revenue and EP/FE
market_cap_list = []
revenue_rolling = []
ep_fe = []
df_list = []
company_tickers_list = []

for i in range(len(tickers_)):
    try:
        print("Iteration ", tickers_[i])
        company_i = tickers_[i]
        # Standard data
        df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_" + company_i + ".csv")

        # Slice key features
        year = df["Year"]
        market_cap = df["Market_Capitalisation"]
        revenue_rolling = df["Revenue_growth_3_f"]
        ep_fe = df["EP/FE"]

        # Append dataframe for specific company
        df_slice = pd.DataFrame([year, market_cap, revenue_rolling, ep_fe]).transpose()
        df_list.append(df_slice)
        company_tickers_list.append(company_i)

    except:
        print("Issue with company data for ", tickers_[i])

# Merge all dataframes into one master
df_merged = pd.concat(df_list, axis=0)

# List of year
year_lb = 2015
year_ub = 2023
year_grid = np.linspace(year_lb, year_ub, year_ub - year_lb + 1)

x_axis_list = []
y_axis_list = []
labels_list = []

for i in range(len(year_grid)):
    df_year = df_merged.loc[df_merged["Year"] == year_grid[i]]
    print("Iteration year ", year_grid[i])

    market_cap_vector = df_year["Market_Capitalisation"].values
    revenue_rolling_vector = np.nan_to_num(df_year["Revenue_growth_3_f"].values)
    ep_fe_vector = np.nan_to_num(df_year["EP/FE"].values)

    # Compute weighted contribution vectors
    mcap_total = market_cap_vector.sum()
    mcap_weighted = market_cap_vector / mcap_total

    # Weighted contribution for each feature
    revenue_rolling_weighted = np.dot(mcap_weighted, revenue_rolling_vector)
    ep_fe_weighted = np.dot(mcap_weighted, ep_fe_vector)

    # Append total
    x_axis_list.append(revenue_rolling_weighted)
    y_axis_list.append(ep_fe_weighted)
    labels_list.append(year_grid[i])

# Append to global list
x_axis_list = [x_axis_list]
y_axis_list = [y_axis_list]
labels_list = [labels_list]

# Generate the grid for each axis
x_pad = len(max(x_axis_list, key=len))
x_fill_list = np.array([i + [0] * (x_pad - len(i)) for i in x_axis_list])
y_pad = len(max(x_axis_list, key=len))
y_fill_list = np.array([i + [0] * (y_pad - len(i)) for i in y_axis_list])
labels_pad = len(max(labels_list, key=len))
labels_fill_list = np.array([i + [0] * (labels_pad - len(i)) for i in labels_list])

x = np.linspace(np.min(np.nan_to_num(x_fill_list)), np.max(np.nan_to_num(x_fill_list)), 100)
y = np.linspace(np.min(np.nan_to_num(y_fill_list)), np.max(np.nan_to_num(y_fill_list)), 100)

# Set automatic parameters for plotting
x_lb = min(-.3, np.min(x))
x_ub = max(.3, np.max(x))
y_lb = min(-.3, np.min(y))
y_ub = max(.3, np.max(y))

x_segment_ranges = [(x_lb, 0), (0, .1), (.1, .2), (.2, x_ub)]
y_segment_ranges = [(y_lb, 0), (0, y_ub)]
label_counter = 0
labels = ["Untenable", "Challenged", "Trapped", "Virtuous", "Brave", "Famous", "Fearless", "Legendary"]

fig, ax = plt.subplots()
for i in range(len(x_fill_list)):
    # Generate plots
    if i == 0:
        plt.plot(x_axis_list[i], y_axis_list[i], '-o', color="blue")
    for j, txt in enumerate(labels_list[i]):
        plt.annotate(labels_list[i][j], (x_axis_list[i][j], y_axis_list[i][j]), fontsize=6)

for x_range in x_segment_ranges:
    for y_range in y_segment_ranges:
        rect = Rectangle((x_range[0], y_range[0]), x_range[1] - x_range[0], y_range[1] - y_range[0],
                         linewidth=0.3, edgecolor='black', facecolor='red', alpha=0.5)
        ax.add_patch(rect)
        # Add labels to each rectangle
        label = labels[label_counter]
        ax.text((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2, label,
                ha='center', va='center', color='black', fontsize=8, fontweight='bold', rotation=15)
        label_counter += 1

plt.title(plot_title)
plt.xlabel("Revenue growth (3 year moving average)")
plt.ylabel("Economic profit / funds employed")
plt.legend()
plt.savefig(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Firefly_plot_CAPIQ_" + plot_title)
plt.show()
