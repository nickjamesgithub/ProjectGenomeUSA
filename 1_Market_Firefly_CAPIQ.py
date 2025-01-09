import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

matplotlib.use('TkAgg')

"""
This is a tool to compute an evolutionary Firefly at the market level
"""

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")

# Choose sectors to include
sector = mapping_data["Sector_new"].values
plot_label = "Market_Firefly"

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values
company_list = ["Market"]

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
        df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_"+company_i+".csv")

        # Slice key features
        year = df["Year"]
        market_cap = df["Market_Capitalisation"]
        revenue_rolling = df["Revenue_growth_3_f"]
        ep_fe = df["EP/FE"]
        company_name = df["Company_name"]

        # Append dataframe for specific company
        df_slice = pd.DataFrame([company_name, year, market_cap, revenue_rolling, ep_fe]).transpose()
        df_list.append(df_slice)
        company_tickers_list.append(company_i)

    except Exception as e:
        print("Issue with company data for ", tickers_[i], "Error:", e)

# Merge all dataframes into one master
df_merged = pd.concat(df_list, axis=0)
df_merged.columns = ["Company_name", "Year", "Market_Capitalisation", "Revenue_growth_3_f", "EP/FE"]

# List of year
year_lb = 2014
year_ub = 2024
year_grid = np.linspace(year_lb, year_ub, int(year_ub - year_lb + 1))

# Loop over respective years
x_axis_list = []
y_axis_list = []
labels_list = []
company_list = []
top_companies_by_year = []  # Store tables for top 10 companies for each year


for i in range(len(year_grid)):
    df_year = df_merged.loc[df_merged["Year"] == year_grid[i]]
    print("Iteration year ", year_grid[i])

    # Ensure Market_Capitalisation is numeric
    df_year["Market_Capitalisation"] = pd.to_numeric(df_year["Market_Capitalisation"], errors="coerce").fillna(0)

    company_vector = df_year["Company_name"].values
    market_cap_vector = np.nan_to_num(df_year["Market_Capitalisation"].values)

    # Ensure revenue_rolling_vector is numeric
    revenue_rolling_vector = pd.to_numeric(df_year["Revenue_growth_3_f"], errors='coerce').fillna(0).values
    ep_fe_vector = pd.to_numeric(df_year["EP/FE"], errors='coerce').fillna(0).values

    # Compute weighted contribution vectors
    mcap_total = np.sum(market_cap_vector)
    mcap_weighted = market_cap_vector / mcap_total

    # Weighted contribution for each feature
    revenue_rolling_weighted = np.dot(mcap_weighted, revenue_rolling_vector)
    ep_fe_weighted = np.dot(mcap_weighted, ep_fe_vector)

    # Append total
    company_list.append(company_vector)
    x_axis_list.append(revenue_rolling_weighted)
    y_axis_list.append(ep_fe_weighted)
    labels_list.append(year_grid[i])

    # Generate table of top 10 companies by market cap
    top_10_companies = df_year.nlargest(10, "Market_Capitalisation")[["Company_name", "Market_Capitalisation", "Revenue_growth_3_f", "EP/FE"]]
    top_10_companies["Year"] = int(year_grid[i])  # Add year column for clarity
    top_companies_by_year.append(top_10_companies)

    # Display the table for the current year
    print(f"\nTop 10 Companies for Year {int(year_grid[i])}:")
    print(top_10_companies)

# Combine results into a single DataFrame
final_table = pd.concat(top_companies_by_year, axis=0)

# Save the table to a CSV file
# final_table.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Top_10_Companies_by_Year.csv", index=False)

# Generate the grid for each axis
x = np.array(x_axis_list)
y = np.array(y_axis_list)

# Set automatic parameters for plotting
x_lb = min(-0.3, np.min(x))
x_ub = max(0.3, np.max(x))
y_lb = min(-0.3, np.min(y))
y_ub = max(0.3, np.max(y))

x_segment_ranges = [(x_lb, 0), (0, 0.1), (0.1, 0.2), (0.2, x_ub)]
y_segment_ranges = [(y_lb, 0), (0, y_ub)]
label_counter = 0
labels = ["Untenable", "Challenged", "Trapped", "Virtuous", "Brave", "Famous", "Fearless", "Legendary"]

fig, ax = plt.subplots()

# Plot the weighted contributions and annotate with year labels
plt.plot(x_axis_list, y_axis_list, '-o', color="blue")
for i in range(len(x_axis_list)):
    plt.annotate(f"{int(labels_list[i])}", (x_axis_list[i], y_axis_list[i]), fontsize=8)

# Draw labeled rectangles
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

# Final plot settings
plt.title(plot_label)
plt.xlabel("Revenue growth (3 year moving average)")
plt.ylabel("Economic profit / funds employed")
plt.savefig(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Firefly_plot_CAPIQ_" + plot_label)
plt.show()
