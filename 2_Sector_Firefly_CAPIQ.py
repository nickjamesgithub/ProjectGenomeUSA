import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

matplotlib.use('TkAgg')

"""
This is a tool to compute an evolutionary Firefly at a country/sector level
"""

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")

# Define countries and sectors to include
countries_to_include = ['USA'] # 'USA', 'AUS', 'INDIA', 'JAPAN', 'EURO', 'UK'
sectors_to_include = ["Banking"]
plot_label = "US_Banking"

# 'Industrials', 'Materials', 'Healthcare', 'Technology',
#        'Insurance', 'Gaming/alcohol', 'Media', 'REIT', 'Utilities',
#        'Consumer staples', 'Consumer Discretionary',
#        'Investment and Wealth', 'Telecommunications', 'Energy', 'Banking',
#        'Metals', 'Financials - other', 'Mining', 'Consumer Staples',
#        'Diversified', 'Rail Transportation', 'Transportation'

# Filter the data based on the selected countries and sectors
filtered_data = data.copy()
filtered_data = filtered_data[filtered_data["Country"].isin(countries_to_include)]
filtered_data = filtered_data[filtered_data["Sector"].isin(sectors_to_include)]

# Drop rows with NaN or Inf values in critical columns
filtered_data["Revenue_growth_3_f"] = pd.to_numeric(filtered_data["Revenue_growth_3_f"], errors="coerce")
filtered_data["EVA_ratio_bespoke"] = pd.to_numeric(filtered_data["EVA_ratio_bespoke"], errors="coerce")
filtered_data = filtered_data.replace([np.inf, -np.inf], np.nan)  # Replace inf/-inf with NaN
filtered_data = filtered_data.dropna(subset=["Revenue_growth_3_f", "EVA_ratio_bespoke"])  # Drop NaN

# Prepare lists for plotting
x_axis_list = []
y_axis_list = []
labels_list = []

# Loop over each year
year_lb = 2014
year_ub = 2024
year_grid = np.linspace(year_lb, year_ub, int(year_ub - year_lb + 1))

for year in year_grid:
    df_year = filtered_data[filtered_data["Year"] == year]

    # Ensure Market_Capitalisation is numeric
    df_year["Market_Capitalisation"] = pd.to_numeric(df_year["Market_Capitalisation"], errors="coerce").fillna(0)

    # Filter out outliers based on thresholds
    mask = (
        (df_year["Revenue_growth_3_f"] >= -10) & (df_year["Revenue_growth_3_f"] <= 10) &
        (df_year["EVA_ratio_bespoke"] >= -10) & (df_year["EVA_ratio_bespoke"] <= 10)
    )
    df_valid = df_year[mask]

    # Ensure numeric vectors
    revenue_rolling_vector = pd.to_numeric(df_valid["Revenue_growth_3_f"], errors='coerce').fillna(0).values
    eva_ratio_bespoke = pd.to_numeric(df_valid["EVA_ratio_bespoke"], errors='coerce').fillna(0).values
    market_cap_vector = np.nan_to_num(df_valid["Market_Capitalisation"].values)

    # Compute weighted contribution vectors
    mcap_total = np.sum(market_cap_vector)
    mcap_weighted = market_cap_vector / mcap_total if mcap_total != 0 else np.zeros_like(market_cap_vector)

    # Weighted contribution for each feature
    revenue_rolling_weighted = np.dot(mcap_weighted, revenue_rolling_vector)
    eva_ratio_weighted = np.dot(mcap_weighted, eva_ratio_bespoke)

    # Append totals for plotting
    x_axis_list.append(revenue_rolling_weighted)
    y_axis_list.append(eva_ratio_weighted)
    labels_list.append(year)

# Plot the firefly plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
ax.plot(x_axis_list, y_axis_list, '-o', label="Firefly", color="blue")
for j in range(len(x_axis_list)):
    ax.annotate(f"{int(labels_list[j])}", (x_axis_list[j], y_axis_list[j]), fontsize=8)

# Draw labeled rectangles for grid
x_lb, x_ub = -0.3, 0.3
y_lb, y_ub = -0.3, 0.3
x_segment_ranges = [(x_lb, 0), (0, 0.1), (0.1, 0.2), (0.2, x_ub)]
y_segment_ranges = [(y_lb, 0), (0, y_ub)]
label_counter = 0
box_labels = ["Untenable", "Challenged", "Trapped", "Virtuous", "Brave", "Famous", "Fearless", "Legendary"]

for x_range in x_segment_ranges:
    for y_range in y_segment_ranges:
        rect = Rectangle((x_range[0], y_range[0]), x_range[1] - x_range[0], y_range[1] - y_range[0],
                         linewidth=0.3, edgecolor='black', facecolor='red', alpha=0.2)
        ax.add_patch(rect)
        box_label = box_labels[label_counter]
        ax.text((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2, box_label,
                ha='center', va='center', fontsize=8, fontweight='bold')
        label_counter += 1

# Final plot settings
plt.title(plot_label)
plt.xlabel("Revenue growth (3 year moving average)")
plt.ylabel("EVA Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Firefly_Single_Plot.png")
plt.show()
