import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use('TkAgg')

# Import data
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")
excel_output = True

# Full ticker list and corresponding start/end years
full_ticker_list = ["ASX:SHC"]
start_years = [2019]
end_years = [2024]
plot_label = "Sonic"

# Extract company names before looping
company_name_list = [data.loc[data["Ticker_full"] == ticker, "Company_name"].iloc[0] for ticker in full_ticker_list]

x_axis_list = []
y_axis_list = []
labels_list = []

for i, full_ticker in enumerate(full_ticker_list):
    idx_i, company_i = full_ticker.split(":")

    # Retrieve the correct country for the ticker
    country_i = data.loc[data["Ticker_full"] == full_ticker, "Country"].iloc[0]

    # Load company data from the correct country subfolder
    df = pd.read_csv(fr"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\{country_i}\_{company_i}.csv")

    # Filter for the selected time window
    company_slice = df[(df["Year"] >= start_years[i]) & (df["Year"] <= end_years[i])]

    # Extract relevant data
    labels = company_slice["Year"].values
    x_axis = company_slice["Revenue_growth_3_f"].values
    y_axis = company_slice["EVA_ratio_bespoke"].values

    # Append results
    x_axis_list.append(x_axis.tolist())
    y_axis_list.append(y_axis.tolist())
    labels_list.append(labels.tolist())

# Generate the grid for each axis
x_pad = max(map(len, x_axis_list))
y_pad = max(map(len, y_axis_list))
labels_pad = max(map(len, labels_list))

x_fill_list = np.array([i + [np.nan] * (x_pad - len(i)) for i in x_axis_list])
y_fill_list = np.array([i + [np.nan] * (y_pad - len(i)) for i in y_axis_list])
labels_fill_list = np.array([i + [np.nan] * (labels_pad - len(i)) for i in labels_list])

x = np.linspace(np.nanmin(x_fill_list), np.nanmax(x_fill_list), 100)
y = np.linspace(np.nanmin(y_fill_list), np.nanmax(y_fill_list), 100)

# Set automatic parameters for plotting
x_lb = min(-.3, np.nanmin(x))
x_ub = max(.3, np.nanmax(x))
y_lb = min(-.3, np.nanmin(y))
y_ub = max(.3, np.nanmax(y))

x_segment_ranges = [(x_lb, 0), (0, .1), (.1, .2), (.2, x_ub)]
y_segment_ranges = [(y_lb, 0), (0, y_ub)]
label_counter = 0
labels = ["Untenable", "Challenged", "Trapped", "Virtuous", "Brave", "Famous", "Fearless", "Legendary"]

fig, ax = plt.subplots()

# Plot each company with proper labels
for i in range(len(x_fill_list)):
    if i == 0:
        plt.plot(x_axis_list[i], y_axis_list[i], '-o', label=company_name_list[i], color="blue")
    else:
        plt.plot(x_axis_list[i], y_axis_list[i], '-o', label=company_name_list[i], alpha=0.4, linestyle='--')

    # Annotate with year labels
    for j, txt in enumerate(labels_list[i]):
        plt.annotate(txt, (x_axis_list[i][j], y_axis_list[i][j]), fontsize=6)

# Draw segmented rectangles and add labels
for x_range in x_segment_ranges:
    for y_range in y_segment_ranges:
        rect = Rectangle((x_range[0], y_range[0]), x_range[1] - x_range[0], y_range[1] - y_range[0],
                         linewidth=0.3, edgecolor='black', facecolor='red', alpha=0.5)
        ax.add_patch(rect)
        label = labels[label_counter]
        ax.text((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2, label,
                ha='center', va='center', color='black', fontsize=8, fontweight='bold', rotation=15)
        label_counter += 1

plt.title(plot_label)
plt.xlabel("Revenue growth (3 year moving average)")
plt.ylabel("EVA Ratio")
plt.legend()
plt.savefig(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\RHC\Firefly_plot_CAPIQ_" + plot_label)
plt.show()

if excel_output:
    # Write elements to csv
    # x_fill_df = pd.DataFrame(x_fill_list).transpose()
    x_stacked_array = np.vstack(x_fill_list)
    x_reshaped_array = x_stacked_array.reshape(-1, 1)
    y_stacked_array = np.vstack(y_fill_list)
    y_reshaped_array = y_stacked_array.reshape(-1, 1)
    labels_stacked_array = np.vstack(labels_list)
    labels_reshaped_array = labels_stacked_array.reshape(-1, 1)

    # Convert arrays to 1D arrays
    x_flat = x_reshaped_array.flatten()
    y_flat = y_reshaped_array.flatten()
    labels_flat = labels_reshaped_array.flatten()

    # Write a function for the marker array
    def create_marker_array(rows, cols):
        marker_array = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            marker_array[i, :] = i + 1
        return marker_array.reshape(-1, 1)


    # Get dimensions of labels list for marker array
    rows = len(labels_fill_list)
    cols = len(labels_fill_list[0])

    # Create the marker array
    marker_array = create_marker_array(rows, cols).flatten()

    # Create a DataFrame from the 1D arrays
    df = pd.DataFrame({
        'Series Labels': labels_flat,
        'X': x_flat,
        'Y': y_flat,
        'Marker and regression grouping': marker_array
    })

    # Write the DataFrame to a CSV file
    df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\RHC\_" + plot_label + "_firefly_projected_data_.csv", index=False)

