import numpy as np
import pandas as pd
import matplotlib
import string
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.cluster import KMeans

# Company name
company_list = ["NVDA:"] #"MSFT:", "NVDA:", "TSLA:", "AMZN:", "GOOG:", "NFLX:", "META:"
plot_label = "NVIDIA" # company[:3]

x_axis_list = []
y_axis_list = []
labels_list = []

for i in range(len(company_list)):
    company, idx = company_list[i].split(":")

    # Import data & slice specific company
    company_slice = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_"+company+".csv")

    # Engineer individual columns
    labels = np.array(company_slice["Year"])
    x_axis = np.array(company_slice["Revenue_growth_3_f"])
    y_axis = np.array(company_slice["EP/FE"])

    # Append results from all companies in a list
    x_axis_list.append(list(x_axis))
    y_axis_list.append(list(y_axis))
    labels_list.append(list(labels))

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
        plt.plot(x_axis_list[i], y_axis_list[i], '-o', label=company_list[i], color="blue")
    # If we want a generic figure for all competitors, just do !=
    if i != 0:
        plt.plot(x_axis_list[i], y_axis_list[i], '-o', label=company_list[i], alpha=0.4, linestyle='--')
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

plt.title(plot_label)
plt.xlabel("Revenue growth (3 year moving average)")
plt.ylabel("EP/FE")
plt.legend()
plt.savefig(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\USA_technology\Firefly_plot_CAPIQ_" + plot_label)
plt.show()
