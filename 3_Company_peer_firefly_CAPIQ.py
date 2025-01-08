import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib
matplotlib.use('TkAgg')

# Company name
company_list = ["MSFT:"] # , "NVDA:", "TSLA:", "AMZN:", "GOOG:", "NFLX:", "META:"
plot_label = "Microsoft" # company[:3]

# Store Economic profit and revenue growth
company_rev_growth_list = []
company_ep_list = []
labels_year = []
for i in range(len(company_list)):
    company, ticker = company_list[i].split(":")

    # Import data & slice specific company
    company_slice = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_"+company+".csv")

    print("Sector ", plot_label)
    print("Company ", company_list[i])

    feature_1 = "Revenue_growth_3_f"
    feature_2 = "TSR"

    # Get key genome criteria
    outputs = [feature_1, feature_2, "Year"]

    # Store values
    company_rev_growth_list.append(company_slice[outputs[0]].values)
    company_ep_list.append(company_slice[outputs[1]].values)
    labels_year.append(company_slice[outputs[2]].values)

# Merged scripts
merged_revenue_growth = list(itertools.chain(*company_rev_growth_list))
merged_economic_profit = list(itertools.chain(*company_ep_list))

# Plot average metric for each company over the years
fig, ax = plt.subplots()
for i in range(len(company_rev_growth_list)):
    if i == 0:
        plt.scatter(company_rev_growth_list[i], company_ep_list[i], label=company_list[i], alpha=0.4, color='red')
        for j, txt in enumerate(labels_year[i]):
            plt.annotate(f"{labels_year[i][j]}" + " " + company_list[i][:3], (company_rev_growth_list[i][j], company_ep_list[i][j]), fontsize=6)
    if i != 0:
        plt.scatter(company_rev_growth_list[i], company_ep_list[i], label=company_list[i], alpha=0.4, color='blue')
        for j, txt in enumerate(labels_year[i]):
            plt.annotate(f"{labels_year[i][j]}" + " " + company_list[i][:3], (company_rev_growth_list[i][j], company_ep_list[i][j]), fontsize=6)
    plt.xlabel(feature_1)
    plt.title(plot_label)
    plt.ylabel(feature_2)
    plt.legend()
plt.savefig("USA_Technology")
plt.show()
