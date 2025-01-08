import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import networkx as nx
import string
matplotlib.use('TkAgg')
from Utilities import generate_market_cap_class, waterfall_value_plot

# Company name
company_list = "CSL:ASX"
company = company_list[:3]

# Year range
beginning_year = 2014
end_year = 2023
tsr_method = "capital_iq" # bain, capital_iq

# Import data & slice specific company
company_slice = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Capiq_data\_"+company+".csv")
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\company_list_asx200.csv")

# Get cost of equity values
df_slice = company_slice.loc[(company_slice["Ticker"]==company) &
                                   (company_slice["Year"] >= beginning_year) &
                                   (company_slice["Year"] <= end_year)][["Year", "TSR", "Cost of Equity", "Stock_Price", "Adjusted_Stock_Price", "DPS", "BBPS", "DBBPS",
                                                                         "Dividend_Yield", "Buyback_Yield", "Stock_gain_loss"]]

# Get vectors of data
cost_of_equity_vector = df_slice["Cost of Equity"]
dividend_yield_vector = df_slice["Dividend_Yield"]
buyback_yield_vector = df_slice["Buyback_Yield"]
adjusted_price_vector = df_slice["Adjusted_Stock_Price"]
price_vector = df_slice["Stock_Price"]
dps_vector = df_slice["DPS"]
bbps_vector = df_slice["BBPS"]

# Cumulative cost of equity
average_cost_of_equity = cost_of_equity_vector.mean()

# Cumulative dividend Yield
avg_dividend_yield = dividend_yield_vector.mean()

# Cumulative buyback yield
avg_buyback_yield = buyback_yield_vector.mean()

# Price accumulation - Cumulative and annualized
cumulative_price_gain = (price_vector.iloc[-1] - price_vector.iloc[0])/price_vector.iloc[0]
annualized_price_gain = (1 + cumulative_price_gain)**(1/(end_year-beginning_year+1)) - 1

# TSR over period - Cumulative and annualized
if tsr_method == "bain":
    tsr_cumulative = (price_vector.iloc[-1] - price_vector.iloc[0] + dps_vector.iloc[1:].sum() + bbps_vector.iloc[1:].sum()) / price_vector.iloc[0]
    tsr_annualized = (1 + tsr_cumulative)**(1/(end_year-beginning_year)) - 1
if tsr_method == "capital_iq":
    tsr_cumulative = adjusted_price_vector.iloc[-1]/adjusted_price_vector.iloc[0] - 1
    tsr_annualized = (1 + tsr_cumulative)**(1/(end_year-beginning_year)) - 1

# Cumulative Shareholder value
sv_average = tsr_annualized - average_cost_of_equity

# Initialise dictionary and add new values one-by-one
graph_labels_clean = {}
graph_labels_clean["Average Shareholder value"] = sv_average
graph_labels_clean["Average Cost of equity"] = average_cost_of_equity
graph_labels_clean["Annualized TSR"] = tsr_annualized
graph_labels_clean["Average Dividend Yield"] = avg_dividend_yield
graph_labels_clean["Average Buyback Yield"] = avg_buyback_yield
graph_labels_clean["Annualized Price accumulation"] = annualized_price_gain

G = nx.DiGraph()
nodes = np.arange(0, 5).tolist()
G.add_nodes_from(nodes)

G.add_edges_from([(0,1), (0,2),
                  (2,3), (2,4), (2,5)])

pos = {0:(7.5, 15),
 1:(5, 12.5), 2:(10, 12.5),
       3: (9, 9), 4: (10, 10), 5:(11,11)}

# Add labels with values formatted as percentages
labels = {
    0: f"Average Shareholder value\n{graph_labels_clean['Average Shareholder value']*100:.1f}%",
    1: f"Average Cost of equity\n{graph_labels_clean['Average Cost of equity']*100:.1f}%",
    2: f"Annualized TSR\n{graph_labels_clean['Annualized TSR']*100:.1f}%",
    3: f"Average Dividend Yield\n{graph_labels_clean['Average Dividend Yield']*100:.1f}%",
    4: f"Average Buyback Yield\n{graph_labels_clean['Average Buyback Yield']*100:.1f}%",
    5: f"Annualized Price accumulation\n{graph_labels_clean['Annualized Price accumulation']*100:.1f}%"
}

fig, axs = plt.subplots(1, 1, figsize=(20, 15))
nx.draw_networkx(G, pos=pos, labels=labels, arrows=True,
                 node_shape="s", node_color="white")
company_flat = company.translate(str.maketrans('', '', string.punctuation))
plt.savefig("Average_Driver_tree_" + str(beginning_year) + "-" + str(end_year) + "-" + company_flat)
plt.title("Average_Driver_tree_" + str(beginning_year) + "-" + str(end_year) + "-" + company_flat)
plt.show()