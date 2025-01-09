import numpy as np
import pandas as pd
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.cluster import KMeans

# Import data & slice specific company
company_slice = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_AAPL.csv")
company_list = ["AAPL"]
plot_label = "AAPL_projected_firefly"
scenario = "N/A" # north, east, north_east

# Current 2023 Financials
fy_23_revenue = company_slice.loc[company_slice["Year"]==2023]["Revenue"].iloc[0]
fy_23_assets = company_slice.loc[company_slice["Year"]==2023]["Total_assets"].iloc[0]
fy_23_liabilities = company_slice.loc[company_slice["Year"]==2023]["Total_liabilities"].iloc[0]

# Next 3 year forecasts
year_append = [2024, 2025, 2026, 2027, 2028]

# THESE ARE BUTTONS TO FILL IN DIRECTLY ON THE PLATFORM
revenue_growth_market = [0.102, 0.0792, 0.0857, .1322, .0793] # USER TO FILL IN BASED ON BROKER REPORT (BUTTON)
asset_growth_market = [.033, .052, .053, .164, .088] # USER TO FILL IN BASED ON BROKER REPORT (BUTTON)
liability_growth_market = [-.027, .003, .017, .12, .021] # USER TO FILL IN BASED ON BROKER REPORT (BUTTON)
npat_market = [4449.8, 5217.4, 6138.8, 7720.9, 8565.7] # USER TO FILL IN BASED ON BROKER REPORT (BUTTON)
cost_of_capital_forecasts = [.078, .078, .078, .078, .078] # USER TO FILL IN BASED ON BROKER REPORT (BUTTON)
# Non-interest debt % liabilities
non_interest_debt_percentage_liabilities = [.25, .25, .25, .25, .25] # USER TO FILL IN BASED ON BROKER REPORT (BUTTON)

# Initialize projections lists with the first year's values
revenue_projections = [fy_23_revenue]
asset_projections = [fy_23_assets]
liabilities_projections = [fy_23_liabilities]

# Calculate the projections for the next years
for i in range(len(revenue_growth_market)):
    # Infer revenue forecasts
    revenue_forecasts = revenue_projections[-1] * (1 + revenue_growth_market[i])
    revenue_projections.append(revenue_forecasts)
    # Infer asset forecasts
    asset_forecasts = asset_projections[-1] * (1 + asset_growth_market[i])
    asset_projections.append(asset_forecasts)
    # Infer liabilities forecasts
    liabilities_forecasts = liabilities_projections[-1] * (1 + liability_growth_market[i])
    liabilities_projections.append(liabilities_forecasts)

# Re-index to remove the FY23 Year
revenue_projections = revenue_projections[1:]
asset_projections = asset_projections[1:]
liabilities_projections = liabilities_projections[1:]

# Compute net asset value projections
net_asset_value_projections = [asset - liability for asset, liability in zip(asset_projections, liabilities_projections)]

# Compute non-interest bearing debt projections
non_interest_debt_projections = [liabilities * percentage for liabilities, percentage in zip(liabilities_projections, non_interest_debt_percentage_liabilities)]

# Compute funds employed
funds_employed = [non_interest_debt + net_asset_value for non_interest_debt, net_asset_value in zip(non_interest_debt_projections, net_asset_value_projections)]

# Compute economic profit
economic_profit = [npat - (funds * cost_of_capital) for npat, funds, cost_of_capital in zip(npat_market, funds_employed, cost_of_capital_forecasts)]

# Compute EP/FE (economic profit to funds employed ratio)
ep_fe_market = [ep / funds for ep, funds in zip(economic_profit, funds_employed)]

# Output the results
print("Revenue Projections:", revenue_projections)
print("Asset Projections:", asset_projections)
print("Liabilities Projections:", liabilities_projections)
print("Net Asset Value Projections:", net_asset_value_projections)
print("Non-interest Debt Projections:", non_interest_debt_projections)
print("Funds Employed:", funds_employed)
print("Economic Profit:", economic_profit)
print("EP/FE:", ep_fe_market)

# Append predictions - Market
year_predictions = pd.concat([company_slice["Year"], pd.Series(year_append)], ignore_index=True)
ep_fe_predictions_market = pd.concat([company_slice["EP/FE"], pd.Series(ep_fe_market)], ignore_index=True)
revenue_growth_predictions_1_market = pd.concat([company_slice["Revenue_growth_1_f"], pd.Series(revenue_growth_market)], ignore_index=True)
revenue_growth_predictions_3_market = revenue_growth_predictions_1_market.rolling(window=3).mean()
x_axis_list_market = [(list(revenue_growth_predictions_3_market))]
y_axis_list_market = [(list(ep_fe_predictions_market))]

labels_list = [(list(year_predictions))]
# Generate the grid for each axis
x_pad = len(max(x_axis_list_market + x_axis_list_market, key=len))
x_fill_list_market = np.array([i + [0] * (x_pad - len(i)) for i in x_axis_list_market])

y_pad = len(max(y_axis_list_market + y_axis_list_market, key=len))
y_fill_list_market = np.array([i + [0] * (y_pad - len(i)) for i in y_axis_list_market])


labels_pad = len(max(labels_list, key=len))
labels_fill_list = np.array([i + [0] * (labels_pad - len(i)) for i in labels_list])

x = np.linspace(np.min(np.nan_to_num(x_fill_list_market)), np.max(np.nan_to_num(x_fill_list_market)), 100)
y = np.linspace(np.min(np.nan_to_num(y_fill_list_market)), np.max(np.nan_to_num(y_fill_list_market)), 100)

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

# Define colors for actuals, market consensus, and scenario 1
actual_color = 'blue'
consensus_color = 'red'
scenario_color = 'black'

# Plot the actual data and market consensus data
for i in range(len(x_fill_list_market)):
    for j in range(len(x_axis_list_market[i]) - 1):
        if labels_list[i][j + 1] in [2024, 2025, 2026, 2027, 2028]:
            plt.plot(x_axis_list_market[i][j:j + 2], y_axis_list_market[i][j:j + 2], '-o', color=consensus_color)
        else:
            plt.plot(x_axis_list_market[i][j:j + 2], y_axis_list_market[i][j:j + 2], '-o', color=actual_color)
    for j, txt in enumerate(labels_list[i]):
        plt.annotate(labels_list[i][j], (x_axis_list_market[i][j], y_axis_list_market[i][j]), fontsize=6, color='black')

# Add placeholder lines for the legend
plt.plot([], [], '-o', color=actual_color, label='Actuals')
plt.plot([], [], '-o', color=consensus_color, label='Market Consensus')
plt.plot([], [], '-o', color=scenario_color, label=scenario)

# Draw the rectangles and labels for the segments
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
plt.ylabel("Economic profit / funds employed")
plt.legend()
plt.savefig(plot_label)
plt.show()