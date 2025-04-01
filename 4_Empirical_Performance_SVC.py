import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
matplotlib.use('TkAgg')

make_plots = False

# Load the main data
df_full = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")
# EV EBITDA
df_full["EV_EBITDA"] = df_full["Enterprise_Value"] /df_full["EBITDA"]
# EV EBIT
df_full["EV_EBIT"] = df_full["Enterprise_Value"] /df_full["EBIT"]

# Load the inflation data
inflation_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\WACC_inputs\Global_inflation_data.csv")

# Reshape the inflation data to facilitate merging
inflation_data_melted = inflation_data.melt(id_vars='Year', var_name='Country', value_name='Inflation')

# Map the country names in the inflation data to match those in the main DataFrame
country_map = {'Australia': 'AUS', 'USA': 'USA', 'Europe': 'EURO', 'UK': 'UK', 'India': 'INDIA', 'Japan': 'JAPAN'}
inflation_data_melted['Country'] = inflation_data_melted['Country'].map(country_map)

# Merge the main DataFrame with the inflation data
df_full = pd.merge(df_full, inflation_data_melted, how='left', on=['Year', 'Country'])

# Filter data for the required years
start_year = 2015
end_year = 2024
df = df_full[(df_full['Year'] >= start_year) & (df_full['Year'] <= end_year)]

# Filter for consumer businesses
# # df = df_.loc[df_[(df_["Sector"]=="Consumer staples") & (df_["Sector"]=="Consumer Discretionary") & (df_["Sector"]=="Consumer Staples")]]
# sector_list = ["Consumer staples", "Consumer Discretionary", "Consumer Staples"]
# df = df_.loc[df_["Sector"].isin(sector_list)]

# Drop duplicates by considering only the first occurrence of each company for each year
df = df.drop_duplicates(subset=['Company_name', 'Year'])

# Apply the criteria and create a new column for whether criteria are met each year
df['SVC_Criteria_Met'] = (df['EVA_ratio_bespoke'] > 0) & (df['Revenue_growth_1_f'] > df['Inflation'])

# Group by company and count the number of years criteria is met
criteria_count = df.groupby('Company_name')['SVC_Criteria_Met'].sum()

# Create a dictionary to collect companies by the number of years they meet the criteria
svc_summary_dict = {i: [] for i in range(0, end_year - start_year + 2)}  # from 0 to 11 years

for company, count in criteria_count.items():
    svc_summary_dict[count].append(company)

# Convert dictionary into a DataFrame for display
svc_summary = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in svc_summary_dict.items()])).fillna('')
svc_summary.columns = ["SVC_0", "SVC_1", "SVC_2", "SVC_3", "SVC_4", "SVC_5", "SVC_6", "SVC_7", "SVC_8", "SVC_9", "SVC_10"]
print(svc_summary)

# Write out to local file
svc_summary.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\svc_summary_global.csv")

### Test if valuation is associated with SVC ###
# Initialize dictionary to store median PBV values for each SVC category
pbv_medians = {}
ev_ebitda_medians = {}
ev_ebit_medians = {}
# Iterate through SVC categories (columns) in svc_summary
for column in svc_summary.columns:
    # Get the list of companies for the current SVC category
    companies = svc_summary[column].dropna().tolist()
    # Filter df for these companies
    filtered_df = df[df["Company_name"].isin(companies)]
    # Compute the median PBV and store it in the dictionary
    pbv_medians[column] = filtered_df["PBV"].median()
    ev_ebitda_medians[column] = filtered_df["EV_EBITDA"].median()
    ev_ebit_medians[column] = filtered_df["EV_EBIT"].median()
# Convert the results dictionary to a DataFrame
pbv_summary = pd.DataFrame(list(pbv_medians.items()), columns=['SVC_Category', 'Median_PBV'])
# Convert the results dictionary to a DataFrame
ev_ebitda_summary = pd.DataFrame(list(ev_ebitda_medians.items()), columns=['SVC_Category', 'Median_EV_EBITDA'])
# Convert the results dictionary to a DataFrame
ev_ebit_summary = pd.DataFrame(list(ev_ebit_medians.items()), columns=['SVC_Category', 'Median_EV_EBIT'])
# Print the resulting DataFrame
print(pbv_summary)

# Save to CSV locally
ev_ebitda_summary.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\SVC_experiments\ev_ebitda.csv")

if make_plots:
    # Plot SVC vs Median PBV
    plt.plot(pbv_summary["SVC_Category"], pbv_summary["Median_PBV"])
    plt.ylabel("Average P:BV")
    plt.xlabel("SVC Years")
    plt.title("SVC Criteria met vs P:BV")
    plt.show()

    # Plot SVC vs Median EV EBITDA
    plt.plot(ev_ebitda_summary["SVC_Category"][:-1], ev_ebitda_summary["Median_EV_EBITDA"][:-1])
    plt.ylabel("Average EV:EBITDA")
    plt.xlabel("SVC Years")
    plt.title("SVC Criteria met vs EV:EBITDA")
    plt.show()

    # Plot SVC vs Median EV EBIT
    plt.plot(ev_ebit_summary["SVC_Category"][:-1], ev_ebit_summary["Median_EV_EBIT"][:-1])
    plt.ylabel("Average EV:EBIT")
    plt.xlabel("SVC Years")
    plt.title("SVC Criteria met vs EV:EBIT")
    plt.show()

import numpy as np
import pandas as pd

# Initialize dictionary to store median and mean TSR values for each SVC category
tsrs_medians = {}
tsrs_means = {}
tsrs_counts = {}

# Number of years in the period
n = 2024 - 2015  # Adjusted for the given timeframe

# Loop over columns in svc_summary
for column in svc_summary.columns:
    # Get the list of companies for the current SVC category
    companies = svc_summary[column].dropna().tolist()

    # Filter df_full for these companies in 2015 and 2024
    df_2015 = df_full[(df_full["Year"] == 2015) & (df_full["Company_name"].isin(companies))]
    df_2024 = df_full[(df_full["Year"] == 2024) & (df_full["Company_name"].isin(companies))]

    # Merge to ensure we have both years for each company
    df_tsr = df_2015[['Company_name', 'Adjusted_Stock_Price']].merge(
        df_2024[['Company_name', 'Adjusted_Stock_Price']],
        on='Company_name',
        suffixes=('_2015', '_2024')
    )

    # Compute TSR for each company
    df_tsr['TSR'] = (df_tsr['Adjusted_Stock_Price_2024'] / df_tsr['Adjusted_Stock_Price_2015']) ** (1 / n) - 1

    # Store median, mean TSR, and count for this SVC category
    tsrs_medians[column] = df_tsr['TSR'].median()
    tsrs_means[column] = df_tsr['TSR'].mean()
    tsrs_counts[column] = len(df_tsr)

# Convert the results dictionary to a DataFrame
tsrs_summary = pd.DataFrame({
    'SVC_Category': tsrs_medians.keys(),
    'Median_TSR': tsrs_medians.values(),
    'N': tsrs_counts.values()
})

# Function to safely compute median and mean, handling NaN and Inf
def safe_median(series):
    return np.median(series.replace([np.inf, -np.inf], np.nan).dropna())

def safe_mean(series):
    return np.mean(series.replace([np.inf, -np.inf], np.nan).dropna())

# Compute statistics for SVC categories 0 to 7
median_tsr_0_to_7 = safe_median(tsrs_summary.loc[tsrs_summary['SVC_Category'].isin(['SVC_0', 'SVC_1', 'SVC_2', 'SVC_3', 'SVC_4', 'SVC_5', 'SVC_6', 'SVC_7']), 'Median_TSR'])
mean_tsr_0_to_7 = safe_mean(tsrs_summary.loc[tsrs_summary['SVC_Category'].isin(['SVC_0', 'SVC_1', 'SVC_2', 'SVC_3', 'SVC_4', 'SVC_5', 'SVC_6', 'SVC_7']), 'Median_TSR'])
median_pbv_0_to_7 = safe_median(pbv_summary.loc[pbv_summary['SVC_Category'].isin(['SVC_0', 'SVC_1', 'SVC_2', 'SVC_3', 'SVC_4', 'SVC_5', 'SVC_6', 'SVC_7']), 'Median_PBV'])
mean_pbv_0_to_7 = safe_mean(pbv_summary.loc[pbv_summary['SVC_Category'].isin(['SVC_0', 'SVC_1', 'SVC_2', 'SVC_3', 'SVC_4', 'SVC_5', 'SVC_6', 'SVC_7']), 'Median_PBV'])

# Compute statistics for SVC categories 8 to 10
median_tsr_8_to_10 = safe_median(tsrs_summary.loc[tsrs_summary['SVC_Category'].isin(['SVC_8', 'SVC_9', 'SVC_10']), 'Median_TSR'])
mean_tsr_8_to_10 = safe_mean(tsrs_summary.loc[tsrs_summary['SVC_Category'].isin(['SVC_8', 'SVC_9', 'SVC_10']), 'Median_TSR'])
median_pbv_8_to_10 = safe_median(pbv_summary.loc[pbv_summary['SVC_Category'].isin(['SVC_8', 'SVC_9', 'SVC_10']), 'Median_PBV'])
mean_pbv_8_to_10 = safe_mean(pbv_summary.loc[pbv_summary['SVC_Category'].isin(['SVC_8', 'SVC_9', 'SVC_10']), 'Median_PBV'])

# Display results
tsr_summary = pd.DataFrame({
    'Group': ['SVC_0-7', 'SVC_8-10'],
    'Median_TSR': [median_tsr_0_to_7, median_tsr_8_to_10],
    'Mean_TSR': [mean_tsr_0_to_7, mean_tsr_8_to_10],
    'Median_PBV': [median_pbv_0_to_7, median_pbv_8_to_10],
    'Mean_PBV': [mean_pbv_0_to_7, mean_pbv_8_to_10]
})


# # quick fix
# df_filtered = df
#
# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
#
# # Sector metric mapping
# sector_metric_mapping = {
#     "Banking": "PE",
#     "Investment and Wealth": "PE",
#     "Insurance": "PE",
#     "Financials - other": "PE"
# }
#
# # Initialize results list
# results = []
#
# # Loop over each company
# for company in df_filtered['Company_name'].unique():
#     try:
#         company_data = df_filtered[df_filtered['Company_name'] == company]
#         sector = company_data['Sector'].iloc[0]
#         country = company_data['Country'].iloc[0]
#
#         # Skip companies with negative EBITDA or Enterprise_Value
#         if (company_data[['EBITDA', 'Enterprise_Value']] <= 0).any().any():
#             continue
#
#         # Determine metric for regression
#         metric = sector_metric_mapping.get(sector, "PBV")
#
#         # Filter data for 2014-2024
#         company_data = company_data[(company_data['Year'] >= 2014) & (company_data['Year'] <= 2024)]
#
#         # Fit regression on selected metric
#         if not company_data[['Year', metric]].dropna().empty:
#             X = sm.add_constant(company_data['Year'])
#             y = company_data[metric]
#             model = sm.OLS(y, X, missing='drop').fit()
#             regression_coef = model.params['Year']
#         else:
#             continue
#
#         # Calculate means
#         avg_revenue_growth = company_data['Revenue_growth_3_f'].mean()
#         avg_eva_ratio = company_data['EVA_ratio_bespoke'].mean()
#         average_tsr = company_data['TSR_CIQ_no_buybacks'].mean()
#
#         # Store results
#         results.append([company, sector, country, regression_coef, avg_revenue_growth, avg_eva_ratio, average_tsr])
#
#     except Exception as e:
#         print(f"Error processing company {company}: {e}")
#         continue
#
# # Convert results to DataFrame
# results_df = pd.DataFrame(results, columns=['Company_name', 'Sector', 'Country', 'Regression_Coefficient', 'Avg_Revenue_Growth',
#                                             'Avg_EVA_Ratio', 'Average_TSR'])
#
# results_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Regression_Results.csv")