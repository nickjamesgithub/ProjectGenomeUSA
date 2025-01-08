import numpy as np
import pandas as pd
from Utilities import compute_percentiles, firefly_plot, geometric_return
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio
pio.renderers.default = 'browser'
from sklearn.linear_model import LinearRegression

make_plots = True

# Apply Genome Filter
asx_200 = True

# Market capitalisation threshold
mcap_threshold = 500

def generate_genome_classification_df(df):
    # Conditions EP/FE
    conditions_genome = [
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"] < 0),
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
        (df["EP/FE"] < 0) & (df["Revenue_growth_3_f"] >= 0.20),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"] < 0),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
        (df["EP/FE"] > 0) & (df["Revenue_growth_3_f"] >= 0.20)
    ]

    # Values to display
    values_genome = ["UNTENABLE", "TRAPPED", "BRAVE", "FEARLESS", "CHALLENGED", "VIRTUOUS", "FAMOUS", "LEGENDARY"]

    df["Genome_classification"] = np.select(conditions_genome, values_genome)

    return df

matplotlib.use('TkAgg')

# Initialise years
beginning_year = 2011
end_year = 2023
# Generate grid of years
year_grid = np.linspace(beginning_year, end_year, end_year-beginning_year+1)
rolling_window = 3

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Company_list_GPT_SP500.csv")

# Get unique tickers
unique_tickers = mapping_data["Ticker"].unique()

# Choose sectors to include
sector = mapping_data["Sector_new"].values

# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

dfs_list = []
for i in range(len(tickers_)):
    company_i = tickers_[i]
    try:
        df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\_" + company_i + ".csv")
        dfs_list.append(df)
        print("Company data ", company_i)
    except:
        print("Error with company ", company_i)

# Merge dataframes
df_concat = pd.concat(dfs_list)
df_merge = generate_genome_classification_df(df_concat)

# Get unique tickers
unique_tickers = df_merge["Ticker"].unique()

# Define the year range for filtering
start_year = 2014
end_year = 2023

# Filter the dataframe by the year range (for other plots)
df_filtered = df_merge[(df_merge['Year'] >= start_year) & (df_merge['Year'] <= end_year)]

# Group by 'Sector' and calculate the median 'EP/FE'
df_grouped_epfe_sector = df_filtered.groupby('Sector')['EP/FE'].median().reset_index()

# Group by 'Sector' and calculate the median 'TSR_CIQ_no_buybacks'
df_grouped_tsr_sector = df_filtered[df_filtered['TSR_CIQ_no_buybacks'] != -1.0].groupby('Sector')['TSR_CIQ_no_buybacks'].median().reset_index()

# Group by 'Genome_classification' and calculate the median 'EP/FE' and 'TSR_CIQ_no_buybacks'
df_grouped_genome = df_filtered[df_filtered['Genome_classification'] != '0'].groupby('Genome_classification')[['EP/FE', 'TSR_CIQ_no_buybacks']].median().reset_index()

# Create a figure with three subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Average EP/FE per Sector
axes[0, 0].bar(df_grouped_epfe_sector['Sector'], df_grouped_epfe_sector['EP/FE'], color='skyblue')
axes[0, 0].set_title('Median EP/FE per Sector ({} - {})'.format(start_year, end_year))
axes[0, 0].set_xlabel('Sector')
axes[0, 0].set_ylabel('Median EP/FE')
axes[0, 0].tick_params(axis='x', rotation=90)

# Subplot 2: Median TSR_CIQ_no_buybacks per Sector
axes[0, 1].bar(df_grouped_tsr_sector['Sector'], df_grouped_tsr_sector['TSR_CIQ_no_buybacks'], color='lightgreen')
axes[0, 1].set_title('Median TSR per Sector ({} - {})'.format(start_year, end_year))
axes[0, 1].set_xlabel('Sector')
axes[0, 1].set_ylabel('Median TSR_CIQ_no_buybacks')
axes[0, 1].tick_params(axis='x', rotation=90)

# Subplot 3: Median EP/FE and TSR_CIQ_no_buybacks per Genome Classification
bar_width = 0.35
index = range(len(df_grouped_genome['Genome_classification']))
axes[1, 0].bar(index, df_grouped_genome['EP/FE'], bar_width, label='EP/FE', color='skyblue')
axes[1, 0].bar([i + bar_width for i in index], df_grouped_genome['TSR_CIQ_no_buybacks'], bar_width, label='TSR', color='lightgreen')
axes[1, 0].set_title('Median EP/FE and TSR per Genome Classification ({} - {})'.format(start_year, end_year))
axes[1, 0].set_xlabel('Genome Classification')
axes[1, 0].set_ylabel('Median Value')
axes[1, 0].set_xticks([i + bar_width / 2 for i in index])
axes[1, 0].set_xticklabels(df_grouped_genome['Genome_classification'])
axes[1, 0].legend()

# Subplot 4: Scatter plot with regression line of EP/FE vs TSR_CIQ_no_buybacks (5th - 95th Percentile)
# Calculate percentiles
lower_percentile_epfe = df_merge['EP/FE'].quantile(0.1)
upper_percentile_epfe = df_merge['EP/FE'].quantile(0.9)
lower_percentile_tsr = df_merge['TSR_CIQ_no_buybacks'].quantile(0.1)
upper_percentile_tsr = df_merge['TSR_CIQ_no_buybacks'].quantile(0.9)

# Filter data within percentiles
df_filtered_percentile = df_merge[
    (df_merge['EP/FE'] >= lower_percentile_epfe) & (df_merge['EP/FE'] <= upper_percentile_epfe) &
    (df_merge['TSR_CIQ_no_buybacks'] >= lower_percentile_tsr) & (df_merge['TSR_CIQ_no_buybacks'] <= upper_percentile_tsr)
]

# Scatter plot with regression line using Seaborn
model = LinearRegression()
m_ = model.fit(np.array(df_filtered_percentile["EP/FE"]).reshape(-1,1), np.array(df_filtered_percentile["TSR_CIQ_no_buybacks"]).reshape(-1,1))

# Compute intercept
intercept = m_.intercept_[0]

# Add vertical line at EP/FE = 0
plt.axvline(x=0, color='red', linestyle='--')

sns.regplot(x='EP/FE', y='TSR_CIQ_no_buybacks', data=df_filtered_percentile, scatter_kws={'s': 10, 'alpha':0.2, 'color':'orange'}, ax=axes[1, 1])

# Add labels and title
axes[1, 1].set_title('EP/FE vs TSR (5th - 95th Percentile)')
axes[1, 1].set_xlabel('EP/FE')
axes[1, 1].set_ylabel('TSR')

# Annotate expected TSR for companies breaking above the line
axes[1, 1].text(0.5, 0.9, f'Expected TSR for companies at the line:\n{intercept * 100 :.2f}% (intercept)', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)

# Adjust layout to make room for the labels
plt.tight_layout()

# Display the plots
plt.savefig("Market_EP_breakdown")
plt.show()

# Group by 'Sector' and calculate the median 'Revenue_growth_3_f'
df_grouped_revenue_sector = df_filtered.groupby('Sector')['Revenue_growth_3_f'].median().reset_index()

# Group by 'Sector' and calculate the median 'TSR_CIQ_no_buybacks'
df_grouped_tsr_sector = df_filtered[df_filtered['TSR_CIQ_no_buybacks'] != -1.0].groupby('Sector')['TSR_CIQ_no_buybacks'].median().reset_index()

# Group by 'Genome_classification' and calculate the median 'Revenue_growth_3_f' and 'TSR_CIQ_no_buybacks'
df_grouped_genome = df_filtered[df_filtered['Genome_classification'] != '0'].groupby('Genome_classification')[['Revenue_growth_3_f', 'TSR_CIQ_no_buybacks']].median().reset_index()

# Create the first figure with three subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Median Revenue_growth_3_f per Sector
axes[0, 0].bar(df_grouped_revenue_sector['Sector'], df_grouped_revenue_sector['Revenue_growth_3_f'], color='skyblue')
axes[0, 0].set_title('Median Revenue Growth per Sector ({} - {})'.format(start_year, end_year))
axes[0, 0].set_xlabel('Sector')
axes[0, 0].set_ylabel('Median Revenue Growth')
axes[0, 0].tick_params(axis='x', rotation=90)

# Subplot 2: Median TSR_CIQ_no_buybacks per Sector
axes[0, 1].bar(df_grouped_tsr_sector['Sector'], df_grouped_tsr_sector['TSR_CIQ_no_buybacks'], color='lightgreen')
axes[0, 1].set_title('Median TSR per Sector ({} - {})'.format(start_year, end_year))
axes[0, 1].set_xlabel('Sector')
axes[0, 1].set_ylabel('Median TSR_CIQ_no_buybacks')
axes[0, 1].tick_params(axis='x', rotation=90)

# Subplot 3: Median Revenue_growth_3_f and TSR_CIQ_no_buybacks per Genome Classification
bar_width = 0.35
index = range(len(df_grouped_genome['Genome_classification']))
axes[1, 0].bar(index, df_grouped_genome['Revenue_growth_3_f'], bar_width, label='Revenue Growth', color='skyblue')
axes[1, 0].bar([i + bar_width for i in index], df_grouped_genome['TSR_CIQ_no_buybacks'], bar_width, label='TSR', color='lightgreen')
axes[1, 0].set_title('Median Revenue Growth and TSR per Genome Classification ({} - {})'.format(start_year, end_year))
axes[1, 0].set_xlabel('Genome Classification')
axes[1, 0].set_ylabel('Median Value')
axes[1, 0].set_xticks([i + bar_width / 2 for i in index])
axes[1, 0].set_xticklabels(df_grouped_genome['Genome_classification'])
axes[1, 0].legend()

# Subplot 4: Scatter plot with regression line of Revenue_growth_3_f vs TSR_CIQ_no_buybacks (5th - 95th Percentile)
# Calculate percentiles
lower_percentile_revenue = df_merge['Revenue_growth_3_f'].quantile(0.1)
upper_percentile_revenue = df_merge['Revenue_growth_3_f'].quantile(0.9)
lower_percentile_tsr = df_merge['TSR_CIQ_no_buybacks'].quantile(0.1)
upper_percentile_tsr = df_merge['TSR_CIQ_no_buybacks'].quantile(0.9)

# Filter data within percentiles
df_filtered_percentile = df_merge[
    (df_merge['Revenue_growth_3_f'] >= lower_percentile_revenue) & (df_merge['Revenue_growth_3_f'] <= upper_percentile_revenue) &
    (df_merge['TSR_CIQ_no_buybacks'] >= lower_percentile_tsr) & (df_merge['TSR_CIQ_no_buybacks'] <= upper_percentile_tsr)
]

# Scatter plot with regression line using Seaborn
model = LinearRegression()
m_ = model.fit(np.array(df_filtered_percentile["Revenue_growth_3_f"]).reshape(-1,1), np.array(df_filtered_percentile["TSR_CIQ_no_buybacks"]).reshape(-1,1))

# Compute intercept
intercept = m_.intercept_[0]

# Add vertical line at Revenue_growth_3_f = 0
plt.axvline(x=0, color='red', linestyle='--')

sns.regplot(x='Revenue_growth_3_f', y='TSR_CIQ_no_buybacks', data=df_filtered_percentile, scatter_kws={'s': 10, 'alpha':0.2, 'color':'orange'}, ax=axes[1, 1])

# Add labels and title
axes[1, 1].set_title('Revenue Growth vs TSR (5th - 95th Percentile)')
axes[1, 1].set_xlabel('Revenue Growth')
axes[1, 1].set_ylabel('TSR')

# Annotate expected TSR for companies breaking above the line
axes[1, 1].text(0.5, 0.9, f'Expected TSR for companies at the line:\n{intercept * 100 :.2f}% (intercept)', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)

# Adjust layout to make room for the labels
plt.tight_layout()

# Display the plots
plt.savefig("Market_Revenue_breakdown")
plt.show()
