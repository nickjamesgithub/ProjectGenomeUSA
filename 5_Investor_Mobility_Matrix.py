import numpy as np
import pandas as pd
from Utilities import compute_percentiles, firefly_plot, geometric_return
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import glob
import os
import seaborn as sns

# Parameters
make_plots = True
top_100_companies = False
asx_100 = False
asx_200 = True

# Define parameters
start_year = 2013
end_year = 2023
feature_name = 'Economic_profit'  # Change this to any feature you want to analyze

def jaccard_similarity(set1, set2):
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))

# Function to assign quintiles
def assign_quintiles(df, column):
    return pd.qcut(df[column], 5, labels=False) + 1

# Function to calculate quintile averages
def calculate_quintile_averages(df, quintile_column, value_column):
    quintile_averages = df.groupby(quintile_column)[value_column].mean().sort_index()
    return quintile_averages

# Function to create transition matrix
def create_transition_matrix(initial_quintiles, final_quintiles):
    transition_matrix = pd.crosstab(initial_quintiles, final_quintiles, normalize='index')
    return transition_matrix

# Plotting function
def plot_power_curve(df_sorted, start_year, end_year, feature_name):
    # Calculate quintile positions
    quintiles = np.percentile(df_sorted.index, [20, 40, 60, 80])

    # Calculate average feature value within each quintile
    quintile_averages = [
        df_sorted[feature_name][:int(quintiles[0])].mean(),
        df_sorted[feature_name][int(quintiles[0]):int(quintiles[1])].mean(),
        df_sorted[feature_name][int(quintiles[1]):int(quintiles[2])].mean(),
        df_sorted[feature_name][int(quintiles[2]):int(quintiles[3])].mean(),
        df_sorted[feature_name][int(quintiles[3]):].mean()
    ]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted[feature_name].values, marker='o', linestyle='-', color='b')

    # Add vertical lines for quintiles
    for quintile in quintiles:
        plt.axvline(x=quintile, color='r', linestyle='--')

    # Add text annotations for average feature value within each quintile
    for i, quintile in enumerate(quintiles):
        if i == 0:
            x_pos = quintile / 2
            y_pos = quintile_averages[i]
            plt.text(x_pos, y_pos, f'Avg: {y_pos:.2f}', ha='center', va='bottom')
        else:
            x_pos = (quintiles[i-1] + quintile) / 2
            y_pos = quintile_averages[i]
            plt.text(x_pos, y_pos, f'Avg: {y_pos:.2f}', ha='center', va='bottom')

    # Last quintile annotation
    x_pos = (quintiles[-1] + len(df_sorted)) / 2
    y_pos = quintile_averages[-1]
    plt.text(x_pos, y_pos, f'Avg: {y_pos:.2f}', ha='center', va='bottom')

    plt.title(f'{feature_name.capitalize()} Power Curve ({start_year}-{end_year})')
    plt.xlabel('Index')
    plt.ylabel(f'{feature_name.capitalize()}')
    plt.grid(True)
    plt.show()

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

# Import data
mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\company_list_asx200_.csv")
# Get unique tickers
unique_tickers = mapping_data["Ticker"].unique()
# Choose sectors to include
sector = mapping_data["Sector_new"].values
# Required tickers
tickers_ = mapping_data.loc[mapping_data["Sector_new"].isin(sector)]["Ticker"].values

if asx_200:
    dfs_list = []
    for i in range(len(tickers_)):
        company_i = tickers_[i]
        try:
            df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\platform_data\_" + company_i + ".csv")
            dfs_list.append(df)
            print("Company data ", company_i)
        except:
            print("Error with company ", company_i)

if asx_100:
    # Directory containing the CSV files
    directory_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Empirical_analysis\ASX100_data"
    # List to store the dataframes
    dfs_list = []
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    # Read each CSV file and append to dfs_list
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs_list.append(df)
            # print(f"Successfully read {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"Error reading {os.path.basename(csv_file)}: {e}")

    # Optionally, print the list of successfully read dataframes
    print(f"Total files read: {len(dfs_list)}")

# Merge dataframes
df_concat = pd.concat(dfs_list)
df_merge = generate_genome_classification_df(df_concat)
# Create feature for Price-to-book
df_merge["Price_to_Book"] = df_merge["Market_Capitalisation"]/df_merge["Book_Value_Equity"]

# Filter the dataframe based on the year range
df_filtered = df_merge[(df_merge['Year'] >= start_year) & (df_merge['Year'] <= end_year)]
# Remove NaN values
df_cleaned = df_filtered.dropna(subset=[feature_name])
# Sort by the selected feature column
df_sorted = df_cleaned.sort_values(by=feature_name).reset_index(drop=True)
# Assign quintiles
df_sorted[f'{feature_name}_quintile'] = assign_quintiles(df_sorted, feature_name)
# Plot economic power curve
plot_power_curve(df_sorted, start_year, end_year, feature_name)

# Define the grid of years
grid = np.linspace(start_year, end_year, 11)

# Function to assign quintiles
def assign_quintiles(values):
    # Define percentile boundaries
    percentiles = np.percentile(values, [20, 40, 60, 80])
    # Assign quintiles based on percentiles
    return np.digitize(values, percentiles) + 1

# Loop through each year in the grid
for year in grid:
    # Filter the data for the specific year
    df_clean_i = df_cleaned[df_cleaned["Year"] == year]
    # Compute the quantiles and assign them to a new column
    df_clean_i['EP_Quantiles'] = assign_quintiles(df_clean_i["Economic_profit"].values)
    df_clean_i['EPFE_Quantiles'] = assign_quintiles(df_clean_i["EP/FE"].values)
    df_clean_i['MCAP_Quantiles'] = assign_quintiles(df_clean_i["Market_Capitalisation"].values)
    # Update the main dataframe with the quantile information for various attributes
    df_cleaned.loc[df_cleaned["Year"] == year, 'EP_Quantiles'] = df_clean_i['EP_Quantiles']
    df_cleaned.loc[df_cleaned["Year"] == year, 'EPFE_Quantiles'] = df_clean_i['EPFE_Quantiles']
    df_cleaned.loc[df_cleaned["Year"] == year, 'MCAP_Quantiles'] = df_clean_i['MCAP_Quantiles']

# Function to generate transition matrix
def generate_transition_matrix(df, feature_name, quantile_column):
    # Ensure the dataframe is sorted by Company_name and Year
    df = df.sort_values(by=['Company_name', 'Year'])
    # Create a column for the next year's quantile
    df['Next_Quantile'] = df.groupby('Company_name')[quantile_column].shift(-1)
    # Filter out rows where the next quantile is NaN (i.e., the last year for each company)
    df_transitions = df.dropna(subset=['Next_Quantile'])
    # Initialize the transition matrix
    num_quantiles = df[quantile_column].nunique()
    transition_matrix = np.zeros((num_quantiles, num_quantiles))
    # Populate the transition matrix
    for _, row in df_transitions.iterrows():
        current_quantile = int(row[quantile_column]) - 1
        next_quantile = int(row['Next_Quantile']) - 1
        transition_matrix[current_quantile, next_quantile] += 1
    # Normalize the transition matrix to get probabilities
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    # Create a DataFrame for better readability
    transition_matrix_df = pd.DataFrame(transition_matrix, index=range(1, num_quantiles + 1), columns=range(1, num_quantiles + 1))
    return transition_matrix_df

# Generate transition matrices for EP/FE and Economic_profit
transition_matrix_epfe = generate_transition_matrix(df_cleaned, 'EP/FE', 'EPFE_Quantiles')
transition_matrix_economic_profit = generate_transition_matrix(df_cleaned, 'Economic_profit', 'EP_Quantiles')

###todo GET ALL OF THE KEY MCKINSEY VARIABLES AND SHOW HOW THE COMPANY IS EVOLVING OVER TIME FOR EACH OF THOSE FEATURES###

### Specific company visualisation ###
company_name = "Telstra Corporation Limited"
# Extract values for the specific company
company_values = df_merge[(df_merge['Company_name'] == company_name) & (df_merge['Year'] >= start_year) & (df_merge['Year'] <= end_year)]

# Filter the main dataframe to include only the years 2010 to 2023
df_merge_filtered = df_merge[(df_merge['Year'] >= start_year) & (df_merge['Year'] <= end_year)]

# Create the figure and axes
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot data for Market_Capitalisation
axs[0].scatter(df_merge_filtered['Year'], df_merge_filtered['Market_Capitalisation'], label='Market_Capitalisation', color='blue', alpha=0.6)
axs[0].scatter(company_values['Year'], company_values['Market_Capitalisation'], color='red', s=100, label=f'{company_name} Market_Capitalisation', marker='x')
axs[0].set_ylabel('Market_Capitalisation')
axs[0].legend()

# Plot data for Debt_percentage
axs[1].scatter(df_merge_filtered['Year'], df_merge_filtered['Debt_percentage'], label='Debt_percentage', color='green', alpha=0.6)
axs[1].scatter(company_values['Year'], company_values['Debt_percentage'], color='red', s=100, label=f'{company_name} Debt_percentage', marker='x')
axs[1].set_ylabel('Debt_percentage')
axs[1].legend()

# Plot data for ROTE
axs[2].scatter(df_merge_filtered['Year'], df_merge_filtered['ROTE'], label='ROTE', color='orange', alpha=0.6)
axs[2].scatter(company_values['Year'], company_values['ROTE'], color='red', s=100, label=f'{company_name} ROTE', marker='x')
axs[2].set_ylabel('ROTE')
axs[2].legend()

# Set the x-axis label
plt.xlabel('Year')

# Set the title for the entire figure
fig.suptitle('Features Plot with Highlighted Company Values', fontsize=16)

# Adjust the layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()


"""
Key metrics to study:
1 - Get mobility metrics
# Endowment
- size
- debt level
- r&d % sales
# Trends
- Trend (industry and geography)...industry trend is most important
# Moves
- M&A
- Resource allocation
- Capex % sales
- Productivity improvement
- Differentiation improvement
2 - Fit a boosting algorithm and include all features + sector
3 - Study evolution in regression coefficients over time
"""

