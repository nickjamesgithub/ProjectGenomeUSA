import numpy as np
import pandas as pd
from Utilities import compute_percentiles, firefly_plot, geometric_return
import matplotlib
from Utilities import generate_market_cap_class, waterfall_value_plot, geometric_return, dendrogram_plot, get_even_clusters
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math
import glob
import os

make_plots = True

# Apply Genome Filter
genome_filtering = True
sp_500 = True
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


def sector_functions_mobility_matrix(df):
    unique_companies = df["Company_name"].unique()
    features_list = []
    for i in range(len(unique_companies)):
        # Slice for each company within sector slice
        slice_i = df.loc[df["Company_name"] == unique_companies[i]]
        ### Endowment ###
        # Revenue
        revenue_avg = slice_i["Revenue"].iloc[1:].mean()
        # Revenue CAGR
        revenue_cagr = (slice_i["Revenue"].iloc[-1] / slice_i["Revenue"].iloc[0]) ** (1 / (len(slice_i)-1)) - 1
        # TSR
        tsr = (slice_i["Adjusted_Stock_Price"].iloc[-1] / slice_i["Adjusted_Stock_Price"].iloc[0]) ** (1 / (len(slice_i) - 1)) - 1
        # Leverage
        leverage = slice_i["Debt_to_equity"].iloc[1:].mean()
        # Investments (R&D % Sales)
        investment = slice_i["RD/Revenue"].iloc[1:].mean()

        ### Trends ###
        ep_fe_avg = slice_i["EP/FE"].iloc[1:].mean()

        ### Moves ###
        # Programmatic M&A
        acquisition_propensity = abs(slice_i["Cash_acquisitions"].iloc[1:]).sum() / slice_i["Market_Capitalisation"].iloc[1:].mean()
        # Dynamic allocation of resources

        # CAPEX (capex % sales)
        capex_per_revenue_avg = slice_i["CAPEX/Revenue"].iloc[1:].mean()
        # Labor productivity (npat per employee)
        npat_per_employee_avg = slice_i["NPAT_per_employee"].iloc[1:].mean()
        # Improvements in Differentiation (Gross Margin)
        gross_margin_avg = slice_i["Gross_margin"].iloc[1:].mean()

        # Append all features to master list
        features_list.append([revenue_avg, revenue_cagr, leverage, investment, ep_fe_avg,  acquisition_propensity,
                              capex_per_revenue_avg, npat_per_employee_avg, gross_margin_avg, tsr])

    # Turn this into a dataframe
    features_df = pd.DataFrame(features_list)
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Calculate the mean, excluding NaN values
    mean_values = features_df.mean(skipna=True)
    return mean_values

def company_functions_mobility_matrix(df):
    # Slice for each company within sector slice
    ### Endowment ###
    # Revenue
    revenue_avg = df["Revenue"].iloc[1:].mean()
    # Revenue CAGR
    revenue_cagr = (df["Revenue"].iloc[-1] / df["Revenue"].iloc[0]) ** (1 / (len(df)-1)) - 1
    # TSR
    tsr = (df["Adjusted_Stock_Price"].iloc[-1] / df["Adjusted_Stock_Price"].iloc[0]) ** (1 / (len(df)-1)) - 1
    # Leverage
    leverage = df["Debt_to_equity"].iloc[1:].mean()
    # Investments (R&D % Sales)
    investment = df["RD/Revenue"].iloc[1:].mean()

    ### Trends ###
    ep_fe_avg = df["EP/FE"].iloc[1:].mean()

    ### Moves ###
    # Programmatic M&A
    acquisition_propensity = abs(df["Cash_acquisitions"].iloc[1:]).sum() / df["Market_Capitalisation"].iloc[1:].mean()
    # Dynamic allocation of resources

    # CAPEX (capex % sales)
    capex_per_revenue_avg = df["CAPEX/Revenue"].iloc[1:].mean()
    # Labor productivity (npat per employee)
    npat_per_employee_avg = df["NPAT_per_employee"].iloc[1:].mean()
    # Improvements in Differentiation (Gross Margin)
    gross_margin_avg = df["Gross_margin"].iloc[1:].mean()

    return revenue_avg, revenue_cagr, leverage, investment, ep_fe_avg, acquisition_propensity, capex_per_revenue_avg, npat_per_employee_avg, gross_margin_avg, tsr

matplotlib.use('TkAgg')

# Initialise years
beginning_year = 2013
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

if sp_500:
    # Directory containing the CSV files
    directory_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data"
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

# Get unique tickers
unique_tickers = df_merge["Ticker"].unique()

# Features to slice
features = ["Company_name", "Sector", "Year", "TSR", "Revenue_growth_3_f", "Stock_Price", "Adjusted_Stock_Price","DPS", "BBPS",
            "DBBPS", "Genome_classification", "Price_to_Book", "PE_Implied", "Market_Capitalisation",
            "Revenue", "Debt_to_equity", "RD/Revenue",
            "EP/FE",
            "Cash_acquisitions", "NPAT_per_employee", "CAPEX/Revenue", "Gross_margin"]

# Loop over all years
df_list = []
df_issue_list = []
for i in range(len(year_grid)-rolling_window+1):
    # Loop over all companies
    for j in range(len(unique_tickers)):
        try:
            # Get years... think of indexing as Years 0, 1 and 2
            year_i = year_grid[i]
            year_i_2 = year_grid[i+rolling_window]

            # Slice dataframes based on company and year
            df_slice_i = df_merge.loc[(df_merge["Year"]==year_i) & (df_merge["Ticker"]==unique_tickers[j])][features]
            df_slice_i_2 = df_merge.loc[(df_merge["Year"] == year_i_2) & (df_merge["Ticker"]==unique_tickers[j])][features]

            # Slice company name
            company_name_j = df_merge.loc[(df_merge["Year"] == year_i_2) & (df_merge["Ticker"]==unique_tickers[j])]["Company_name"].values[0]
            unique_ticker_j = unique_tickers[j]
            sector_j = df_slice_i["Sector"].values[0]

            # Slice dataframe based on entire date range for sector and company
            df_slice_company_range_i = df_merge.loc[(df_merge["Year"] >= year_i) & (df_merge["Year"] <= year_i_2) &
                                          (df_merge["Ticker"] == unique_tickers[j])][features]
            df_slice_sector_range_i = df_merge.loc[(df_merge["Year"] >= year_i) & (df_merge["Year"] <= year_i_2) &
                                          (df_merge["Sector"] == sector_j)][features]

            # Compute sector mobility matrix
            sector_revenue_avg, sector_revenue_cagr, sector_leverage, sector_investment,\
                sector_epfe_avg, \
            sector_acquisition_propensity, sector_capex_per_revenue, sector_npat_per_employee, sector_gross_margin, sector_tsr = sector_functions_mobility_matrix(df_slice_sector_range_i)

            # Compute Company mobility matrix
            company_revenue_avg, company_revenue_cagr, company_leverage, company_investment,\
                company_epfe_avg,  \
            company_acquisition_propensity, company_capex_per_revenue, company_npat_per_employee, company_gross_margin, company_tsr = company_functions_mobility_matrix(df_slice_company_range_i)

            # Compute mobility matrix delta
            delta_revenue_avg = company_revenue_avg - sector_revenue_avg
            delta_revenue_cagr= company_revenue_cagr - sector_revenue_cagr
            delta_leverage= company_leverage - sector_leverage
            delta_investment= company_investment - sector_investment
            delta_epfe_avg= company_epfe_avg - sector_epfe_avg
            delta_acquisition_propensity= company_acquisition_propensity - sector_acquisition_propensity
            delta_capex_per_revenue= company_capex_per_revenue - sector_capex_per_revenue
            delta_npat_per_employee= company_npat_per_employee - sector_npat_per_employee
            delta_gross_margin= company_gross_margin - sector_gross_margin

            # Get Genome classification
            genome_classification_beginning = df_slice_i["Genome_classification"].values[0]
            genome_classification_end = df_slice_i_2["Genome_classification"].values[0]

            # Get the P/E
            pe_implied_beginning = df_slice_i["PE_Implied"].values[0]
            pe_implied_end = df_slice_i_2["PE_Implied"].values[0]

            # Get Market Capitalisation
            market_capitalisation_beginning = df_slice_i["Market_Capitalisation"].values[0]
            market_capitalisation_end = df_slice_i_2["Market_Capitalisation"].values[0]

            # Get coordinates of firefly with X & Y
            firefly_y_beginning = df_slice_i["EP/FE"].values[0]
            firefly_y_end = df_slice_i_2["EP/FE"].values[0]
            firefly_x_beginning = df_slice_i["Revenue_growth_3_f"].values[0]
            firefly_x_end = df_slice_i_2["Revenue_growth_3_f"].values[0]

            # Compute angle between X & Y - first in radians and then in degrees
            radians = math.atan2(firefly_y_end - firefly_y_beginning,
                                 firefly_x_end - firefly_x_beginning)  # (End Y - Begin Y, End X - Begin X)
            degree_angle = math.degrees(radians)
            degrees = (degree_angle + 360) % 360  # for implementations where mod returns negative numbers

            # Get all Dividends and buybacks in that range
            dbbps_slice = df_merge.loc[(df_merge["Year"] >= year_i) &
                                        (df_merge["Year"] <= year_i_2) &
                                        (df_merge["Ticker"]==unique_tickers[j])]["DBBPS"]
            # Sum over all dividends & buybacks per share
            dbbps_total = dbbps_slice.values.sum()

            ### Scenario 1 - Remain Negative ###
            if df_slice_i["EP/FE"].values[0] < 0 and df_slice_i_2["EP/FE"].values[0] < 0:
                print("Company " + unique_ticker_j + " Remain negative scenario " + "Years " + str(year_i) + "-" + str(year_i_2))

                # Compute Cumulative & Annualized TSR
                cumulative_tsr = (df_slice_i_2["Stock_Price"].values[0] - df_slice_i["Stock_Price"].values[0] + dbbps_total) / df_slice_i["Stock_Price"].values[0]
                annualized_tsr = (1 + cumulative_tsr) ** (1 / rolling_window) - 1

                # Compute cumulative & annualized TSR - Capiq method
                cumulative_tsr_capiq = df_slice_i_2["Adjusted_Stock_Price"].values[0]/df_slice_i["Adjusted_Stock_Price"].values[0] - 1
                annualized_tsr_capiq = (1 + cumulative_tsr_capiq) ** (1 / rolling_window) - 1

                # Append Direction, company name, Year_0, Year_2, EP/FE_0, EP/FE_2, Stock Price_0, Stock_Price_2, DBBPS, Annualized TSR
                df_list.append(["Remain_negative", genome_classification_beginning, genome_classification_end, company_name_j, sector_j, int(year_i), int(year_i_2), df_slice_i["Revenue_growth_3_f"].values[0], df_slice_i_2["Revenue_growth_3_f"].values[0],
                                df_slice_i["EP/FE"].values[0],df_slice_i_2["EP/FE"].values[0],
                                df_slice_i["Stock_Price"].values[0], df_slice_i_2["Stock_Price"].values[0], df_slice_i["Price_to_Book"].values[0], dbbps_total, degrees, radians,
                                annualized_tsr, annualized_tsr_capiq, pe_implied_beginning, pe_implied_end, market_capitalisation_beginning, market_capitalisation_end,
                                company_revenue_avg, sector_revenue_avg, delta_revenue_avg,
                                company_revenue_cagr, sector_revenue_cagr, delta_revenue_cagr,
                                company_leverage, sector_leverage, delta_leverage,
                                company_investment, sector_investment, delta_investment,
                                company_epfe_avg, sector_epfe_avg, delta_epfe_avg,
                                company_acquisition_propensity, sector_acquisition_propensity, delta_acquisition_propensity,
                                company_capex_per_revenue, sector_capex_per_revenue, delta_capex_per_revenue,
                                company_npat_per_employee, sector_npat_per_employee, delta_npat_per_employee,
                                company_gross_margin, sector_gross_margin, delta_gross_margin, sector_tsr, company_tsr])

            ### Scenario 2 - Move Up ###
            if df_slice_i["EP/FE"].values[0] < 0 and df_slice_i_2["EP/FE"].values[0] >= 0:
                print("Company " + unique_ticker_j + " Move up scenario "+ "Years " + str(year_i) + "-" + str(year_i_2))

                # Compute Cumulative & Annualized TSR
                cumulative_tsr = (df_slice_i_2["Stock_Price"].values[0] - df_slice_i["Stock_Price"].values[0] + dbbps_total) / df_slice_i["Stock_Price"].values[0]
                annualized_tsr = (1 + cumulative_tsr) ** (1 / rolling_window) - 1

                # Compute cumulative & annualized TSR - Capiq method
                cumulative_tsr_capiq = df_slice_i_2["Adjusted_Stock_Price"].values[0]/df_slice_i["Adjusted_Stock_Price"].values[0] - 1
                annualized_tsr_capiq = (1 + cumulative_tsr_capiq) ** (1 / rolling_window) - 1

                # Append Direction, company name, Year_0, Year_2, EP/FE_0, EP/FE_2, Stock Price_0, Stock_Price_2, DBBPS, Annualized TSR
                df_list.append(["Move_up", genome_classification_beginning, genome_classification_end, company_name_j,sector_j, int(year_i), int(year_i_2), df_slice_i["Revenue_growth_3_f"].values[0], df_slice_i_2["Revenue_growth_3_f"].values[0],
                                df_slice_i["EP/FE"].values[0], df_slice_i_2["EP/FE"].values[0],
                                df_slice_i["Stock_Price"].values[0], df_slice_i_2["Stock_Price"].values[0], df_slice_i["Price_to_Book"].values[0], dbbps_total, degrees, radians,
                                annualized_tsr, annualized_tsr_capiq, pe_implied_beginning, pe_implied_end, market_capitalisation_beginning, market_capitalisation_end,
                                company_revenue_avg, sector_revenue_avg, delta_revenue_avg,
                                company_revenue_cagr, sector_revenue_cagr, delta_revenue_cagr,
                                company_leverage, sector_leverage, delta_leverage,
                                company_investment, sector_investment, delta_investment,
                                company_epfe_avg, sector_epfe_avg, delta_epfe_avg,
                                company_acquisition_propensity, sector_acquisition_propensity, delta_acquisition_propensity,
                                company_capex_per_revenue, sector_capex_per_revenue, delta_capex_per_revenue,
                                company_npat_per_employee, sector_npat_per_employee, delta_npat_per_employee,
                                company_gross_margin, sector_gross_margin, delta_gross_margin, sector_tsr, company_tsr])

            ### Scenario 3 - Remain Positive ###
            if df_slice_i["EP/FE"].values[0] >= 0 and df_slice_i_2["EP/FE"].values[0] >= 0:
                print("Company " + unique_ticker_j + " Remain positive scenario " + "Years " + str(year_i) + "-" + str(year_i_2))

                # Compute Cumulative & Annualized TSR
                cumulative_tsr = (df_slice_i_2["Stock_Price"].values[0] - df_slice_i["Stock_Price"].values[0] + dbbps_total)/df_slice_i["Stock_Price"].values[0]
                annualized_tsr = (1+cumulative_tsr)**(1/rolling_window) - 1

                # Compute cumulative & annualized TSR - Capiq method
                cumulative_tsr_capiq = df_slice_i_2["Adjusted_Stock_Price"].values[0]/df_slice_i["Adjusted_Stock_Price"].values[0] - 1
                annualized_tsr_capiq = (1 + cumulative_tsr_capiq) ** (1 / rolling_window) - 1

                # Append Direction, company name, Year_0, Year_2, EP/FE_0, EP/FE_2, Stock Price_0, Stock_Price_2, DBBPS, Annualized TSR
                df_list.append(["Remain_positive", genome_classification_beginning, genome_classification_end, company_name_j,sector_j, int(year_i), int(year_i_2), df_slice_i["Revenue_growth_3_f"].values[0], df_slice_i_2["Revenue_growth_3_f"].values[0],
                                df_slice_i["EP/FE"].values[0], df_slice_i_2["EP/FE"].values[0],
                                df_slice_i["Stock_Price"].values[0], df_slice_i_2["Stock_Price"].values[0], df_slice_i["Price_to_Book"].values[0], dbbps_total, degrees, radians,
                                annualized_tsr, annualized_tsr_capiq, pe_implied_beginning, pe_implied_end, market_capitalisation_beginning, market_capitalisation_end,
                                company_revenue_avg, sector_revenue_avg, delta_revenue_avg,
                                company_revenue_cagr, sector_revenue_cagr, delta_revenue_cagr,
                                company_leverage, sector_leverage, delta_leverage,
                                company_investment, sector_investment, delta_investment,
                                company_epfe_avg, sector_epfe_avg, delta_epfe_avg,
                                company_acquisition_propensity, sector_acquisition_propensity, delta_acquisition_propensity,
                                company_capex_per_revenue, sector_capex_per_revenue, delta_capex_per_revenue,
                                company_npat_per_employee, sector_npat_per_employee, delta_npat_per_employee,
                                company_gross_margin, sector_gross_margin, delta_gross_margin, sector_tsr, company_tsr])

            ### Scenario 4 - Move Down ###
            if df_slice_i["EP/FE"].values[0] > 0 and df_slice_i_2["EP/FE"].values[0] < 0:
                print("Company " + unique_ticker_j + " Move down scenario " + "Years " + str(year_i) + "-" + str(year_i_2))

                # Compute Cumulative & Annualized TSR
                cumulative_tsr = (df_slice_i_2["Stock_Price"].values[0] - df_slice_i["Stock_Price"].values[0] + dbbps_total) / df_slice_i["Stock_Price"].values[0]
                annualized_tsr = (1 + cumulative_tsr) ** (1 / rolling_window) - 1

                # Compute cumulative & annualized TSR - Capiq method
                cumulative_tsr_capiq = df_slice_i_2["Adjusted_Stock_Price"].values[0]/df_slice_i["Adjusted_Stock_Price"].values[0] - 1
                annualized_tsr_capiq = (1 + cumulative_tsr_capiq) ** (1 / rolling_window) - 1

                # Append Direction, company name, Year_0, Year_2, EP/FE_0, EP/FE_2, Revenue_growth_0, Revenue_growth_2, Stock Price_0, Stock_Price_2, DBBPS, Annualized TSR
                df_list.append(["Move_down",genome_classification_beginning, genome_classification_end, company_name_j, sector_j, int(year_i), int(year_i_2), df_slice_i["Revenue_growth_3_f"].values[0], df_slice_i_2["Revenue_growth_3_f"].values[0],
                                df_slice_i["EP/FE"].values[0], df_slice_i_2["EP/FE"].values[0],
                                df_slice_i["Stock_Price"].values[0], df_slice_i_2["Stock_Price"].values[0], df_slice_i["Price_to_Book"].values[0], dbbps_total, degrees, radians,
                                annualized_tsr, annualized_tsr_capiq, pe_implied_beginning, pe_implied_end, market_capitalisation_beginning, market_capitalisation_end,
                                company_revenue_avg, sector_revenue_avg, delta_revenue_avg,
                                company_revenue_cagr, sector_revenue_cagr, delta_revenue_cagr,
                                company_leverage, sector_leverage, delta_leverage,
                                company_investment, sector_investment, delta_investment,
                                company_epfe_avg, sector_epfe_avg, delta_epfe_avg,
                                company_acquisition_propensity, sector_acquisition_propensity, delta_acquisition_propensity,
                                company_capex_per_revenue, sector_capex_per_revenue, delta_capex_per_revenue,
                                company_npat_per_employee, sector_npat_per_employee, delta_npat_per_employee,
                                company_gross_margin, sector_gross_margin, delta_gross_margin, sector_tsr, company_tsr])
        except:
            print("Issue with " + company_name_j + " Years: " + str(year_i) + "-" + str(year_i_2))
            df_issue_list.append([company_name_j, year_i, year_i_2])

# Create collapsed journeys dataframe
df_journey_collapsed = pd.DataFrame(df_list)

# Create columns
df_journey_collapsed.columns = ["Journey", "Genome_classification_beginning", "Genome_classification_end", "Company_name", "Sector", "Year_beginning", "Year_final",
                                "Revenue_growth_beginning", "Revenue_growth_end", "EP/FE_beginning", "EP/FE_end",
                                "Stock_price_beginning", "Stock_price_final", "Price_to_book", "DBBPS_total", "Angle", "Radians",
                                "Annualized_TSR", "Annualized_TSR_Capiq", "PE_beginning", "PE_end", "Market_Capitalisation_beginning", "Market_Capitalisation_end",
                                "Company_revenue_avg", "Sector_revenue_avg", "Delta_revenue_avg",
                                "Company_revenue_cagr", "Sector_revenue_cagr", "Delta_revenue_cagr",
                                "Company_leverage", "Sector_leverage", "Delta_leverage",
                                "Company_investment", "Sector_investment", "Delta_investment",
                                "Company_EPFE_avg", "Sector_EPFE_avg", "delta_EPFE_avg",
                                "Company_acquisition_propensity", "Sector_acquisition_propensity", "delta_acquisition_propensity",
                                "Company_capex/revenue", "Sector_capex/revenue", "Delta_capex/revenue",
                                "Company_npat_per_employee", "Sector_npat_per_employee", "Delta_npat_per_employee",
                                "Company_gross_margin", "Sector_gross_margin", "Delta_gross_margin", "Sector_TSR", "Company_TSR"]

# Replace all inf values with nan
df_journey_collapsed = df_journey_collapsed.replace([np.inf, -np.inf], np.nan)

# Remove rows with NaN values in 'Annualized_TSR'
df_journey_collapsed = df_journey_collapsed.dropna(subset=['Annualized_TSR_Capiq'], axis=0)

# Add change in X axis and Y axis
df_journey_collapsed["X_change"] = df_journey_collapsed["Revenue_growth_end"] - df_journey_collapsed["Revenue_growth_beginning"]
df_journey_collapsed["Y_change"] = df_journey_collapsed["EP/FE_end"] - df_journey_collapsed["EP/FE_beginning"]

if genome_filtering:
    df_journey_collapsed = df_journey_collapsed.loc[(df_journey_collapsed["EP/FE_end"] >= -.3) & (df_journey_collapsed["EP/FE_end"] <= .5) &
                                     (df_journey_collapsed["Revenue_growth_end"] >= -.3) & (df_journey_collapsed["Revenue_growth_end"] <= 1.5) &
                                     (df_journey_collapsed["Annualized_TSR_Capiq"] >= -.4) & (df_journey_collapsed["Annualized_TSR_Capiq"] <= 1) &
                                     (df_journey_collapsed["Price_to_book"] > -200)]

# Write out csv file locally
df_journey_collapsed.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_Journeys_summary.csv")

# Filter out infinite and NaN values before plotting
df_journey_collapsed = df_journey_collapsed.replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN values in 'Annualized_TSR' column
df_journey_collapsed = df_journey_collapsed.dropna(subset=['Annualized_TSR_Capiq'], axis=0)

# Ensure that 'Angle' and 'Annualized_TSR' columns contain only numeric values
df_journey_collapsed['Angle'] = pd.to_numeric(df_journey_collapsed['Angle'], errors='coerce')
df_journey_collapsed['Annualized_TSR_Capiq'] = pd.to_numeric(df_journey_collapsed['Annualized_TSR_Capiq'], errors='coerce')

if make_plots:
    # Loop over unique values in "Journey" column
    for journey in df_journey_collapsed["Journey"].unique():
        # Filter dataframe for the current journey
        df_filtered = df_journey_collapsed[df_journey_collapsed["Journey"] == journey]

        # Filter out infinite and NaN values from the current journey data
        df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=['Angle', 'Annualized_TSR_Capiq'], how='any')

        # Plot TSR vs angle for the current journey
        plt.figure(figsize=(8, 6))

        # Create scatter plot with KDE
        plt.scatter(df_filtered["Angle"], df_filtered["Annualized_TSR_Capiq"], alpha=0.5)

        # Add KDE (Kernel Density Estimation)
        kde = gaussian_kde(df_filtered[["Angle", "Annualized_TSR_Capiq"]].T, bw_method=0.3)
        xi, yi = np.mgrid[df_filtered["Angle"].min():df_filtered["Angle"].max():100j,
                 df_filtered["Annualized_TSR_Capiq"].min():df_filtered["Annualized_TSR_Capiq"].max():100j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
        plt.contour(xi, yi, zi.reshape(xi.shape), colors='k')

        plt.xlabel("Angle (degrees)")
        plt.ylabel("Annualized TSR")
        plt.title("Annualized TSR vs Angle for Journey: " + journey)
        plt.grid(True)
        plt.savefig("Distribution_joint_angles_Journey_"+journey)
        plt.show()

if make_plots:

    plt.figure(figsize=(10, 8))

    # Create a dictionary to store the color of each journey and its median
    journey_colors = {}
    journey_medians = {}

    # Loop over unique values in "Journey" column
    for journey in df_journey_collapsed["Journey"].unique():
        # Filter dataframe for the current journey
        df_filtered = df_journey_collapsed[df_journey_collapsed["Journey"] == journey]

        # Filter out infinite and NaN values from the current journey data
        df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=['Annualized_TSR_Capiq'], how='any')
        tsr_vector = df_filtered["Annualized_TSR_Capiq"] * 100

        # Create KDE plot
        kde = gaussian_kde(tsr_vector, bw_method=0.3)
        x = np.linspace(tsr_vector.min(), tsr_vector.max(), 100)
        y = kde(x)
        plt.plot(x, y, label=journey)

        # Get color of current journey plot
        line_color = plt.gca().lines[-1].get_color()
        journey_colors[journey] = line_color

        # Calculate median
        median_tsr = np.median(tsr_vector)
        journey_medians[journey] = median_tsr

    plt.xlabel("Annualized TSR %")
    plt.ylabel("Density")
    plt.title("Distribution of Annualized TSR by Journey")

    # Create legend for distribution plots
    plt.legend()

    # Add vertical lines and median values
    for journey, color in journey_colors.items():
        # Calculate median
        median_tsr = journey_medians[journey]

        # Add vertical line at the median with matching color
        plt.axvline(median_tsr, linestyle='--', color=color, alpha=0.5, label=f"{journey} - Median: {median_tsr:.2f}%")

    plt.grid(True)
    plt.legend()
    plt.savefig("Distribution_Annualized_TSR_ByJourney")
    plt.show()