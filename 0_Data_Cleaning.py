import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import os
import pandas as pd

# Define a dictionary to map specific sectors to the y-axis metric, with others defaulting to "EP/FE"
sector_metric_mapping = {
    "Banking": "CROTE_TE",
    "Investment and Wealth": "ROE_above_Cost_of_equity",
    "Insurance": "ROE_above_Cost_of_equity",
    "Financials - other": "ROE_above_Cost_of_equity",
    # Other sectors will use "EP/FE" by default
}

# Function to process genome classification
def generate_bespoke_genome_classification_df(df):
    for sector, metric in sector_metric_mapping.items():
        if metric not in df.columns:
            raise ValueError(f"Missing required metric '{metric}' in DataFrame for sector '{sector}'.")

    classified_dfs = []
    for sector in df["Sector"].unique():
        metric = sector_metric_mapping.get(sector, "EVA_ratio_bespoke")
        if metric not in df.columns:
            continue

        sector_df = df[df["Sector"] == sector].copy()
        conditions_genome = [
            (sector_df["EVA_ratio_bespoke"] < 0) & (sector_df["Revenue_growth_3_f"] < 0),
            (sector_df["EVA_ratio_bespoke"] < 0) & (sector_df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
            (sector_df["EVA_ratio_bespoke"] < 0) & (sector_df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
            (sector_df["EVA_ratio_bespoke"] < 0) & (sector_df["Revenue_growth_3_f"] >= 0.20),
            (sector_df["EVA_ratio_bespoke"] > 0) & (sector_df["Revenue_growth_3_f"] < 0),
            (sector_df["EVA_ratio_bespoke"] > 0) & (sector_df["Revenue_growth_3_f"].between(0, 0.10, inclusive='right')),
            (sector_df["EVA_ratio_bespoke"] > 0) & (sector_df["Revenue_growth_3_f"].between(0.10, 0.20, inclusive='right')),
            (sector_df["EVA_ratio_bespoke"] > 0) & (sector_df["Revenue_growth_3_f"] >= 0.20)
        ]

        values_genome = ["UNTENABLE", "TRAPPED", "BRAVE", "FEARLESS", "CHALLENGED", "VIRTUOUS", "FAMOUS", "LEGENDARY"]
        sector_df["Genome_classification_bespoke"] = np.select(conditions_genome, values_genome, default="UNKNOWN")
        classified_dfs.append(sector_df)

    return pd.concat(classified_dfs)

# Data Preparation
# Import data
usa_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents\Company_list_GPT_SP500.csv", encoding = 'cp1252')
aus_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents\Company_list_GPT_AUS.csv", encoding = 'cp1252')
india_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents\Company_list_GPT_Nifty.csv", encoding = 'cp1252')
japan_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents\Company_list_GPT_Nikkei.csv", encoding = 'cp1252')
europe_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents\Company_list_GPT_Euro.csv", encoding = 'cp1252')
uk_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents\Company_list_GPT_FTSE.csv", encoding = 'cp1252')

usa_tickers_ = usa_mapping_data["Ticker"].unique()
aus_tickers_ = aus_mapping_data["Ticker"].unique()
india_tickers_ = india_mapping_data["Ticker"].unique()
japan_tickers_ = japan_mapping_data["Ticker"].unique()
europe_tickers_ = europe_mapping_data["Ticker"].unique()
uk_tickers_ = uk_mapping_data["Ticker"].unique()

base_dir = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data"
country_dirs = ["USA", "AUS", "INDIA", "JAPAN", "EURO", "UK"]

def df_country_creator(country, tickers):
    dfs_list = []
    for ticker in tickers:
        print(str(country) + " " + str(ticker))
        try:
            file_path = os.path.join(base_dir, country, f"_{ticker}.csv")
            df = pd.read_csv(file_path, encoding='cp1252')
            df["Country"] = country  # Add a new column for the country
            dfs_list.append(df)
        except Exception as e:
            print(f"Error with {ticker} in {country}: {e}")
    return pd.concat(dfs_list, ignore_index=True) if dfs_list else pd.DataFrame()

# Example usage for each country
usa_df = df_country_creator("USA", usa_tickers_)
aus_df = df_country_creator("AUS", aus_tickers_)
india_df = df_country_creator("INDIA", india_tickers_)
japan_df = df_country_creator("JAPAN", japan_tickers_)
euro_df = df_country_creator("EURO", europe_tickers_)
uk_df = df_country_creator("UK", uk_tickers_)

# Genome encodings
df_merge_usa = generate_bespoke_genome_classification_df(usa_df)
df_merge_aus = generate_bespoke_genome_classification_df(aus_df)
df_merge_india = generate_bespoke_genome_classification_df(india_df)
df_merge_japan = generate_bespoke_genome_classification_df(japan_df)
df_merge_euro = generate_bespoke_genome_classification_df(euro_df)
df_merge_uk = generate_bespoke_genome_classification_df(uk_df)

# All country data merged
df_merge_global = pd.concat([df_merge_usa, df_merge_aus, df_merge_india, df_merge_japan, df_merge_euro, df_merge_uk], axis=0)
df_merge_global.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\Global_data.csv")
