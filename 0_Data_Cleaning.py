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
australia_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Australia.csv", encoding = 'cp1252')
belgium_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Belgium.csv", encoding = 'cp1252')
canada_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Canada.csv", encoding = 'cp1252')
chile_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Chile.csv", encoding = 'cp1252')
china_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_China.csv", encoding = 'cp1252')
denmark_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Denmark.csv", encoding = 'cp1252')
france_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_France.csv", encoding = 'cp1252')
germany_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Germany.csv", encoding = 'cp1252')
hong_kong_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Hong_Kong.csv", encoding = 'cp1252')
india_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_India.csv", encoding = 'cp1252')
italy_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Italy.csv", encoding = 'cp1252')
japan_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Japan.csv", encoding = 'cp1252')
luxembourg_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Luxembourg.csv", encoding = 'cp1252')
malaysia_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Malaysia.csv", encoding = 'cp1252')
netherlands_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Netherlands.csv", encoding = 'cp1252')
philippines_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Philippines.csv", encoding = 'cp1252')
saudi_arabia_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Saudi_Arabia.csv", encoding = 'cp1252')
singapore_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Singapore.csv", encoding = 'cp1252')
south_korea_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_South_Korea.csv", encoding = 'cp1252')
sweden_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Sweden.csv", encoding = 'cp1252')
switzerland_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Switzerland.csv", encoding = 'cp1252')
thailand_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_Thailand.csv", encoding = 'cp1252')
uae_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_UAE.csv", encoding = 'cp1252')
uk_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_United_Kingdom.csv", encoding = 'cp1252')
usa_mapping_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25\Company_list_GPT_USA.csv", encoding = 'cp1252')

# Get ticker list
australia_tickers_ = australia_mapping_data["Ticker"].unique()
belgium_tickers_ = belgium_mapping_data["Ticker"].unique()
canada_tickers_ = canada_mapping_data["Ticker"].unique()
chile_tickers_ = chile_mapping_data["Ticker"].unique()
china_tickers_ = china_mapping_data["Ticker"].unique()
denmark_tickers_ = denmark_mapping_data["Ticker"].unique()
france_tickers_ = france_mapping_data["Ticker"].unique()
germany_tickers_ = germany_mapping_data["Ticker"].unique()
hong_kong_tickers_ = hong_kong_mapping_data["Ticker"].unique()
india_tickers_ = india_mapping_data["Ticker"].unique()
italy_tickers_ = italy_mapping_data["Ticker"].unique()
japan_tickers_ = japan_mapping_data["Ticker"].unique()
luxembourg_tickers_ = luxembourg_mapping_data["Ticker"].unique()
malaysia_tickers_ = malaysia_mapping_data["Ticker"].unique()
netherlands_tickers_ = netherlands_mapping_data["Ticker"].unique()
philippines_tickers_ = philippines_mapping_data["Ticker"].unique()
saudi_arabia_tickers_ = saudi_arabia_mapping_data["Ticker"].unique()
singapore_tickers_ = singapore_mapping_data["Ticker"].unique()
south_korea_tickers_ = south_korea_mapping_data["Ticker"].unique()
sweden_tickers_ = sweden_mapping_data["Ticker"].unique()
switzerland_tickers_ = switzerland_mapping_data["Ticker"].unique()
thailand_tickers_ = thailand_mapping_data["Ticker"].unique()
uae_tickers_ = uae_mapping_data["Ticker"].unique()
uk_tickers_ = uk_mapping_data["Ticker"].unique()
usa_tickers_ = usa_mapping_data["Ticker"].unique()

base_dir = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_platform_data_25"
country_dirs = ["AUSTRALIA", "BELGIUM", "CANADA", "CHILE", "CHINA", "DENMARK", "FRANCE", "GERMANY", "HONG KONG", "INDIA", "ITALY", "JAPAN", "LUXEMBOURG", "MALAYSIA",
                "NETHERLANDS", "PHILIPPINES", "SAUDI ARABIA", "SINGAPORE", "SOUTH KOREA", "SWEDEN", "SWITZERLAND", "THAILAND", "UAE", "UNITED KINGDOM", "USA"]

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
australia_df = df_country_creator("AUSTRALIA", australia_tickers_)
belgium_df = df_country_creator("BELGIUM", belgium_tickers_)
canada_df = df_country_creator("CANADA", canada_tickers_)
chile_df = df_country_creator("CHILE", chile_tickers_)
china_df = df_country_creator("CHINA", china_tickers_)
denmark_df = df_country_creator("DENMARK", denmark_tickers_)
france_df = df_country_creator("FRANCE", france_tickers_)
germany_df = df_country_creator("GERMANY", germany_tickers_)
hong_kong_df = df_country_creator("HONG KONG", hong_kong_tickers_)
india_df = df_country_creator("INDIA", india_tickers_)
italy_df = df_country_creator("ITALY", italy_tickers_)
japan_df = df_country_creator("JAPAN", japan_tickers_)
luxembourg_df = df_country_creator("LUXEMBOURG", luxembourg_tickers_)
malaysia_df = df_country_creator("MALAYSIA", malaysia_tickers_)
netherlands_df = df_country_creator("NETHERLANDS", netherlands_tickers_)
philippines_df = df_country_creator("PHILIPPINES", philippines_tickers_)
saudi_arabia_df = df_country_creator("SAUDI ARABIA", saudi_arabia_tickers_)
singapore_df = df_country_creator("SINGAPORE", singapore_tickers_)
south_korea_df = df_country_creator("SOUTH KOREA", south_korea_tickers_)
sweden_df = df_country_creator("SWEDEN", sweden_tickers_)
switzerland_df = df_country_creator("SWITZERLAND", switzerland_tickers_)
thailand_df = df_country_creator("THAILAND", thailand_tickers_)
uae_df = df_country_creator("UAE", uae_tickers_)
uk_df = df_country_creator("UNITED KINGDOM", uk_tickers_)
usa_df = df_country_creator("USA", usa_tickers_)

# Genome encodings
df_merge_australia = generate_bespoke_genome_classification_df(australia_df)
df_merge_belgium = generate_bespoke_genome_classification_df(belgium_df)
df_merge_canada = generate_bespoke_genome_classification_df(canada_df)
df_merge_chile = generate_bespoke_genome_classification_df(chile_df)
df_merge_china = generate_bespoke_genome_classification_df(china_df)
df_merge_denmark = generate_bespoke_genome_classification_df(denmark_df)
df_merge_france = generate_bespoke_genome_classification_df(france_df)
df_merge_germany = generate_bespoke_genome_classification_df(germany_df)
df_merge_hong_kong = generate_bespoke_genome_classification_df(hong_kong_df)
df_merge_india = generate_bespoke_genome_classification_df(india_df)
df_merge_italy = generate_bespoke_genome_classification_df(italy_df)
df_merge_japan = generate_bespoke_genome_classification_df(japan_df)
df_merge_luxembourg = generate_bespoke_genome_classification_df(luxembourg_df)
df_merge_malaysia = generate_bespoke_genome_classification_df(malaysia_df)
df_merge_netherlands = generate_bespoke_genome_classification_df(netherlands_df)
df_merge_philippines = generate_bespoke_genome_classification_df(philippines_df)
df_merge_saudi_arabia = generate_bespoke_genome_classification_df(saudi_arabia_df)
df_merge_singapore = generate_bespoke_genome_classification_df(singapore_df)
df_merge_south_korea = generate_bespoke_genome_classification_df(south_korea_df)
df_merge_sweden = generate_bespoke_genome_classification_df(sweden_df)
df_merge_switzerland = generate_bespoke_genome_classification_df(switzerland_df)
df_merge_thailand = generate_bespoke_genome_classification_df(thailand_df)
df_merge_uae = generate_bespoke_genome_classification_df(uae_df)
df_merge_uk = generate_bespoke_genome_classification_df(uk_df)
df_merge_usa = generate_bespoke_genome_classification_df(usa_df)

# All country data merged
df_merge_global = pd.concat([
    df_merge_australia, df_merge_belgium, df_merge_canada, df_merge_chile,
    df_merge_china, df_merge_denmark, df_merge_france, df_merge_germany,
    df_merge_hong_kong, df_merge_india, df_merge_italy, df_merge_japan,
    df_merge_luxembourg, df_merge_malaysia, df_merge_netherlands,
    df_merge_philippines, df_merge_saudi_arabia, df_merge_singapore,
    df_merge_south_korea, df_merge_sweden, df_merge_switzerland, df_merge_thailand, df_merge_uae,
    df_merge_uk, df_merge_usa
], axis=0)

# Export merged global data
df_merge_global.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data_25\Global_data_25.csv", index=False)
