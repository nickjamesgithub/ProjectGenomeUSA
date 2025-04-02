import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('TkAgg')

# --------------------------------------------
# 1. Sector-Metric Mapping
# --------------------------------------------
sector_metric_mapping = {
    "Banking": "CROTE_TE",
    "Investment and Wealth": "ROE_above_Cost_of_equity",
    "Insurance": "ROE_above_Cost_of_equity",
    "Financials - other": "ROE_above_Cost_of_equity",
}

# Global list to capture file-level errors
error_files = []

# --------------------------------------------
# 2. Genome Classification Function
# --------------------------------------------
def generate_bespoke_genome_classification_df(df):
    for sector, metric in sector_metric_mapping.items():
        if sector in df["Sector"].unique() and metric not in df.columns:
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

    return pd.concat(classified_dfs) if classified_dfs else df

# --------------------------------------------
# 3. Country-Level File Loader
# --------------------------------------------
def df_country_creator(country, tickers):
    dfs_list = []
    for ticker in tickers:
        print(f"{country} {ticker}")
        try:
            file_path = os.path.join(base_dir, country, f"_{ticker}.csv")
            df = pd.read_csv(file_path, encoding='cp1252')
            df["Country"] = country
            dfs_list.append(df)
        except Exception as e:
            print(f"Error with {ticker} in {country}: {e}")
            error_files.append({
                "Country": country,
                "Ticker": ticker,
                "File_Path": file_path,
                "Error_Message": str(e)
            })
    return pd.concat(dfs_list, ignore_index=True) if dfs_list else pd.DataFrame()

# --------------------------------------------
# 4. Setup Paths
# --------------------------------------------
base_mapping_dir = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_market_constituents_25"
base_dir = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\Global_platform_data_25"

# --------------------------------------------
# 5. Load All Ticker Mapping Files (CHINA FIX HERE)
# --------------------------------------------
def load_mapping(file_name, force_str_ticker=False):
    if force_str_ticker:
        return pd.read_csv(os.path.join(base_mapping_dir, file_name), encoding='cp1252', dtype={'Ticker': str})
    else:
        return pd.read_csv(os.path.join(base_mapping_dir, file_name), encoding='cp1252')

australia_mapping_data = load_mapping("Company_list_GPT_Australia.csv")
belgium_mapping_data = load_mapping("Company_list_GPT_Belgium.csv")
canada_mapping_data = load_mapping("Company_list_GPT_Canada.csv")
chile_mapping_data = load_mapping("Company_list_GPT_Chile.csv")
china_mapping_data = load_mapping("Company_list_GPT_China.csv", force_str_ticker=True)  # 👈 FIX APPLIED HERE
denmark_mapping_data = load_mapping("Company_list_GPT_Denmark.csv")
france_mapping_data = load_mapping("Company_list_GPT_France.csv")
germany_mapping_data = load_mapping("Company_list_GPT_Germany.csv")
hong_kong_mapping_data = load_mapping("Company_list_GPT_Hong_Kong.csv")
india_mapping_data = load_mapping("Company_list_GPT_India.csv")
italy_mapping_data = load_mapping("Company_list_GPT_Italy.csv")
japan_mapping_data = load_mapping("Company_list_GPT_Japan.csv")
luxembourg_mapping_data = load_mapping("Company_list_GPT_Luxembourg.csv")
malaysia_mapping_data = load_mapping("Company_list_GPT_Malaysia.csv")
netherlands_mapping_data = load_mapping("Company_list_GPT_Netherlands.csv")
philippines_mapping_data = load_mapping("Company_list_GPT_Philippines.csv")
saudi_arabia_mapping_data = load_mapping("Company_list_GPT_Saudi_Arabia.csv")
singapore_mapping_data = load_mapping("Company_list_GPT_Singapore.csv")
south_korea_mapping_data = load_mapping("Company_list_GPT_South_Korea.csv")
sweden_mapping_data = load_mapping("Company_list_GPT_Sweden.csv")
switzerland_mapping_data = load_mapping("Company_list_GPT_Switzerland.csv")
thailand_mapping_data = load_mapping("Company_list_GPT_Thailand.csv")
uae_mapping_data = load_mapping("Company_list_GPT_UAE.csv")
uk_mapping_data = load_mapping("Company_list_GPT_United_Kingdom.csv")
usa_mapping_data = load_mapping("Company_list_GPT_USA.csv")

# --------------------------------------------
# 6. Extract Tickers
# --------------------------------------------
australia_tickers = australia_mapping_data["Ticker"].unique()
belgium_tickers = belgium_mapping_data["Ticker"].unique()
canada_tickers = canada_mapping_data["Ticker"].unique()
chile_tickers = chile_mapping_data["Ticker"].unique()
china_tickers = china_mapping_data["Ticker"].unique()
denmark_tickers = denmark_mapping_data["Ticker"].unique()
france_tickers = france_mapping_data["Ticker"].unique()
germany_tickers = germany_mapping_data["Ticker"].unique()
hong_kong_tickers = hong_kong_mapping_data["Ticker"].unique()
india_tickers = india_mapping_data["Ticker"].unique()
italy_tickers = italy_mapping_data["Ticker"].unique()
japan_tickers = japan_mapping_data["Ticker"].unique()
luxembourg_tickers = luxembourg_mapping_data["Ticker"].unique()
malaysia_tickers = malaysia_mapping_data["Ticker"].unique()
netherlands_tickers = netherlands_mapping_data["Ticker"].unique()
philippines_tickers = philippines_mapping_data["Ticker"].unique()
saudi_arabia_tickers = saudi_arabia_mapping_data["Ticker"].unique()
singapore_tickers = singapore_mapping_data["Ticker"].unique()
south_korea_tickers = south_korea_mapping_data["Ticker"].unique()
sweden_tickers = sweden_mapping_data["Ticker"].unique()
switzerland_tickers = switzerland_mapping_data["Ticker"].unique()
thailand_tickers = thailand_mapping_data["Ticker"].unique()
uae_tickers = uae_mapping_data["Ticker"].unique()
uk_tickers = uk_mapping_data["Ticker"].unique()
usa_tickers = usa_mapping_data["Ticker"].unique()

# --------------------------------------------
# 7. Load All DataFrames
# --------------------------------------------
df_merge_global = pd.concat([
    generate_bespoke_genome_classification_df(df_country_creator("AUSTRALIA", australia_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("BELGIUM", belgium_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("CANADA", canada_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("CHILE", chile_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("CHINA", china_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("DENMARK", denmark_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("FRANCE", france_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("GERMANY", germany_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("HONG KONG", hong_kong_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("INDIA", india_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("ITALY", italy_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("JAPAN", japan_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("LUXEMBOURG", luxembourg_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("MALAYSIA", malaysia_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("NETHERLANDS", netherlands_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("PHILIPPINES", philippines_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("SAUDI ARABIA", saudi_arabia_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("SINGAPORE", singapore_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("SOUTH KOREA", south_korea_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("SWEDEN", sweden_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("SWITZERLAND", switzerland_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("THAILAND", thailand_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("UAE", uae_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("UNITED KINGDOM", uk_tickers)),
    generate_bespoke_genome_classification_df(df_country_creator("USA", usa_tickers))
], axis=0)

# --------------------------------------------
# 8. Optional Data Cleanup
# --------------------------------------------
df_merge_global["Sector"] = df_merge_global["Sector"].replace("Consumer staples", "Consumer Staples")

if "Ticker_full" in df_merge_global.columns:
    df_merge_global["Ticker"] = df_merge_global["Ticker_full"].str.split(":").str[-1]

# --------------------------------------------
# 9. Export Global Data
# --------------------------------------------
output_path = os.path.join(base_dir, "Global_data_25.csv")
df_merge_global.to_csv(output_path, index=False)
print(f"\n✅ Global data saved to: {output_path}")

# --------------------------------------------
# 10. Export Error Log if Needed
# --------------------------------------------
if error_files:
    error_df = pd.DataFrame(error_files)
    error_log_path = os.path.join(base_dir, "Error_log.csv")
    error_df.to_csv(error_log_path, index=False)
    print(f"⚠️ Error log saved to: {error_log_path}")
else:
    print("✅ No errors during file loading.")
