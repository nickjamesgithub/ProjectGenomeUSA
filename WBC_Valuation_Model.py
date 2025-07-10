import pandas as pd
import numpy as np

# ------------------ Setup ------------------ #
start_year = 2015
end_year = 2024
n_years = end_year - start_year
tsr_method = "capital_iq"

selected_countries = [
    "Australia", "Denmark", "Hong_Kong", "Italy", "Malaysia", "Netherlands",
    "Singapore", "Sweden", "Switzerland", "Thailand", "USA", "United_Kingdom"
]

# ------------------ Load Data ------------------ #
global_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv"
bespoke_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\bespoke_data.csv"

df_global = pd.read_csv(global_path)
df_bespoke = pd.read_csv(bespoke_path)

# Add source flag
df_global["Source"] = "Global"
df_bespoke["Source"] = "Bespoke"

# Combine datasets
df_all = pd.concat([df_global, df_bespoke], ignore_index=True)

# ------------------ Filter to Banking Sector and Selected Countries ------------------ #
df_all = df_all[
    (df_all["Sector"] == "Banking") &
    (df_all["Country"].isin(selected_countries)) &
    (df_all["Year"].between(start_year, end_year))
].copy()

# ------------------ Filter Companies with Full Year Coverage ------------------ #
complete_tickers = (
    df_all.groupby("Ticker")["Year"]
    .nunique()
    .reset_index()
    .query(f"Year == {n_years + 1}")["Ticker"]
    .tolist()
)
df_all = df_all[df_all["Ticker"].isin(complete_tickers)].copy()

# ------------------ Compute Derived Fields ------------------ #
df_all["BVE_per_Share"] = df_all["Book_Value_Equity"] / df_all["Shares_outstanding"]
df_all = df_all.sort_values(["Company_name", "Year"])

# ------------------ Decomposition Metrics ------------------ #
decomp_list = []

for company, group in df_all.groupby("Company_name"):
    group = group.sort_values("Year")

    try:
        price_start = group.loc[group["Year"] == start_year, "Adjusted_Stock_Price"].values[0]
        price_end = group.loc[group["Year"] == end_year, "Adjusted_Stock_Price"].values[0]
        pbv_start = group.loc[group["Year"] == start_year, "PBV"].values[0]
        pbv_end = group.loc[group["Year"] == end_year, "PBV"].values[0]
        bveps_start = group.loc[group["Year"] == start_year, "Book_Value_Equity"].values[0] / group.loc[group["Year"] == start_year, "Shares_outstanding"].values[0]
        bveps_end = group.loc[group["Year"] == end_year, "Book_Value_Equity"].values[0] / group.loc[group["Year"] == end_year, "Shares_outstanding"].values[0]
        cash_return = group["Dividend_Buyback_Yield"].mean()

        # Basic checks
        if price_start == 0 or pd.isnull(price_start) or pd.isnull(price_end):
            continue

        ann_tsr = ((price_end / price_start) ** (1 / n_years) - 1) * 100
        pbv_cagr = ((pbv_end / pbv_start) ** (1 / n_years) - 1) * 100
        bveps_cagr = ((bveps_end / bveps_start) ** (1 / n_years) - 1) * 100
        cash_return_pct = cash_return * 100

        decomp_list.append({
            "Company_name": group["Company_name"].iloc[0],
            "Country": group["Country"].iloc[0],
            "Source": group["Source"].iloc[0],
            "Annualized_TSR_%": ann_tsr,
            "PBV_CAGR_%": pbv_cagr,
            "BVEps_CAGR_%": bveps_cagr,
            "Cash_Return_%": cash_return_pct
        })

    except Exception:
        continue

decomp_df = pd.DataFrame(decomp_list)

# ------------------ Compute EVA & Growth Metrics ------------------ #
metrics_list = []

for company, group in df_all.groupby("Company_name"):
    try:
        avg_eva = group["EVA_ratio_bespoke"].mean()
        avg_growth = group["Revenue_growth_3_f"].mean()
        eva_positive_years = (group["EVA_ratio_bespoke"] >= 0).sum()
        growth_positive_years = (group["Revenue_growth_3_f"] >= 0.03).sum()

        metrics_list.append({
            "Company_name": company,
            "Avg_EVA_ratio_bespoke": avg_eva,
            "Avg_Revenue_growth_3_f": avg_growth,
            "Years_EVA_Positive": eva_positive_years,
            "Years_growth_Positive": growth_positive_years
        })
    except Exception:
        continue

metrics_df = pd.DataFrame(metrics_list)

# ------------------ Merge TSR Components with Metrics ------------------ #
final_df = pd.merge(decomp_df, metrics_df, on="Company_name", how="left")

# ------------------ Add Average Row ------------------ #
average_row = {
    "Company_name": "Average",
    "Country": "",
    "Source": "",
    "Annualized_TSR_%": final_df["Annualized_TSR_%"].mean(),
    "PBV_CAGR_%": final_df["PBV_CAGR_%"].mean(),
    "BVEps_CAGR_%": final_df["BVEps_CAGR_%"].mean(),
    "Cash_Return_%": final_df["Cash_Return_%"].mean(),
    "Avg_EVA_ratio_bespoke": final_df["Avg_EVA_ratio_bespoke"].mean(),
    "Avg_Revenue_growth_3_f": final_df["Avg_Revenue_growth_3_f"].mean(),
    "Years_EVA_Positive": final_df["Years_EVA_Positive"].mean(),
    "Years_growth_Positive": final_df["Years_growth_Positive"].mean()
}

final_df = pd.concat([final_df, pd.DataFrame([average_row])], ignore_index=True)

# ------------------ Display Final Output ------------------ #
print(final_df.tail(21))  # Shows last 20 + average
