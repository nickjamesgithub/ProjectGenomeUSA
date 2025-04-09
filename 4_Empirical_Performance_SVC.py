import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
make_plots = True
base_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\WBC\Capital_markets_review"
main_data_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\global_banking_data.csv"

# === PARAMETERS ===
start_year = 2015
end_year = 2024
flat_inflation_rate = 0.027  # 2.7%
n_years = end_year - start_year + 1

# === LOAD DATA ===
df_full = pd.read_csv(main_data_path)
df_full['Year'] = df_full['Year'].astype(int)
df = df_full[(df_full['Year'] >= start_year) & (df_full['Year'] <= end_year)].copy()
df = df.drop_duplicates(subset=['Company_name', 'Year'])

# === APPLY SVC CRITERIA ===
df['Growth_threshold_met'] = df['Revenue_growth_1_f'] > flat_inflation_rate
df['EVA_threshold_met'] = df['EVA_ratio_bespoke'] >= 0
df['SVC_Criteria_Met'] = df['Growth_threshold_met'] & df['EVA_threshold_met']

# === COUNT SVC YEARS MET ===
criteria_count = df.groupby('Company_name')['SVC_Criteria_Met'].sum()
svc_summary_dict = {i: [] for i in range(n_years + 1)}

for company, count in criteria_count.items():
    svc_summary_dict[count].append(company)

# === BUILD SVC SUMMARY TABLE ===
svc_summary = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in svc_summary_dict.items()])).fillna('')
svc_summary.columns = [f"SVC_{i}" for i in range(len(svc_summary.columns))]
svc_summary.to_csv(os.path.join(base_path, "svc_summary_global_banks_peers.csv"), index=False)

# === PERFORMANCE SUMMARY: GROWTH AND EVA YEARS MET ===
performance_summary = df.groupby('Company_name').agg(
    Growth_threshold=('Growth_threshold_met', 'sum'),
    EVA_threshold=('EVA_threshold_met', 'sum')
).reset_index()
performance_summary.to_csv(os.path.join(base_path, "svc_threshold_summary_global_banks_peers.csv"), index=False)

# === MEDIAN VALUATION AND TSR BY SVC GROUP ===
pbv_medians, tsr_medians = {}, {}
pbv_counts, tsr_counts = {}, {}

for column in svc_summary.columns:
    companies = svc_summary[column].dropna().tolist()
    filtered_df = df[df["Company_name"].isin(companies)]

    pbv_medians[column] = filtered_df["PBV"].median()
    pbv_counts[column] = filtered_df["PBV"].notna().sum()

    tsr_medians[column] = filtered_df["TSR_CIQ_no_buybacks"].median()
    tsr_counts[column] = filtered_df["TSR_CIQ_no_buybacks"].notna().sum()

# === CREATE SUMMARY DATAFRAMES ===
pbv_summary = pd.DataFrame({
    'SVC_Category': pbv_medians.keys(),
    'Median_PBV': pbv_medians.values(),
    'N': pbv_counts.values()
})
tsr_summary = pd.DataFrame({
    'SVC_Category': tsr_medians.keys(),
    'Median_TSR_CIQ_No_Buybacks': tsr_medians.values(),
    'N': tsr_counts.values()
})

pbv_summary.to_csv(os.path.join(base_path, "pbv_summary_global_banks.csv"), index=False)
tsr_summary.to_csv(os.path.join(base_path, "tsr_no_buybacks_summary_global_banks.csv"), index=False)

# === LONG-TERM TSR CALCULATION (2015 to 2024) ===
df_2015 = df_full[df_full["Year"] == 2015][["Company_name", "Adjusted_Stock_Price"]]
df_2024 = df_full[df_full["Year"] == 2024][["Company_name", "Adjusted_Stock_Price"]]
df_tsr = df_2015.merge(df_2024, on='Company_name', suffixes=('_2015', '_2024'))
df_tsr['Annualized_TSR'] = (df_tsr['Adjusted_Stock_Price_2024'] / df_tsr['Adjusted_Stock_Price_2015']) ** (1 / 9) - 1

# === SVC CONDITIONS PER COMPANY ===
df_condition_summary = df.groupby('Company_name').agg({
    'Growth_threshold_met': 'sum',
    'EVA_threshold_met': 'sum',
    'SVC_Criteria_Met': 'sum'
}).reset_index()
df_condition_summary = df_condition_summary.merge(df_tsr[['Company_name', 'Annualized_TSR']], on='Company_name', how='left')

df_condition_summary.rename(columns={
    'Growth_threshold_met': 'Years_Growth_Above_Inflation',
    'EVA_threshold_met': 'Years_EVA_Positive',
    'SVC_Criteria_Met': 'Years_SVC_Met'
}, inplace=True)

df_condition_summary.to_csv(os.path.join(base_path, "company_condition_summary_global_banks.csv"), index=False)

# === OPTIONAL PLOTS ===
if make_plots:
    plt.plot(pbv_summary["SVC_Category"], pbv_summary["Median_PBV"])
    plt.ylabel("Median P:BV")
    plt.xlabel("SVC Years Met")
    plt.title("SVC Criteria vs P:BV")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.plot(tsr_summary["SVC_Category"], tsr_summary["Median_TSR_CIQ_No_Buybacks"])
    plt.ylabel("Median TSR (CIQ No Buybacks)")
    plt.xlabel("SVC Years Met")
    plt.title("SVC Criteria vs TSR")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
