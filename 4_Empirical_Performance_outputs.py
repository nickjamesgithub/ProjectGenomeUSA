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

# Write out csv file locally
df = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_Journeys_summary.csv")
# Get unique Genome classes
genome_class_unique = df["Genome_classification_beginning"].unique()

results_list = []
for i in range(len(genome_class_unique)):
    # Print Genome class
    print("Genome class ", genome_class_unique[i])

    ### NOTE - DATA CAPS BASED ON PREVIOUS ANALYSIS ###
    # Slice genome class
    df_genome_class_i = df.loc[(df["Genome_classification_end"]==genome_class_unique[i]) &
                               (df["EP/FE_end"] >= -.3) & (df["EP/FE_end"] <= .5) &
                               (df["Revenue_growth_end"] >= -.3) & (df["Revenue_growth_end"] <= 1.5) &
                               (df["Annualized_TSR_Capiq"] >= -.4) & (df["Annualized_TSR_Capiq"] <= 1) &
                               (df["Price_to_book"] > -200)]

    df_market_return_proxy = df.loc[(df["EP/FE_end"] >= -.3) & (df["EP/FE_end"] <= .5) &
                               (df["Revenue_growth_end"] >= -.3) & (df["Revenue_growth_end"] <= 1.5) &
                               (df["Annualized_TSR_Capiq"] >= -.4) & (df["Annualized_TSR_Capiq"] <= 1) &
                               (df["Price_to_book"] > -200)]
    # Market return proxy
    market_return_median = np.median(df_market_return_proxy["Annualized_TSR_Capiq"].values)

    # Compute annualized TSR values
    annualised_tsr_capiq_i = df_genome_class_i["Annualized_TSR_Capiq"].mean()
    # Compute standard deviation
    annualised_tsr_capiq_i_std = df_genome_class_i["Annualized_TSR_Capiq"].std()
    annualised_tsr_capiq_i_min = df_genome_class_i["Annualized_TSR_Capiq"].min()
    annualised_tsr_capiq_i_max = df_genome_class_i["Annualized_TSR_Capiq"].max()
    normalized_annualized_tsr = annualised_tsr_capiq_i - market_return_median

    # Get length
    n = len(df_genome_class_i)
    results_list.append([genome_class_unique[i], n, annualised_tsr_capiq_i, annualised_tsr_capiq_i_std, annualised_tsr_capiq_i_min,
                         annualised_tsr_capiq_i_max, normalized_annualized_tsr])

# Make results dataframe
results_df = pd.DataFrame(results_list)
results_df.columns = ["Genome_class", "N", "Annualized_TSR", "Standard_deviation_TSR", "Minimum_TSR", "Maximum_TSR", "Normalized_TSR"]

results_df.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_Genome_grid_output.csv")

X=1
Y=2