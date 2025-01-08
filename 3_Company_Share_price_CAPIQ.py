import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
import matplotlib.pyplot as pyplot
import matplotlib
matplotlib.use('TkAgg')

# Company tickers
stock_names = ["MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "NFLX", "META"]
plot_title = "Technology_names"

# Plot adjusted close having rebased
fig, ax = plt.subplots()

for i in range(len(stock_names)):
    print("Iteration ", stock_names[i])
    # Slice specific company
    stock_i = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\USA_platform_data\share_price\_"+stock_names[i]+"_price.csv")
    # Get date and price
    stock_i_date = stock_i["Date"]
    stock_i_price = stock_i["Price"]
    stock_i_price_adjusted = [float(i) for i in stock_i_price]

    # Plot adjusted close
    log_returns = np.log(stock_i_price) - np.log(stock_i_price).shift(1)
    rebased = 100 * np.exp(np.nan_to_num(log_returns.cumsum()))

    # Plot date vs stock price
    plt.plot(stock_i_date, rebased, label = stock_names[i])
    plt.title(plot_title)
    company_flat = plot_title.translate(str.maketrans('', '', string.punctuation))
    plt.xlabel("Year")
    plt.ylabel("Share price rebased to 100")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
plt.legend()
plt.savefig(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\plots\Share_price_"+"_"+str(plot_title))
plt.show()