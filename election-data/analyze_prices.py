import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Define election dates
election_dates = {
    2000: "2000-11-07",
    2004: "2004-11-02",
    2008: "2008-11-04",
    2012: "2012-11-06",
    2016: "2016-11-08",
    2020: "2020-11-03"
}

# Function to calculate relative dates
def get_relative_dates(election_date):
    """
    Calculate dates relative to the election date.
    
    Parameters:
        election_date (str): Election date in 'YYYY-MM-DD' format.
        
    Returns:
        dict: Dictionary containing relative dates as pd.Timestamp objects.
    """
    date = pd.to_datetime(election_date)
    return {
        "2_months_before": date - pd.Timedelta(days=60),
        "1_month_before": date - pd.Timedelta(days=30),
        "1_month_after": date + pd.Timedelta(days=30),
        "2_months_after": date + pd.Timedelta(days=60)
    }

# Function to find the closest trading day on or before the given date
def get_closest_trading_day(date, data):
    """
    Find the closest available trading day on or before the given date.
    
    Parameters:
        date (pd.Timestamp): The target date.
        data (pd.DataFrame): The stock data with DatetimeIndex.
        
    Returns:
        pd.Timestamp: The closest trading day.
        
    Raises:
        ValueError: If no trading days are found on or before the given date.
    """
    available_dates = data.index[data.index <= date]
    if available_dates.empty:
        raise ValueError(f"No trading days found on or before {date.date()}")
    return available_dates[-1]

# Download historical data
start_date = "2000-01-01"
end_date = "2021-01-02"  # Adjusted to include 2021-01-01 if it's a trading day
vix_data = yf.download("^VIX", start=start_date, end=end_date)
spy_data = yf.download("SPY", start=start_date, end=end_date)

# Initialize results dictionary
results = {year: {} for year in election_dates.keys()}

# Iterate over each election year and calculate relative prices
for year, election_date in election_dates.items():
    relative_dates = get_relative_dates(election_date)
    
    for period, date in relative_dates.items():
        # Retrieve VIX Close price
        try:
            vix_closest_day = get_closest_trading_day(date, vix_data)
            vix_close = vix_data.loc[vix_closest_day, 'Close']
            results[year][f"VIX_{period}"] = vix_close
        except (ValueError, KeyError) as e:
            print(f"Warning: VIX data unavailable for {period} ({date.date()}) in {year}. Setting as None.")
            results[year][f"VIX_{period}"] = None  # Placeholder for missing data
        
        # Retrieve SPY Close price
        try:
            spy_closest_day = get_closest_trading_day(date, spy_data)
            spy_close = spy_data.loc[spy_closest_day, 'Close']
            results[year][f"SPY_{period}"] = spy_close
        except (ValueError, KeyError) as e:
            print(f"Warning: SPY data unavailable for {period} ({date.date()}) in {year}. Setting as None.")
            results[year][f"SPY_{period}"] = None  # Placeholder for missing data

# Convert results to DataFrame
df_results = pd.DataFrame.from_dict(results, orient='index')

# Optional: Sort the DataFrame columns for better readability
cols = sorted(df_results.columns, key=lambda x: (x.split('_')[1], x.split('_')[0]))
df_results = df_results[cols]

# Display results
print(df_results)

# Optionally, save results to CSV
df_results.to_csv("election_price_movements.csv")
