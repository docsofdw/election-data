import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define election dates
ELECTION_DATES = {
    2000: "2000-11-07",
    2004: "2004-11-02",
    2008: "2008-11-04",
    2012: "2012-11-06",
    2016: "2016-11-08",
    2020: "2020-11-03"
}

# Define relative periods in months
RELATIVE_PERIODS = {
    '2_months_before': -2,
    '1_month_before': -1,
    '1_month_after': 1,
    '2_months_after': 2
}

def get_closest_trading_day(target_date, data):
    """
    Find the closest available trading day on or before the given date.
    
    Parameters:
        target_date (pd.Timestamp): The target date.
        data (pd.DataFrame): The stock data with DatetimeIndex.
        
    Returns:
        pd.Timestamp: The closest trading day.
        
    Raises:
        ValueError: If no trading days are found on or before the given date.
    """
    available_dates = data.index[data.index <= target_date]
    if available_dates.empty:
        raise ValueError(f"No trading days found on or before {target_date.date()}")
    return available_dates[-1]

def get_relative_dates(election_date_str):
    """
    Calculate dates relative to the election date.
    
    Parameters:
        election_date_str (str): Election date in 'YYYY-MM-DD' format.
        
    Returns:
        dict: Dictionary containing relative dates as pd.Timestamp objects.
    """
    election_date = pd.to_datetime(election_date_str)
    relative_dates = {}
    for period, months in RELATIVE_PERIODS.items():
        relative_date = election_date + pd.DateOffset(months=months)
        relative_dates[period] = relative_date
    return relative_dates

def download_data(symbol, start, end):
    """
    Download historical data for a given symbol.
    
    Parameters:
        symbol (str): Stock ticker symbol.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: Downloaded stock data.
    """
    logging.info(f"Downloading data for {symbol} from {start} to {end}")
    data = yf.download(symbol, start=start, end=end, progress=False)
    if data.empty:
        logging.warning(f"No data downloaded for {symbol}")
    return data

def calculate_price_movements(election_dates, periods, vix_data, spy_data):
    """
    Calculate VIX and SPY prices at specified relative dates around each election.
    
    Parameters:
        election_dates (dict): Dictionary of election years and dates.
        periods (dict): Dictionary of relative periods in months.
        vix_data (pd.DataFrame): VIX historical data.
        spy_data (pd.DataFrame): SPY historical data.
        
    Returns:
        pd.DataFrame: DataFrame containing the price movements.
    """
    results = {year: {} for year in election_dates.keys()}
    
    for year, date_str in election_dates.items():
        logging.info(f"Processing election year: {year} on {date_str}")
        relative_dates = get_relative_dates(date_str)
        
        for period, rel_date in relative_dates.items():
            try:
                # Get closest trading day for VIX
                vix_trading_day = get_closest_trading_day(rel_date, vix_data)
                vix_price = vix_data.loc[vix_trading_day, 'Close']
                results[year][f'VIX_{period}'] = vix_price
            except (ValueError, KeyError) as e:
                logging.warning(f"VIX data unavailable for {period} ({rel_date.date()}) in {year}. Setting as NaN.")
                results[year][f'VIX_{period}'] = np.nan
            
            try:
                # Get closest trading day for SPY
                spy_trading_day = get_closest_trading_day(rel_date, spy_data)
                spy_price = spy_data.loc[spy_trading_day, 'Close']
                results[year][f'SPY_{period}'] = spy_price
            except (ValueError, KeyError) as e:
                logging.warning(f"SPY data unavailable for {period} ({rel_date.date()}) in {year}. Setting as NaN.")
                results[year][f'SPY_{period}'] = np.nan
                
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'Year'
    df.reset_index(inplace=True)
    return df

def create_visualizations(df):
    """
    Create and save visualizations based on the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing price movements.
    """
    logging.info("Creating visualizations")
    
    # Line Plot for VIX and SPY Trends
    plt.figure(figsize=(14, 7))
    for asset in ['VIX', 'SPY']:
        for period in ['1_month_before', '1_month_after']:
            plt.plot(df['Year'], df[f'{asset}_{period}'], marker='o', label=f'{asset} {period.replace("_", " ")}')
    plt.title('VIX and SPY Trends Around Elections')
    plt.xlabel('Election Year')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('election_trends.png')
    plt.close()
    logging.info("Saved plot: election_trends.png")
    
    # Bar Chart for VIX Comparison
    plt.figure(figsize=(14, 7))
    x = np.arange(len(df['Year']))
    width = 0.35
    plt.bar(x - width/2, df['VIX_2_months_before'], width, label='VIX 2 months before', color='skyblue')
    plt.bar(x + width/2, df['VIX_2_months_after'], width, label='VIX 2 months after', color='salmon')
    plt.title('VIX Before and After Elections')
    plt.xlabel('Election Year')
    plt.ylabel('VIX Price')
    plt.xticks(x, df['Year'])
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('vix_comparison.png')
    plt.close()
    logging.info("Saved plot: vix_comparison.png")
    
    # Correlation Heatmap
    correlation = df.drop(columns=['Year']).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Price Movements')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    logging.info("Saved plot: correlation_heatmap.png")

def calculate_percentage_changes(df):
    """
    Calculate percentage changes between 'after' and 'before' periods.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing price movements.
        
    Returns:
        pd.DataFrame: Updated DataFrame with percentage changes.
    """
    df['VIX_1m_change (%)'] = ((df['VIX_1_month_after'] - df['VIX_1_month_before']) / df['VIX_1_month_before']) * 100
    df['SPY_1m_change (%)'] = ((df['SPY_1_month_after'] - df['SPY_1_month_before']) / df['SPY_1_month_before']) * 100
    return df

def main():
    # Define date range
    START_DATE = "2000-01-01"
    END_DATE = "2021-02-01"  # Extended to ensure coverage
    
    # Download data
    vix_data = download_data("^VIX", START_DATE, END_DATE)
    spy_data = download_data("SPY", START_DATE, END_DATE)
    
    # Check if data is downloaded successfully
    if vix_data.empty or spy_data.empty:
        logging.error("One or more datasets failed to download. Exiting script.")
        return
    
    # Calculate price movements
    df_movements = calculate_price_movements(ELECTION_DATES, RELATIVE_PERIODS, vix_data, spy_data)
    
    # Save initial results to CSV
    df_movements.to_csv('election_price_movements.csv', index=False)
    logging.info("Data saved to election_price_movements.csv")
    
    # Create visualizations
    create_visualizations(df_movements)
    
    # Calculate percentage changes
    df_movements = calculate_percentage_changes(df_movements)
    
    # Display percentage changes
    logging.info("\nPercentage Changes:")
    print(df_movements[['Year', 'VIX_1m_change (%)', 'SPY_1m_change (%)']])
    
    # Save updated DataFrame with percentage changes
    df_movements.to_csv('election_price_movements_with_changes.csv', index=False)
    logging.info("Updated data saved to election_price_movements_with_changes.csv")

if __name__ == "__main__":
    main()
