"""
Data collection for pairs trading project.
Fetches price data and company info from Yahoo Finance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
from tqdm import tqdm

from config import DATA_DIR, START_DATE, END_DATE, TEST_TICKERS



def fetch_price_data(tickers, start_date, end_date):
    """Fetch historical price data."""
    print(f"Fetching price data for {len(tickers)} tickers...")
    
    all_data = []
    failed = []
    
    for ticker in tqdm(tickers):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                failed.append(ticker)
                continue
            
            df = df.reset_index()
            df['Ticker'] = ticker
            all_data.append(df)
            time.sleep(0.1)  # Be nice to Yahoo
            
        except Exception as e:
            print(f"Failed {ticker}: {e}")
            failed.append(ticker)
    
    if failed:
        print(f"\nFailed tickers: {failed}")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined.columns = ['Date', 'Open', 'High', 'Low', 'Close', 
                        'Volume', 'Dividends', 'Stock Splits', 'Ticker']
    combined = combined[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return combined


def fetch_company_info(tickers):
    """Fetch company metadata."""
    print(f"\nFetching company info for {len(tickers)} tickers...")
    
    company_data = []
    
    for ticker in tqdm(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
        
            
            company_data.append({
                'Ticker': ticker,
                'Name': info.get('longName', ''),
                'Sector': info.get('sector', ''),
                'Industry': info.get('industry', ''),
                'Description': info.get('longBusinessSummary', ''),
                'Market_Cap': info.get('marketCap', None),
                'Country': info.get('country', '')
            })
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Failed info for {ticker}: {e}")
    
    return pd.DataFrame(company_data)


def compute_returns(price_df):
    """Compute daily log returns."""
    print("\nComputing returns...")
    
    price_df = price_df.sort_values(['Ticker', 'Date'])
    price_df['Return'] = price_df.groupby('Ticker')['Close'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    
    return price_df[['Date', 'Ticker', 'Return']].dropna()


def main():
    """Run data collection pipeline."""
    
    print("=" * 60)
    print("PAIRS TRADING - DATA COLLECTION")
    print("=" * 60)
    
    # Fetch prices
    prices = fetch_price_data(TEST_TICKERS, START_DATE, END_DATE)
    prices.to_csv(DATA_DIR / "prices.csv", index=False)
    print(f"\n✓ Saved prices: {len(prices)} rows, {prices['Ticker'].nunique()} tickers")
    
    # Fetch company info
    companies = fetch_company_info(TEST_TICKERS)
    companies.to_csv(DATA_DIR / "company_info.csv", index=False)
    print(f"✓ Saved company info: {len(companies)} companies")
    
    # Compute returns
    returns = compute_returns(prices)
    returns.to_csv(DATA_DIR / "returns.csv", index=False)
    print(f"✓ Saved returns: {len(returns)} rows")
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nDate range: {prices['Date'].min()} to {prices['Date'].max()}")
    print(f"Tickers: {prices['Ticker'].nunique()}")
    print(f"\nFiles in: {DATA_DIR}")


if __name__ == "__main__":
    main()