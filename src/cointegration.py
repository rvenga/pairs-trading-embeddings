"""
Cointegration testing for pairs trading.
Uses Engle-Granger two-step method.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations
from tqdm import tqdm


def split_train_test(prices_df, train_ratio=0.7):
    """
    Split price data into train and test sets chronologically.
    
    Args:
        prices_df: DataFrame with columns [Date, Ticker, Close, ...]
        train_ratio: Proportion of data for training (default 0.7)
        
    Returns:
        tuple of (train_df, test_df)
    """
    # Get unique sorted dates
    dates = prices_df['Date'].sort_values().unique()
    
    # Calculate split point
    split_idx = int(len(dates) * train_ratio)
    split_date = dates[split_idx]
    
    # Split data
    train_df = prices_df[prices_df['Date'] < split_date].copy()
    test_df = prices_df[prices_df['Date'] >= split_date].copy()
    
    print(f"Train period: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Test period: {test_df['Date'].min()} to {test_df['Date'].max()}")
    print(f"Train size: {len(train_df)} rows, Test size: {len(test_df)} rows")
    
    return train_df, test_df


def test_cointegration(price1, price2):
    """
    Test if two price series are cointegrated using Engle-Granger method.
    
    Args:
        price1: pandas Series of prices for stock 1
        price2: pandas Series of prices for stock 2
        
    Returns:
        dict with keys:
            - cointegrated: bool, whether pair is cointegrated (p < 0.05)
            - p_value: float, p-value from ADF test
            - hedge_ratio: float, beta coefficient from regression
            - spread: pandas Series, the cointegrated spread
    """
    # Run cointegration test
    score, p_value, _ = coint(price1, price2)
    
    # Estimate hedge ratio via OLS
    # price1 = alpha + beta * price2 + epsilon
    hedge_ratio = np.polyfit(price2, price1, 1)[0]
    
    # Compute spread
    spread = price1 - hedge_ratio * price2
    
    return {
        'cointegrated': p_value < 0.05,
        'p_value': p_value,
        'hedge_ratio': hedge_ratio,
        'spread': spread
    }


def find_cointegrated_pairs(prices_df, tickers=None, max_pairs=None):
    """
    Test all possible pairs for cointegration.
    
    Args:
        prices_df: DataFrame with columns [Date, Ticker, Close]
        tickers: List of tickers to test (None = all)
        max_pairs: Maximum pairs to return (None = all cointegrated)
        
    Returns:
        DataFrame with columns [Ticker1, Ticker2, P_Value, Hedge_Ratio]
        Sorted by p-value (most cointegrated first)
    """
    # Pivot to wide format
    price_pivot = prices_df.pivot(index='Date', columns='Ticker', values='Close')
    
    if tickers is None:
        tickers = price_pivot.columns.tolist()
    
    # Test all combinations
    results = []
    
    print(f"Testing {len(list(combinations(tickers, 2)))} pairs for cointegration...")
    
    for ticker1, ticker2 in tqdm(list(combinations(tickers, 2))):
        # Get price series (drop NaNs)
        series1 = price_pivot[ticker1].dropna()
        series2 = price_pivot[ticker2].dropna()
        
        # Align dates
        common_dates = series1.index.intersection(series2.index)
        if len(common_dates) < 100:  # Need enough data
            continue
            
        series1 = series1.loc[common_dates]
        series2 = series2.loc[common_dates]
        
        # Test cointegration
        result = test_cointegration(series1, series2)
        
        if result['cointegrated']:
            results.append({
                'Ticker1': ticker1,
                'Ticker2': ticker2,
                'P_Value': result['p_value'],
                'Hedge_Ratio': result['hedge_ratio']
            })
    
    # Create DataFrame and sort
    pairs_df = pd.DataFrame(results)
    
    if len(pairs_df) == 0:
        print("No cointegrated pairs found!")
        return pairs_df
    
    pairs_df = pairs_df.sort_values('P_Value')
    
    if max_pairs is not None:
        pairs_df = pairs_df.head(max_pairs)
    
    print(f"\nFound {len(pairs_df)} cointegrated pairs (p < 0.05)")
    
    return pairs_df


def compute_spread_stats(spread):
    """
    Compute statistics for a spread series.
    
    Returns:
        dict with mean, std, half-life
    """
    # Mean and std
    mean = spread.mean()
    std = spread.std()
    
    # Estimate half-life using AR(1)
    # Δs_t = φ * s_{t-1} + ε
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    
    # Drop NaNs
    valid = ~(spread_lag.isna() | spread_diff.isna())
    spread_lag_clean = spread_lag[valid]
    spread_diff_clean = spread_diff[valid]
    
    # OLS: Δs_t ~ s_{t-1}
    phi = np.polyfit(spread_lag_clean, spread_diff_clean, 1)[0]
    
    # Half-life = -log(2) / log(1 + phi)
    if phi < 0:
        half_life = -np.log(2) / np.log(1 + phi)
    else:
        half_life = np.inf
    
    return {
        'mean': mean,
        'std': std,
        'half_life': half_life
    }