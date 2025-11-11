"""
Cointegration testing for pairs trading with monitoring capabilities.
Uses Engle-Granger two-step method.

ENHANCED VERSION: Adds periodic re-testing and dynamic hedge ratio updates.
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


def test_cointegration_window(price1, price2, window_size=120, min_periods=100):
    """
    Test cointegration on a rolling window of data.
    
    This is used for periodic re-testing during backtesting.
    
    Args:
        price1, price2: Price series
        window_size: Size of lookback window (default 120 days)
        min_periods: Minimum data required (default 100 days)
        
    Returns:
        dict with cointegration results, or None if insufficient data
    """
    # Use most recent window_size observations
    if len(price1) < min_periods or len(price2) < min_periods:
        return None
    
    # Get trailing window
    recent_price1 = price1.iloc[-window_size:] if len(price1) >= window_size else price1
    recent_price2 = price2.iloc[-window_size:] if len(price2) >= window_size else price2
    
    # Align
    common_idx = recent_price1.index.intersection(recent_price2.index)
    recent_price1 = recent_price1.loc[common_idx]
    recent_price2 = recent_price2.loc[common_idx]
    
    if len(recent_price1) < min_periods:
        return None
    
    # Test cointegration
    return test_cointegration(recent_price1, recent_price2)


def monitor_cointegration(price1, price2, hedge_ratio_initial,
                          retest_frequency=60,
                          p_value_threshold=0.10,
                          retest_window=120):
    """
    Monitor cointegration relationship over time with periodic re-testing.
    
    This function:
    1. Re-tests cointegration every retest_frequency days
    2. Updates hedge ratio when cointegration holds
    3. Flags when cointegration breaks
    
    Args:
        price1, price2: Full price series
        hedge_ratio_initial: Initial hedge ratio from training
        retest_frequency: Days between re-tests (default 60)
        p_value_threshold: Stop trading if p > this (default 0.10)
        retest_window: Lookback window for re-testing (default 120 days)
        
    Returns:
        DataFrame with columns:
            - date: Date of check
            - p_value: Cointegration p-value
            - hedge_ratio: Current hedge ratio
            - is_cointegrated: Whether pair is still cointegrated
            - action: 'CONTINUE' or 'STOP'
    """
    monitoring_log = []
    
    # Initial state
    current_hedge_ratio = hedge_ratio_initial
    is_active = True
    
    # Retest at regular intervals
    for i in range(retest_frequency, len(price1), retest_frequency):
        if not is_active:
            # Already stopped trading
            break
        
        # Get data up to this point
        price1_window = price1.iloc[:i]
        price2_window = price2.iloc[:i]
        
        # Retest cointegration
        result = test_cointegration_window(
            price1_window, 
            price2_window, 
            window_size=retest_window
        )
        
        if result is None:
            # Insufficient data
            continue
        
        p_value = result['p_value']
        new_hedge_ratio = result['hedge_ratio']
        
        # Check if cointegration still holds
        if p_value <= p_value_threshold:
            # Still cointegrated - update hedge ratio
            current_hedge_ratio = new_hedge_ratio
            action = 'CONTINUE'
            is_cointegrated = True
        else:
            # Cointegration broke - stop trading
            action = 'STOP'
            is_cointegrated = False
            is_active = False
        
        # Log this check
        monitoring_log.append({
            'date': price1.index[i],
            'day_index': i,
            'p_value': p_value,
            'hedge_ratio': current_hedge_ratio,
            'is_cointegrated': is_cointegrated,
            'action': action
        })
    
    return pd.DataFrame(monitoring_log)


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