"""
Trading strategy for pairs trading.
Generates signals based on spread z-scores.
"""

import numpy as np
import pandas as pd


def compute_spread(price1, price2, hedge_ratio):
    """
    Compute the spread for a pair.
    
    spread = price1 - hedge_ratio * price2
    """
    return price1 - hedge_ratio * price2


def compute_z_score(spread, window=60):
    """
    Compute rolling z-score of spread.
    
    z = (spread - mean) / std
    
    Uses rolling window for mean and std.
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    
    z_score = (spread - rolling_mean) / rolling_std
    
    return z_score


def generate_signals(z_score, entry_threshold=2.0, exit_threshold=0.5, stop_loss=3.5):
    """
    Generate trading signals based on z-score.
    
    Signal logic:
        - Enter long (1) when z < -entry_threshold
        - Enter short (-1) when z > entry_threshold
        - Exit (0) when |z| < exit_threshold
        - Stop loss when |z| > stop_loss
    
    Returns:
        Series of positions: 1 (long spread), -1 (short spread), 0 (no position)
    """
    signals = pd.Series(0, index=z_score.index)
    position = 0  # Track current position
    
    for i in range(len(z_score)):
        z = z_score.iloc[i]
        
        if pd.isna(z):
            signals.iloc[i] = 0
            continue
        
        # Stop loss - force exit
        if abs(z) > stop_loss and position != 0:
            position = 0
            signals.iloc[i] = 0
            continue
        
        # Entry signals
        if position == 0:
            if z < -entry_threshold:
                position = 1  # Long spread
            elif z > entry_threshold:
                position = -1  # Short spread
        
        # Exit signals
        elif abs(z) < exit_threshold:
            position = 0
        
        signals.iloc[i] = position
    
    return signals


def calculate_pnl(price1, price2, hedge_ratio, signals):
    """
    Calculate P&L for a pairs trading strategy.
    
    For a long spread position (signal = 1):
        - Long stock 1 (price1)
        - Short stock 2 (hedge_ratio units)
    
    Returns:
        Series of cumulative P&L
    """
    # Compute returns
    ret1 = price1.pct_change()
    ret2 = price2.pct_change()
    
    # Spread return = ret1 - hedge_ratio * ret2
    spread_return = ret1 - hedge_ratio * ret2
    
    # Apply signals (shift by 1 to avoid look-ahead bias)
    position = signals.shift(1).fillna(0)
    
    # P&L = position * spread_return
    pnl = position * spread_return
    
    # Cumulative P&L
    cum_pnl = (1 + pnl).cumprod() - 1
    
    return cum_pnl