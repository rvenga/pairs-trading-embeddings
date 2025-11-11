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
    Compute expanding z-score of spread.
    
    z = (spread - mean) / std
    
    Uses expanding window for mean and std to prevent lookahead bias.
    Only uses past data (no future information).
    """
    expanding_mean = spread.expanding(min_periods=window).mean()
    expanding_std = spread.expanding(min_periods=window).std()
    
    z_score = (spread - expanding_mean) / expanding_std
    
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
            # FIX #7: Reset position when we hit NaN to maintain state consistency
            position = 0
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


def calculate_pnl(price1, price2, hedge_ratio, signals, initial_capital=1.0):
    """
    Calculate P&L for a pairs trading strategy with proper dollar-neutral mechanics.
    
    For a dollar-neutral pairs trade:
    - Total capital = initial_capital (e.g., $1.00)
    - When signal = 1 (long spread):
        * Long stock1 with capital = initial_capital / (1 + hedge_ratio)
        * Short stock2 with capital = (initial_capital * hedge_ratio) / (1 + hedge_ratio)
    - When signal = -1 (short spread):
        * Short stock1 with capital = initial_capital / (1 + hedge_ratio)
        * Long stock2 with capital = (initial_capital * hedge_ratio) / (1 + hedge_ratio)
    
    Args:
        price1: Price series for stock 1
        price2: Price series for stock 2
        hedge_ratio: Hedge ratio (beta from cointegration)
        signals: Trading signals (1 = long spread, -1 = short spread, 0 = no position)
        initial_capital: Starting capital per trade (default $1.00)
        
    Returns:
        Series of cumulative P&L (as a fraction of initial capital)
    """
    # Compute returns for each stock
    ret1 = price1.pct_change()
    ret2 = price2.pct_change()
    
    # Apply signals (shift by 1 to avoid look-ahead bias)
    # Position at time t is based on signal at time t-1
    position = signals.shift(1).fillna(0)
    
    # Calculate position sizes for dollar neutrality
    # For long spread (position = 1):
    #   - weight1 = 1 / (1 + hedge_ratio)
    #   - weight2 = -hedge_ratio / (1 + hedge_ratio)
    # For short spread (position = -1): flip the signs
    
    weight1 = position / (1 + hedge_ratio)
    weight2 = -position * hedge_ratio / (1 + hedge_ratio)
    
    # Calculate P&L for each leg
    pnl1 = weight1 * ret1 * initial_capital
    pnl2 = weight2 * ret2 * initial_capital
    
    # Total P&L = sum of both legs
    daily_pnl = (pnl1 + pnl2).fillna(0)
    
    # Cumulative P&L (expressed as cumulative return on initial capital)
    # This is the cumulative sum of daily P&Ls divided by initial capital
    cum_pnl = daily_pnl.cumsum() / initial_capital
    
    return cum_pnl


def calculate_pnl_with_costs(price1, price2, hedge_ratio, signals, 
                              initial_capital=1.0, transaction_cost=0.001):
    """
    Calculate P&L with transaction costs properly applied.
    
    Transaction costs are applied as a percentage of the traded value
    whenever a position changes.
    
    Args:
        price1, price2: Price series
        hedge_ratio: Hedge ratio
        signals: Trading signals
        initial_capital: Starting capital per trade
        transaction_cost: Cost per trade as a fraction (0.001 = 0.1%)
        
    Returns:
        Series of cumulative P&L (as a fraction of initial capital)
    """
    # Get base P&L without costs
    cum_pnl = calculate_pnl(price1, price2, hedge_ratio, signals, initial_capital)
    
    # Calculate position changes (when we trade)
    position_changes = signals.diff().fillna(0)
    
    # Calculate transaction costs
    # Cost = transaction_cost * (|position1| + |position2|) * initial_capital
    # Since position1 + position2 is dollar neutral, total position size = initial_capital
    # So cost per trade = transaction_cost * initial_capital
    
    costs = (position_changes != 0).astype(float) * transaction_cost * initial_capital
    
    # Cumulative costs as a fraction of initial capital
    cum_costs = costs.cumsum() / initial_capital
    
    # P&L after costs
    cum_pnl_after_costs = cum_pnl - cum_costs
    
    return cum_pnl_after_costs
