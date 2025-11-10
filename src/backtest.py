"""
Backtesting framework for pairs trading.
"""

import numpy as np
import pandas as pd
from trading_strategy import compute_spread, compute_z_score, generate_signals, calculate_pnl


def backtest_pair(price1, price2, hedge_ratio, 
                  entry_threshold=2.0, exit_threshold=0.5, 
                  window=60, transaction_cost=0.001):
    """
    Backtest a single pair.
    
    Args:
        price1, price2: Price series
        hedge_ratio: Beta coefficient
        entry_threshold: Z-score to enter
        exit_threshold: Z-score to exit
        window: Rolling window for z-score
        transaction_cost: Cost per trade (0.001 = 0.1%)
        
    Returns:
        dict with performance metrics
    """
    # Compute spread and z-score
    spread = compute_spread(price1, price2, hedge_ratio)
    z_score = compute_z_score(spread, window=window)
    
    # Generate signals
    signals = generate_signals(z_score, entry_threshold, exit_threshold)
    
    # Calculate P&L
    cum_pnl = calculate_pnl(price1, price2, hedge_ratio, signals)
    
    # Count trades (position changes)
    position_changes = signals.diff().fillna(0)
    num_trades = (position_changes != 0).sum()
    
    # Apply transaction costs
    trade_costs = (position_changes != 0) * transaction_cost
    cum_pnl_after_costs = cum_pnl - trade_costs.cumsum()
    
    # Calculate metrics
    total_return = cum_pnl_after_costs.iloc[-1]
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    daily_returns = cum_pnl_after_costs.diff().fillna(0)
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    
    # Max drawdown
    cummax = (1 + cum_pnl_after_costs).cummax()
    drawdown = (1 + cum_pnl_after_costs) / cummax - 1
    max_drawdown = drawdown.min()
    
    # Win rate
    trade_returns = []
    entry_pnl = None
    for i in range(len(signals)):
        if signals.iloc[i] != 0 and (i == 0 or signals.iloc[i-1] == 0):
            # Entering position
            entry_pnl = cum_pnl_after_costs.iloc[i]
        elif signals.iloc[i] == 0 and (i > 0 and signals.iloc[i-1] != 0):
            # Exiting position
            if entry_pnl is not None:
                trade_return = cum_pnl_after_costs.iloc[i] - entry_pnl
                trade_returns.append(trade_return)
    
    win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'cum_pnl': cum_pnl_after_costs,
        'signals': signals
    }


def backtest_portfolio(prices_df, pairs_df):
    """
    Backtest a portfolio of pairs.
    
    Args:
        prices_df: DataFrame with [Date, Ticker, Close]
        pairs_df: DataFrame with [Ticker1, Ticker2, Hedge_Ratio]
        
    Returns:
        DataFrame with performance metrics per pair
    """
    # Pivot prices
    price_pivot = prices_df.pivot(index='Date', columns='Ticker', values='Close')
    
    results = []
    
    print(f"Backtesting {len(pairs_df)} pairs...")
    
    for idx, row in pairs_df.iterrows():
        ticker1 = row['Ticker1']
        ticker2 = row['Ticker2']
        hedge_ratio = row['Hedge_Ratio']
        
        # Get aligned prices
        price1 = price_pivot[ticker1].dropna()
        price2 = price_pivot[ticker2].dropna()
        common_dates = price1.index.intersection(price2.index)
        
        if len(common_dates) < 100:
            continue
        
        price1 = price1.loc[common_dates]
        price2 = price2.loc[common_dates]
        
        # Backtest
        metrics = backtest_pair(price1, price2, hedge_ratio)
        
        results.append({
            'Ticker1': ticker1,
            'Ticker2': ticker2,
            'Total_Return': metrics['total_return'],
            'Sharpe_Ratio': metrics['sharpe_ratio'],
            'Max_Drawdown': metrics['max_drawdown'],
            'Num_Trades': metrics['num_trades'],
            'Win_Rate': metrics['win_rate']
        })
    
    return pd.DataFrame(results)