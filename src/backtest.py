"""
Backtesting framework for pairs trading.
"""

import numpy as np
import pandas as pd
from trading_strategy import (
    compute_spread, 
    compute_z_score, 
    generate_signals, 
    calculate_pnl_with_costs
)


def backtest_pair(price1, price2, hedge_ratio, 
                  entry_threshold=2.0, exit_threshold=0.5, stop_loss=3.5,
                  window=60, transaction_cost=0.001, initial_capital=1.0):
    """
    Backtest a single pair with corrected P&L calculation.
    
    Args:
        price1, price2: Price series (pandas Series with datetime index)
        hedge_ratio: Beta coefficient from cointegration
        entry_threshold: Z-score threshold to enter position (default 2.0)
        exit_threshold: Z-score threshold to exit position (default 0.5)
        stop_loss: Z-score threshold to force exit (default 3.5)
        window: Rolling window for z-score calculation (default 60)
        transaction_cost: Cost per trade as fraction (default 0.001 = 0.1%)
        initial_capital: Starting capital per trade (default $1.00)
        
    Returns:
        dict with performance metrics:
            - annualized_return: Annualized return
            - sharpe_ratio: Annualized Sharpe ratio
            - max_drawdown: Maximum drawdown
            - num_trades: Number of trades
            - win_rate: Fraction of winning trades
            - cum_pnl: Series of cumulative P&L
            - signals: Series of trading signals
    """
    # Compute spread and z-score
    spread = compute_spread(price1, price2, hedge_ratio)
    z_score = compute_z_score(spread, window=window)
    
    # Generate signals (now includes stop_loss)
    signals = generate_signals(z_score, entry_threshold, exit_threshold, stop_loss)
    
    # Calculate P&L with transaction costs
    cum_pnl_after_costs = calculate_pnl_with_costs(
        price1, price2, hedge_ratio, signals, 
        initial_capital, transaction_cost
    )
    
    # Calculate daily returns
    daily_returns = cum_pnl_after_costs.diff().fillna(0)
    
    # Count trades (position changes)
    position_changes = signals.diff().fillna(0)
    num_trades = (position_changes != 0).sum()
    
    # Calculate metrics
    num_days = len(cum_pnl_after_costs)
    total_return = cum_pnl_after_costs.iloc[-1]
    
    # Annualized return using the correct formula
    if num_days > 0 and total_return > -1:  # Avoid log of negative number
        annualized_return = (1 + total_return) ** (252 / num_days) - 1
    else:
        annualized_return = 0
    
    # Sharpe ratio (annualized)
    if daily_returns.std() > 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0
    
    # Max drawdown
    cummax = (1 + cum_pnl_after_costs).cummax()
    drawdown = (1 + cum_pnl_after_costs) / cummax - 1
    max_drawdown = drawdown.min()
    
    # FIXED: Win rate calculation - track individual trade returns properly
    trade_returns, win_rate = calculate_trade_statistics(signals, daily_returns)
    
    return {
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'cum_pnl': cum_pnl_after_costs,
        'signals': signals,
        'daily_returns': daily_returns,
        'trade_returns': trade_returns  # Individual trade returns for analysis
    }


def calculate_trade_statistics(signals, daily_returns):
    """
    Calculate statistics for individual trades.
    
    FIX #5: Properly track P&L for each individual trade.
    
    Args:
        signals: Series of trading signals (1, -1, or 0)
        daily_returns: Series of daily P&L changes
        
    Returns:
        tuple of (list of trade_returns, win_rate)
    """
    trade_returns = []
    current_trade_pnl = 0.0
    in_trade = False
    
    for i in range(len(signals)):
        current_signal = signals.iloc[i]
        prev_signal = signals.iloc[i-1] if i > 0 else 0
        
        # Check if entering a trade
        if current_signal != 0 and prev_signal == 0:
            in_trade = True
            current_trade_pnl = 0.0
        
        # Accumulate P&L while in trade
        if in_trade:
            current_trade_pnl += daily_returns.iloc[i]
        
        # Check if exiting a trade
        if current_signal == 0 and prev_signal != 0:
            trade_returns.append(current_trade_pnl)
            in_trade = False
            current_trade_pnl = 0.0
    
    # Handle case where we're still in a trade at the end
    if in_trade and len(trade_returns) == 0:
        # If we never exited, count the accumulated P&L
        trade_returns.append(current_trade_pnl)
    
    # Calculate win rate
    if len(trade_returns) > 0:
        win_rate = sum(1 for ret in trade_returns if ret > 0) / len(trade_returns)
    else:
        win_rate = 0.0
    
    return trade_returns, win_rate


def backtest_portfolio(prices_df, pairs_df, entry_threshold=2.0, exit_threshold=0.5,
                       stop_loss=3.5, window=60, transaction_cost=0.001):
    """
    Backtest a portfolio of pairs.
    
    Args:
        prices_df: DataFrame with [Date, Ticker, Close]
        pairs_df: DataFrame with [Ticker1, Ticker2, Hedge_Ratio]
        entry_threshold: Z-score to enter
        exit_threshold: Z-score to exit
        stop_loss: Z-score to force exit
        window: Rolling window for z-score
        transaction_cost: Cost per trade (0.001 = 0.1%)
        
    Returns:
        DataFrame with performance metrics per pair
    """
    # Pivot prices
    price_pivot = prices_df.pivot(index='Date', columns='Ticker', values='Close')
    
    results = []
    
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
        try:
            metrics = backtest_pair(
                price1, price2, hedge_ratio,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                stop_loss=stop_loss,
                window=window,
                transaction_cost=transaction_cost
            )
            
            results.append({
                'Ticker1': ticker1,
                'Ticker2': ticker2,
                'Annualized_Return': metrics['annualized_return'],
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Max_Drawdown': metrics['max_drawdown'],
                'Num_Trades': metrics['num_trades'],
                'Win_Rate': metrics['win_rate']
            })
        except Exception:
            continue
    
    return pd.DataFrame(results)