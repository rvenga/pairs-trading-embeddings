"""
Backtesting framework for pairs trading with cointegration monitoring.

ENHANCED VERSION: 
- Monitors cointegration during test period
- Updates hedge ratios dynamically
- Stops trading when relationships break
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from trading_strategy import compute_spread, compute_z_score, generate_signals, calculate_pnl


def backtest_pair_with_monitoring(price1, price2, hedge_ratio_initial,
                                   entry_threshold=2.0, exit_threshold=0.5, stop_loss=3.5,
                                   window=60, transaction_cost=0.001, initial_capital=1.0,
                                   enable_monitoring=True, retest_frequency=60,
                                   p_value_threshold=0.10, retest_window=120):
    """
    Backtest a single pair WITH cointegration monitoring and dynamic hedge ratio updates.
    
    This is the enhanced version that:
    1. Re-tests cointegration every retest_frequency days
    2. Updates hedge ratios when cointegration holds
    3. Stops trading when cointegration breaks (p > threshold)
    4. Returns detailed monitoring log
    
    Args:
        price1, price2: Price series (pandas Series with datetime index)
        hedge_ratio_initial: Initial hedge ratio from training
        entry_threshold: Z-score threshold to enter (default 2.0)
        exit_threshold: Z-score threshold to exit (default 0.5)
        stop_loss: Z-score threshold to force exit (default 3.5)
        window: Rolling window for z-score (default 60)
        transaction_cost: Cost per trade (default 0.001 = 0.1%)
        initial_capital: Starting capital (default $1.00)
        enable_monitoring: Whether to monitor cointegration (default True)
        retest_frequency: Days between cointegration checks (default 60)
        p_value_threshold: Stop if p-value > this (default 0.10)
        retest_window: Lookback window for retesting (default 120 days)
        
    Returns:
        dict with:
            - All standard metrics (return, sharpe, etc.)
            - monitoring_log: DataFrame of cointegration checks
            - stopped_early: Whether trading was stopped due to broken cointegration
            - active_days: Number of days actually traded
    """
    # Initialize
    monitoring_log = []
    current_hedge_ratio = hedge_ratio_initial
    is_active = True
    last_check_idx = 0
    
    # Create series to track when we're allowed to trade
    can_trade = pd.Series(True, index=price1.index)
    hedge_ratios = pd.Series(hedge_ratio_initial, index=price1.index)
    
    # Monitoring loop (if enabled)
    if enable_monitoring:
        for i in range(retest_frequency, len(price1), retest_frequency):
            if not is_active:
                # Mark all future dates as inactive
                can_trade.iloc[i:] = False
                break
            
            # Get data up to this point (lookback window)
            window_start = max(0, i - retest_window)
            price1_window = price1.iloc[window_start:i]
            price2_window = price2.iloc[window_start:i]
            
            # Align
            common_idx = price1_window.index.intersection(price2_window.index)
            price1_window = price1_window.loc[common_idx]
            price2_window = price2_window.loc[common_idx]
            
            if len(price1_window) < 100:
                continue
            
            # Retest cointegration
            try:
                score, p_value, _ = coint(price1_window, price2_window)
                new_hedge_ratio = np.polyfit(price2_window, price1_window, 1)[0]
            except:
                # If test fails, keep current hedge ratio
                p_value = 0.05
                new_hedge_ratio = current_hedge_ratio
            
            # Decision logic
            if p_value <= p_value_threshold:
                # Still cointegrated - update hedge ratio
                current_hedge_ratio = new_hedge_ratio
                action = 'CONTINUE'
                is_cointegrated = True
                
                # Update hedge ratios for this period
                hedge_ratios.iloc[last_check_idx:i] = current_hedge_ratio
                last_check_idx = i
            else:
                # Cointegration broke - stop trading
                action = 'STOP'
                is_cointegrated = False
                is_active = False
                
                # Mark all future dates as inactive
                can_trade.iloc[i:] = False
            
            # Log this check
            monitoring_log.append({
                'date': price1.index[i],
                'day_index': i,
                'p_value': p_value,
                'hedge_ratio': current_hedge_ratio,
                'is_cointegrated': is_cointegrated,
                'action': action
            })
        
        # Fill remaining period with last hedge ratio
        hedge_ratios.iloc[last_check_idx:] = current_hedge_ratio
    
    # Now run backtest with dynamic hedge ratios
    all_signals = pd.Series(0, index=price1.index)
    all_cum_pnl = pd.Series(0.0, index=price1.index)
    
    # Split into segments based on hedge ratio changes
    for date_idx in range(len(price1)):
        if not can_trade.iloc[date_idx]:
            # Not allowed to trade anymore
            all_signals.iloc[date_idx:] = 0
            break
        
        # Use current hedge ratio for this period
        hr = hedge_ratios.iloc[date_idx]
        
        # Compute spread with current hedge ratio
        spread = price1 - hr * price2
        z_score = compute_z_score(spread, window=window)
        
        # Generate signals
        signals = generate_signals(z_score, entry_threshold, exit_threshold, stop_loss)
        
        # Calculate P&L with current hedge ratio
        cum_pnl = calculate_pnl(price1, price2, hr, signals, initial_capital)
        
        # Store
        all_signals = signals
        all_cum_pnl = cum_pnl
    
    # Apply transaction costs
    position_changes = all_signals.diff().fillna(0)
    num_trades = (position_changes != 0).sum()
    
    costs = (position_changes != 0).astype(float) * transaction_cost * initial_capital
    cum_costs = costs.cumsum() / initial_capital
    cum_pnl_after_costs = all_cum_pnl - cum_costs
    
    # Calculate daily returns
    daily_returns = cum_pnl_after_costs.diff().fillna(0)
    
    # Calculate metrics
    num_days = len(cum_pnl_after_costs)
    active_days = can_trade.sum()
    total_return = cum_pnl_after_costs.iloc[-1]
    
    if num_days > 0 and total_return > -1:
        annualized_return = (1 + total_return) ** (252 / num_days) - 1
    else:
        annualized_return = 0
    
    if daily_returns.std() > 0:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0
    
    # Max drawdown
    cummax = (1 + cum_pnl_after_costs).cummax()
    drawdown = (1 + cum_pnl_after_costs) / cummax - 1
    max_drawdown = drawdown.min()
    
    # Win rate
    trade_returns, win_rate = calculate_trade_statistics(all_signals, daily_returns)
    
    # Check if stopped early
    stopped_early = not can_trade.iloc[-1]
    
    return {
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'cum_pnl': cum_pnl_after_costs,
        'signals': all_signals,
        'daily_returns': daily_returns,
        'trade_returns': trade_returns,
        'monitoring_log': pd.DataFrame(monitoring_log) if monitoring_log else None,
        'stopped_early': stopped_early,
        'active_days': active_days,
        'total_days': num_days
    }


def calculate_trade_statistics(signals, daily_returns):
    """
    Calculate statistics for individual trades.
    Properly tracks P&L for each trade separately.
    """
    trade_returns = []
    current_trade_pnl = 0.0
    in_trade = False
    
    for i in range(len(signals)):
        current_signal = signals.iloc[i]
        prev_signal = signals.iloc[i-1] if i > 0 else 0
        
        # Entering a trade
        if current_signal != 0 and prev_signal == 0:
            in_trade = True
            current_trade_pnl = 0.0
        
        # Accumulate P&L while in trade
        if in_trade:
            current_trade_pnl += daily_returns.iloc[i]
        
        # Exiting a trade
        if current_signal == 0 and prev_signal != 0:
            trade_returns.append(current_trade_pnl)
            in_trade = False
    
    # Handle case where still in trade at end
    if in_trade and current_trade_pnl != 0:
        trade_returns.append(current_trade_pnl)
    
    # Calculate win rate
    if len(trade_returns) > 0:
        win_rate = sum(1 for ret in trade_returns if ret > 0) / len(trade_returns)
    else:
        win_rate = 0.0
    
    return trade_returns, win_rate


def backtest_pair(price1, price2, hedge_ratio, 
                  entry_threshold=2.0, exit_threshold=0.5, stop_loss=3.5,
                  window=60, transaction_cost=0.001, initial_capital=1.0):
    """
    Standard backtest (without monitoring) - for backward compatibility.
    """
    return backtest_pair_with_monitoring(
        price1, price2, hedge_ratio,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        window=window,
        transaction_cost=transaction_cost,
        initial_capital=initial_capital,
        enable_monitoring=False  # Disabled for standard backtest
    )


def backtest_portfolio_with_monitoring(prices_df, pairs_df,
                                       entry_threshold=2.0, exit_threshold=0.5,
                                       stop_loss=3.5, window=60, transaction_cost=0.001,
                                       enable_monitoring=True, retest_frequency=60,
                                       p_value_threshold=0.10, verbose=True):
    """
    Backtest a portfolio of pairs WITH cointegration monitoring.
    
    Args:
        prices_df: DataFrame with [Date, Ticker, Close]
        pairs_df: DataFrame with [Ticker1, Ticker2, Hedge_Ratio]
        entry_threshold: Z-score to enter
        exit_threshold: Z-score to exit
        stop_loss: Z-score to force exit
        window: Rolling window for z-score
        transaction_cost: Cost per trade
        enable_monitoring: Enable cointegration monitoring
        retest_frequency: Days between checks
        p_value_threshold: Stop trading if p > this
        verbose: Print progress
        
    Returns:
        DataFrame with performance metrics per pair
    """
    # Pivot prices
    price_pivot = prices_df.pivot(index='Date', columns='Ticker', values='Close')
    
    results = []
    monitoring_logs = {}
    
    if verbose:
        print(f"Backtesting {len(pairs_df)} pairs with monitoring={enable_monitoring}...")
    
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
            metrics = backtest_pair_with_monitoring(
                price1, price2, hedge_ratio,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                stop_loss=stop_loss,
                window=window,
                transaction_cost=transaction_cost,
                enable_monitoring=enable_monitoring,
                retest_frequency=retest_frequency,
                p_value_threshold=p_value_threshold
            )
            
            pair_name = f"{ticker1}-{ticker2}"
            
            results.append({
                'Ticker1': ticker1,
                'Ticker2': ticker2,
                'Annualized_Return': metrics['annualized_return'],
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Max_Drawdown': metrics['max_drawdown'],
                'Num_Trades': metrics['num_trades'],
                'Win_Rate': metrics['win_rate'],
                'Stopped_Early': metrics['stopped_early'],
                'Active_Days': metrics['active_days'],
                'Total_Days': metrics['total_days'],
                'Utilization': metrics['active_days'] / metrics['total_days']
            })
            
            # Store monitoring log
            if metrics['monitoring_log'] is not None:
                monitoring_logs[pair_name] = metrics['monitoring_log']
            
            if verbose and metrics['stopped_early']:
                print(f"   ⚠️  {pair_name}: Stopped early due to broken cointegration")
                
        except Exception as e:
            if verbose:
                print(f"   Error backtesting {ticker1}-{ticker2}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    return results_df, monitoring_logs


def backtest_portfolio(prices_df, pairs_df, entry_threshold=2.0, exit_threshold=0.5,
                       stop_loss=3.5, window=60, transaction_cost=0.001):
    """
    Standard backtest (without monitoring) - for backward compatibility.
    """
    results_df, _ = backtest_portfolio_with_monitoring(
        prices_df, pairs_df,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        window=window,
        transaction_cost=transaction_cost,
        enable_monitoring=False,
        verbose=False
    )
    
    return results_df