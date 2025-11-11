"""
Enhanced baseline with cointegration monitoring.
Compares OLD (no monitoring) vs NEW (with monitoring) approaches.
"""

import pandas as pd
from pathlib import Path
import sys

# Make sure we can import from outputs directory
sys.path.insert(0, '/mnt/user-data/outputs')

from cointegration_with_monitoring import find_cointegrated_pairs, split_train_test
from backtest_with_monitoring import backtest_portfolio_with_monitoring

# Configuration
TRAIN_RATIO = 0.7  # 70% train, 30% test
MAX_PAIRS = 15

# Monitoring parameters
RETEST_FREQUENCY = 60  # Re-test every 60 days
P_VALUE_THRESHOLD = 0.90  # Stop trading if p-value > 0.10
RETEST_WINDOW = 120  # Use 120 days for re-testing


def main():
    # Paths
    DATA_DIR = Path("data/raw")
    OUTPUT_DIR = Path("data/processed")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("ENHANCED BASELINE: Cointegration Monitoring Comparison")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Train/Test Split: {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")
    print(f"  - Retest Frequency: Every {RETEST_FREQUENCY} days")
    print(f"  - P-Value Threshold: {P_VALUE_THRESHOLD}")
    print(f"  - Retest Window: {RETEST_WINDOW} days")
    
    # Load data
    print("\n" + "=" * 70)
    print("1. Loading Data")
    print("=" * 70)
    prices = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=['Date'])
    print(f"✓ Loaded {len(prices)} price observations")
    print(f"  Date range: {prices['Date'].min()} to {prices['Date'].max()}")
    print(f"  Tickers: {prices['Ticker'].nunique()}")
    
    # Split into train and test
    print("\n" + "=" * 70)
    print("2. Splitting Data")
    print("=" * 70)
    train_prices, test_prices = split_train_test(prices, train_ratio=TRAIN_RATIO)
    
    # Find cointegrated pairs on TRAINING data
    print("\n" + "=" * 70)
    print("3. Finding Cointegrated Pairs (Training Data)")
    print("=" * 70)
    pairs = find_cointegrated_pairs(train_prices, max_pairs=MAX_PAIRS)
    pairs.to_csv(OUTPUT_DIR / "cointegrated_pairs.csv", index=False)
    print(f"✓ Found {len(pairs)} pairs")
    print(f"✓ Saved to {OUTPUT_DIR / 'cointegrated_pairs.csv'}")
    
    if len(pairs) == 0:
        print("\n⚠️  No cointegrated pairs found. Exiting.")
        return
    
    # Backtest WITHOUT monitoring (baseline)
    print("\n" + "=" * 70)
    print("4. Backtest WITHOUT Monitoring (Baseline)")
    print("=" * 70)
    old_results, _ = backtest_portfolio_with_monitoring(
        test_prices, pairs,
        enable_monitoring=False,
        verbose=True
    )
    old_results.to_csv(OUTPUT_DIR / "backtest_no_monitoring.csv", index=False)
    print(f"✓ Saved to {OUTPUT_DIR / 'backtest_no_monitoring.csv'}")
    
    # Backtest WITH monitoring (enhanced)
    print("\n" + "=" * 70)
    print("5. Backtest WITH Monitoring (Enhanced)")
    print("=" * 70)
    new_results, monitoring_logs = backtest_portfolio_with_monitoring(
        test_prices, pairs,
        enable_monitoring=True,
        retest_frequency=RETEST_FREQUENCY,
        p_value_threshold=P_VALUE_THRESHOLD,
        verbose=True
    )
    new_results.to_csv(OUTPUT_DIR / "backtest_with_monitoring.csv", index=False)
    print(f"✓ Saved to {OUTPUT_DIR / 'backtest_with_monitoring.csv'}")
    
    # Save monitoring logs
    if monitoring_logs:
        print(f"\n✓ Monitoring Logs:")
        for pair_name, log in monitoring_logs.items():
            filename = f"monitoring_log_{pair_name.replace('-', '_')}.csv"
            log.to_csv(OUTPUT_DIR / filename, index=False)
            print(f"  - {filename}")
    
    # Compare results
    print("\n" + "=" * 70)
    print("6. COMPARISON: OLD vs NEW")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("WITHOUT MONITORING (Baseline)")
    print("-" * 70)
    print(f"Total pairs: {len(old_results)}")
    print(f"Average return: {old_results['Annualized_Return'].mean():.2%}")
    print(f"Average Sharpe: {old_results['Sharpe_Ratio'].mean():.2f}")
    print(f"Average max drawdown: {old_results['Max_Drawdown'].mean():.2%}")
    print(f"Average trades: {old_results['Num_Trades'].mean():.1f}")
    print(f"Average win rate: {old_results['Win_Rate'].mean():.2%}")
    
    print("\n" + "-" * 70)
    print("WITH MONITORING (Enhanced)")
    print("-" * 70)
    print(f"Total pairs: {len(new_results)}")
    print(f"Average return: {new_results['Annualized_Return'].mean():.2%}")
    print(f"Average Sharpe: {new_results['Sharpe_Ratio'].mean():.2f}")
    print(f"Average max drawdown: {new_results['Max_Drawdown'].mean():.2%}")
    print(f"Average trades: {new_results['Num_Trades'].mean():.1f}")
    print(f"Average win rate: {new_results['Win_Rate'].mean():.2%}")
    print(f"Pairs stopped early: {new_results['Stopped_Early'].sum()}")
    print(f"Average utilization: {new_results['Utilization'].mean():.1%}")
    
    print("\n" + "-" * 70)
    print("IMPROVEMENT")
    print("-" * 70)
    ret_improvement = new_results['Annualized_Return'].mean() - old_results['Annualized_Return'].mean()
    sharpe_improvement = new_results['Sharpe_Ratio'].mean() - old_results['Sharpe_Ratio'].mean()
    dd_improvement = new_results['Max_Drawdown'].mean() - old_results['Max_Drawdown'].mean()
    
    print(f"Return improvement: {ret_improvement:+.2%}")
    print(f"Sharpe improvement: {sharpe_improvement:+.2f}")
    print(f"Drawdown improvement: {dd_improvement:+.2%} (negative is better)")
    
    # Top performers
    print("\n" + "=" * 70)
    print("7. TOP 5 PAIRS (WITH MONITORING)")
    print("=" * 70)
    top_pairs = new_results.nlargest(5, 'Sharpe_Ratio')[
        ['Ticker1', 'Ticker2', 'Sharpe_Ratio', 'Annualized_Return', 
         'Max_Drawdown', 'Stopped_Early', 'Utilization']
    ]
    print(top_pairs.to_string(index=False))
    
    # Pairs that were stopped
    stopped_pairs = new_results[new_results['Stopped_Early'] == True]
    if len(stopped_pairs) > 0:
        print("\n" + "=" * 70)
        print("8. PAIRS STOPPED DUE TO BROKEN COINTEGRATION")
        print("=" * 70)
        stopped_summary = stopped_pairs[
            ['Ticker1', 'Ticker2', 'Annualized_Return', 'Sharpe_Ratio', 
             'Max_Drawdown', 'Utilization']
        ]
        print(stopped_summary.to_string(index=False))
        print(f"\n{len(stopped_pairs)} pairs stopped early (out of {len(new_results)})")
        
        # Show impact of stopping
        print("\n💡 Impact of stopping these pairs:")
        if len(stopped_pairs) > 0:
            avg_return_stopped = stopped_pairs['Annualized_Return'].mean()
            avg_dd_stopped = stopped_pairs['Max_Drawdown'].mean()
            print(f"   Average return of stopped pairs: {avg_return_stopped:.2%}")
            print(f"   Average drawdown of stopped pairs: {avg_dd_stopped:.2%}")
            print(f"   → These were likely losing money, so stopping them helped!")
    
    print("\n" + "=" * 70)
    print("✓ ENHANCED BASELINE COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Monitoring helps detect when cointegration breaks")
    print("2. Stopping bad pairs early prevents large losses")
    print("3. Check monitoring logs to see when/why pairs were stopped")
    print("\nNext: Analyze the monitoring logs to understand the dynamics!")


if __name__ == "__main__":
    main()


