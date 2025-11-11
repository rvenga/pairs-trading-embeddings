"""
Traditional cointegration-based pairs trading baseline.
End-to-end pipeline: data → pairs → backtest → results.
Implements train-test split to prevent data leakage.
"""

import pandas as pd
from pathlib import Path
from cointegration import find_cointegrated_pairs, split_train_test
from backtest import backtest_portfolio

# Configuration
TRAIN_RATIO = 0.7  # 70% train, 30% test
MAX_PAIRS = 15


def main():
    # Paths
    DATA_DIR = Path("data/raw")
    OUTPUT_DIR = Path("data/processed")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("BASELINE: Traditional Cointegration Pairs Trading")
    print(f"WITH TRAIN-TEST SPLIT ({int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)})")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    prices = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=['Date'])
    print(f"   {len(prices)} rows | {prices['Date'].min().date()} to {prices['Date'].max().date()}")
    
    # Split into train and test
    print(f"\n2. Splitting data ({int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)})...")
    train_prices, test_prices = split_train_test(prices, train_ratio=TRAIN_RATIO)
    print(f"   Train: {train_prices['Date'].min().date()} to {train_prices['Date'].max().date()} ({len(train_prices)} rows)")
    print(f"   Test:  {test_prices['Date'].min().date()} to {test_prices['Date'].max().date()} ({len(test_prices)} rows)")
    
    # Find cointegrated pairs on TRAINING data only
    print("\n3. Finding cointegrated pairs on TRAINING data...")
    pairs = find_cointegrated_pairs(train_prices, max_pairs=MAX_PAIRS)
    pairs.to_csv(OUTPUT_DIR / "cointegrated_pairs.csv", index=False)
    print(f"   {len(pairs)} pairs found (p < 0.05)")
    
    if len(pairs) == 0:
        print("\n⚠ No cointegrated pairs found. Exiting.")
        return
    
    # Backtest on TEST data only (out-of-sample)
    print("\n4. Backtesting on TEST data (out-of-sample)...")
    test_results = backtest_portfolio(test_prices, pairs)
    test_results.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)
    print(f"   Complete")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Out-of-Sample)")
    print("=" * 60)
    print(f"\nTotal pairs tested: {len(test_results)}")
    print(f"Average annualized return: {test_results['Annualized_Return'].mean():.2%}")
    print(f"Average Sharpe: {test_results['Sharpe_Ratio'].mean():.2f}")
    print(f"Average max drawdown: {test_results['Max_Drawdown'].mean():.2%}")
    print(f"Average trades: {test_results['Num_Trades'].mean():.0f}")
    print(f"Average win rate: {test_results['Win_Rate'].mean():.2%}")
    
    print("\n=== TOP 5 PAIRS BY SHARPE ===")
    print(test_results.nlargest(5, 'Sharpe_Ratio')[['Ticker1', 'Ticker2', 'Sharpe_Ratio', 'Annualized_Return']])
    
    print("\n✓ Baseline complete!")
    print("\nNote: All results are out-of-sample (test set only).")


if __name__ == "__main__":
    main()