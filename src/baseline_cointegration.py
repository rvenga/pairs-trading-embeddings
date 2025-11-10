"""
Traditional cointegration-based pairs trading baseline.
End-to-end pipeline: data → pairs → backtest → results.
"""

import pandas as pd
from pathlib import Path
from cointegration import find_cointegrated_pairs
from backtest import backtest_portfolio


def main():
    # Paths
    DATA_DIR = Path("data/raw")
    OUTPUT_DIR = Path("data/processed")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("BASELINE: Traditional Cointegration Pairs Trading")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    prices = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=['Date'])
    print(f"   Loaded {len(prices)} price observations")
    
    # Find cointegrated pairs
    print("\n2. Finding cointegrated pairs...")
    pairs = find_cointegrated_pairs(prices, max_pairs=15)
    pairs.to_csv(OUTPUT_DIR / "cointegrated_pairs.csv", index=False)
    print(f"   Found {len(pairs)} pairs")
    print(f"   Saved to {OUTPUT_DIR / 'cointegrated_pairs.csv'}")
    
    # Backtest
    print("\n3. Backtesting pairs...")
    results = backtest_portfolio(prices, pairs)
    results.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)
    print(f"   Saved to {OUTPUT_DIR / 'backtest_results.csv'}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nTotal pairs tested: {len(results)}")
    print(f"Average return: {results['Total_Return'].mean():.2%}")
    print(f"Average Sharpe: {results['Sharpe_Ratio'].mean():.2f}")
    print(f"Average max drawdown: {results['Max_Drawdown'].mean():.2%}")
    print(f"Average trades: {results['Num_Trades'].mean():.0f}")
    print(f"Average win rate: {results['Win_Rate'].mean():.2%}")
    
    print("\n=== TOP 5 PAIRS BY SHARPE ===")
    print(results.nlargest(5, 'Sharpe_Ratio')[['Ticker1', 'Ticker2', 'Sharpe_Ratio', 'Total_Return']])
    
    print("\n✓ Baseline complete!")


if __name__ == "__main__":
    main()