import yfinance as yf
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import TEST_TICKERS, DATA_DIR, START_DATE, END_DATE

def collect_fundamental_data(tickers, start_date=START_DATE, end_date=END_DATE):
    """
    Collect fundamental data for all tickers with time-varying metrics.
    """
    fundamentals = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get historical price data
            price_hist = stock.history(start=start_date, end=end_date)
            if price_hist.empty:
                print(f"✗ No price data for {ticker}, skipping")
                continue

            # Get quarterly financial statements
            try:
                fin_quarterly = stock.quarterly_financials.T
                bs_quarterly = stock.quarterly_balance_sheet.T
                fin_quarterly.index = pd.to_datetime(fin_quarterly.index).tz_localize(None)
                bs_quarterly.index = pd.to_datetime(bs_quarterly.index).tz_localize(None)
            except Exception as e:
                print(f"✗ No quarterly data for {ticker}: {e}")
                continue

            # Static info (sector, industry, shares outstanding)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            shares_outstanding = info.get('sharesOutstanding', None)
            
            # Create a daily forward-filled fundamental dataframe
            date_range = pd.DataFrame(index=price_hist.index)
            date_range.index = pd.to_datetime(date_range.index).tz_localize(None)
            
            # Helper to forward-fill quarterly data to daily
            def forward_fill_metric(df, col):
                if col not in df.columns:
                    return pd.Series(index=date_range.index, dtype=float)
                return df[col].reindex(date_range.index, method='ffill')
            
            # Helper to get metric with fallback options
            def get_metric_with_fallback(df, options, metric_name=""):
                """Try multiple column names, return first that exists."""
                for col_name in options:
                    if col_name in df.columns:
                        return forward_fill_metric(df, col_name), col_name
                if metric_name:
                    print(f"  ⚠ {ticker}: {metric_name} not found (tried: {options[0]})")
                return pd.Series(index=date_range.index, dtype=float), None
            
            # Extract and forward-fill quarterly metrics with fallback options
            net_income, ni_col = get_metric_with_fallback(fin_quarterly, ['Net Income', 'Net Income Common Stockholders'], 'Net Income')
            total_revenue, rev_col = get_metric_with_fallback(fin_quarterly, ['Total Revenue', 'Operating Revenue', 'Revenue'], 'Total Revenue')
            stockholder_equity, equity_col = get_metric_with_fallback(bs_quarterly, ['Stockholders Equity', 'Stockholder Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'], 'Equity')
            total_assets, assets_col = get_metric_with_fallback(bs_quarterly, ['Total Assets', 'Total Assets Net'], 'Total Assets')
            total_debt, debt_col = get_metric_with_fallback(bs_quarterly, ['Total Debt', 'Net Debt'], 'Total Debt')
            current_assets, ca_col = get_metric_with_fallback(bs_quarterly, ['Current Assets'], 'Current Assets')
            current_liabilities, cl_col = get_metric_with_fallback(bs_quarterly, ['Current Liabilities'], 'Current Liabilities')
            
            # Calculate daily metrics using historical prices
            for date in price_hist.index:
                date_normalized = pd.Timestamp(date).tz_localize(None)
                price = price_hist.loc[date, 'Close']
                
                # Get forward-filled quarterly values for this date
                ni = net_income.get(date_normalized, np.nan)
                rev = total_revenue.get(date_normalized, np.nan)
                equity = stockholder_equity.get(date_normalized, np.nan)
                assets = total_assets.get(date_normalized, np.nan)
                debt = total_debt.get(date_normalized, np.nan)
                cur_assets = current_assets.get(date_normalized, np.nan)
                cur_liab = current_liabilities.get(date_normalized, np.nan)
                
                # Calculate time-varying metrics
                market_cap = price * shares_outstanding if shares_outstanding else np.nan
                
                # PE Ratio: Price / (Net Income / Shares Outstanding)
                eps = ni / shares_outstanding if shares_outstanding and not pd.isna(ni) else np.nan
                pe_ratio = price / eps if eps and eps > 0 else np.nan
                
                # PB Ratio: Price / (Stockholder Equity / Shares Outstanding)
                book_value_per_share = equity / shares_outstanding if shares_outstanding and not pd.isna(equity) else np.nan
                pb_ratio = price / book_value_per_share if book_value_per_share and book_value_per_share > 0 else np.nan
                
                # ROE: Net Income / Stockholder Equity (annualized quarterly)
                roe = (ni * 4) / equity if not pd.isna(ni) and not pd.isna(equity) and equity != 0 else np.nan
                
                # ROA: Net Income / Total Assets (annualized quarterly)
                roa = (ni * 4) / assets if not pd.isna(ni) and not pd.isna(assets) and assets != 0 else np.nan
                
                # Profit Margin: Net Income / Revenue
                profit_margin = ni / rev if not pd.isna(ni) and not pd.isna(rev) and rev != 0 else np.nan
                
                # Debt to Equity
                debt_to_equity = debt / equity if not pd.isna(debt) and not pd.isna(equity) and equity != 0 else np.nan
                
                # Current Ratio
                current_ratio = cur_assets / cur_liab if not pd.isna(cur_assets) and not pd.isna(cur_liab) and cur_liab != 0 else np.nan

                fundamentals.append({
                    'Date': str(date.date()),
                    'Ticker': ticker,
                    'Sector': sector,
                    'Industry': industry,
                    'Market_Cap': market_cap,
                    'PE_Ratio': pe_ratio,
                    'PB_Ratio': pb_ratio,
                    'ROE': roe,
                    'ROA': roa,
                    'Profit_Margin': profit_margin,
                    'Debt_to_Equity': debt_to_equity,
                    'Current_Ratio': current_ratio,
                })
            
            # Report what was successfully collected
            metrics_found = []
            if equity_col: metrics_found.append(f"Equity({equity_col})")
            if debt_col: metrics_found.append(f"Debt({debt_col})")
            if ni_col: metrics_found.append("NI")
            if rev_col: metrics_found.append("Rev")
            if assets_col: metrics_found.append("Assets")
            if ca_col and cl_col: metrics_found.append("CR")
            
            print(f"✓ Collected {ticker} ({len(price_hist)} days) - Metrics: {', '.join(metrics_found) if metrics_found else 'basic only'}")
            
        except Exception as e:
            print(f"✗ Error with {ticker}: {e}")
            continue

    return pd.DataFrame(fundamentals)



fundamentals = collect_fundamental_data(tickers=TEST_TICKERS)
fundamentals.to_csv(DATA_DIR / "fundamentals.csv", index=False)