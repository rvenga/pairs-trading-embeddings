"""
Visualize fundamental data.
Creates comprehensive visualizations including time series, distributions,
correlations, and sector comparisons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

#TODO: Move this to make import work for all scripts
# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_DIR, VISUALIZATION_DIR


def load_data():
    """Load fundamental data."""
    fundamentals = pd.read_csv(DATA_DIR / "raw" / "fundamentals.csv")
    fundamentals['Date'] = pd.to_datetime(fundamentals['Date'])
    return fundamentals


def plot_time_series(fundamentals):
    """Plot time series of key metrics for sample tickers."""
    print("\n📊 Creating time series plots...")
    
    # Select key metrics and sample tickers
    metrics = ['PE_Ratio', 'ROE', 'ROA', 'Debt_to_Equity']
    sample_tickers = fundamentals['Ticker'].unique()[:5]  # First 5 tickers
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for ticker in sample_tickers:
            data = fundamentals[fundamentals['Ticker'] == ticker]
            # Filter out outliers for better visualization
            metric_data = data[['Date', metric]].dropna()
            if len(metric_data) > 0:
                # Remove extreme outliers (beyond 3 std)
                if metric_data[metric].std() > 0:
                    mean_val = metric_data[metric].mean()
                    std_val = metric_data[metric].std()
                    metric_data = metric_data[
                        (metric_data[metric] >= mean_val - 3*std_val) &
                        (metric_data[metric] <= mean_val + 3*std_val)
                    ]
                
                if len(metric_data) > 0:
                    ax.plot(metric_data['Date'], metric_data[metric], 
                           label=ticker, alpha=0.7, linewidth=1.5)
        
        ax.set_title(f'{metric.replace("_", " ")} Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "fundamentals" / "timeseries.png", bbox_inches='tight')
    print(f"✓ Saved: {VISUALIZATION_DIR / 'fundamentals' / 'timeseries.png'}")
    plt.close()


def plot_distributions(fundamentals):
    """Plot distributions of key fundamental metrics."""
    print("\n📊 Creating distribution plots...")
    
    metrics = ['PE_Ratio', 'ROE', 'ROA', 'Profit_Margin', 
               'Debt_to_Equity', 'Current_Ratio']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Remove NaN and extreme outliers
        data = fundamentals[metric].dropna()
        if len(data) > 0:
            # Remove values beyond 99th percentile for better visualization
            q99 = data.quantile(0.99)
            q01 = data.quantile(0.01)
            data_filtered = data[(data >= q01) & (data <= q99)]
            
            # Histogram
            ax.hist(data_filtered, bins=50, alpha=0.6, color='steelblue', 
                   edgecolor='black', linewidth=0.5)
            
            # Add vertical lines for mean and median
            mean_val = data_filtered.mean()
            median_val = data_filtered.median()
            ax.axvline(mean_val, color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', 
                      linewidth=2, label=f'Median: {median_val:.2f}')
            
            ax.set_title(f'{metric.replace("_", " ")} Distribution', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel(metric.replace("_", " "))
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "fundamentals" / "distributions.png", bbox_inches='tight')
    print(f"✓ Saved: {VISUALIZATION_DIR / 'fundamentals' / 'distributions.png'}")
    plt.close()


def plot_correlation_heatmap(fundamentals):
    """Plot correlation heatmap of fundamental metrics."""
    print("\n📊 Creating correlation heatmap...")
    
    # Select numeric columns
    metrics = ['Market_Cap', 'PE_Ratio', 'PB_Ratio', 'ROE', 'ROA', 
               'Profit_Margin', 'Debt_to_Equity', 'Current_Ratio']
    
    # Get latest data for each ticker (to avoid time-series correlation)
    latest_data = fundamentals.sort_values('Date').groupby('Ticker').last().reset_index()
    
    # Calculate correlation matrix
    corr_data = latest_data[metrics].dropna()
    corr_matrix = corr_data.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Correlation Matrix of Fundamental Metrics', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "fundamentals" / "correlation.png", bbox_inches='tight')
    print(f"✓ Saved: {VISUALIZATION_DIR / 'fundamentals' / 'correlation.png'}")
    plt.close() 


def plot_sector_comparison(fundamentals):
    """Plot sector-wise comparison of key metrics."""
    print("\n📊 Creating sector comparison plots...")
    
    # Get latest data for each ticker
    latest_data = fundamentals.sort_values('Date').groupby('Ticker').last().reset_index()
    
    # Metrics to compare
    metrics = ['PE_Ratio', 'ROE', 'Profit_Margin', 'Debt_to_Equity']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Calculate sector averages
        sector_data = latest_data.groupby('Sector')[metric].agg(['mean', 'std']).reset_index()
        sector_data = sector_data.dropna()
        
        if len(sector_data) > 0:
            # Sort by mean value
            sector_data = sector_data.sort_values('mean', ascending=False)
            
            # Bar plot with error bars
            x_pos = np.arange(len(sector_data))
            ax.bar(x_pos, sector_data['mean'], 
                  yerr=sector_data['std'], 
                  alpha=0.7, color='steelblue',
                  error_kw={'linewidth': 2, 'ecolor': 'darkred', 'capsize': 5})
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sector_data['Sector'], rotation=45, ha='right')
            ax.set_title(f'Average {metric.replace("_", " ")} by Sector', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace("_", " "))
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (mean_val, std_val) in enumerate(zip(sector_data['mean'], sector_data['std'])):
                ax.text(i, mean_val + std_val, f'{mean_val:.2f}', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "fundamentals" / "sector_comparison.png", bbox_inches='tight')
    print(f"✓ Saved: {VISUALIZATION_DIR / 'fundamentals' / 'sector_comparison.png'}")
    plt.close()


def plot_dashboard(fundamentals):
    """Create comprehensive dashboard with multiple views."""
    print("\n📊 Creating comprehensive dashboard...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. PE Ratio by Sector (box plot)
    ax1 = fig.add_subplot(gs[0, 0])
    latest_data = fundamentals.sort_values('Date').groupby('Ticker').last().reset_index()
    sector_pe = latest_data[['Sector', 'PE_Ratio']].dropna()
    # Filter outliers
    q99 = sector_pe['PE_Ratio'].quantile(0.99)
    sector_pe = sector_pe[sector_pe['PE_Ratio'] <= q99]
    
    sectors = sector_pe['Sector'].unique()
    pe_by_sector = [sector_pe[sector_pe['Sector'] == s]['PE_Ratio'].values for s in sectors]
    
    ax1.boxplot(pe_by_sector, labels=sectors)
    ax1.set_title('PE Ratio Distribution by Sector', fontweight='bold')
    ax1.set_ylabel('PE Ratio')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. ROE vs ROA Scatter
    ax2 = fig.add_subplot(gs[0, 1])
    scatter_data = latest_data[['ROE', 'ROA', 'Sector']].dropna()
    # Filter outliers
    scatter_data = scatter_data[
        (scatter_data['ROE'] >= scatter_data['ROE'].quantile(0.01)) &
        (scatter_data['ROE'] <= scatter_data['ROE'].quantile(0.99)) &
        (scatter_data['ROA'] >= scatter_data['ROA'].quantile(0.01)) &
        (scatter_data['ROA'] <= scatter_data['ROA'].quantile(0.99))
    ]
    
    for sector in scatter_data['Sector'].unique():
        sector_data = scatter_data[scatter_data['Sector'] == sector]
        ax2.scatter(sector_data['ROE'], sector_data['ROA'], 
                   label=sector, alpha=0.6, s=50)
    
    ax2.set_title('ROE vs ROA', fontweight='bold')
    ax2.set_xlabel('ROE')
    ax2.set_ylabel('ROA')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Market Cap Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    market_cap_data = latest_data['Market_Cap'].dropna() / 1e9  # Convert to billions
    ax3.hist(market_cap_data, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax3.set_title('Market Cap Distribution', fontweight='bold')
    ax3.set_xlabel('Market Cap (Billions $)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Profit Margin Time Series
    ax4 = fig.add_subplot(gs[1, :2])
    sample_tickers = fundamentals['Ticker'].unique()[:5]
    for ticker in sample_tickers:
        ticker_data = fundamentals[fundamentals['Ticker'] == ticker]
        profit_data = ticker_data[['Date', 'Profit_Margin']].dropna()
        if len(profit_data) > 0:
            ax4.plot(profit_data['Date'], profit_data['Profit_Margin'], 
                    label=ticker, linewidth=2, alpha=0.7)
    
    ax4.set_title('Profit Margin Over Time (Sample Tickers)', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Profit Margin')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Debt to Equity by Sector
    ax5 = fig.add_subplot(gs[1, 2])
    debt_data = latest_data.groupby('Sector')['Debt_to_Equity'].mean().sort_values(ascending=False)
    debt_data = debt_data[debt_data <= debt_data.quantile(0.95)]  # Remove outliers
    
    ax5.barh(debt_data.index, debt_data.values, color='coral', alpha=0.7)
    ax5.set_title('Avg Debt-to-Equity by Sector', fontweight='bold')
    ax5.set_xlabel('Debt-to-Equity Ratio')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Current Ratio Distribution
    ax6 = fig.add_subplot(gs[2, 0])
    current_ratio_data = latest_data['Current_Ratio'].dropna()
    current_ratio_data = current_ratio_data[(current_ratio_data >= 0) & (current_ratio_data <= 5)]  # Reasonable range
    
    ax6.hist(current_ratio_data, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax6.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Critical Level')
    ax6.set_title('Current Ratio Distribution', fontweight='bold')
    ax6.set_xlabel('Current Ratio')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. PE Ratio vs PB Ratio
    ax7 = fig.add_subplot(gs[2, 1])
    valuation_data = latest_data[['PE_Ratio', 'PB_Ratio', 'Sector']].dropna()
    # Filter outliers
    valuation_data = valuation_data[
        (valuation_data['PE_Ratio'] >= valuation_data['PE_Ratio'].quantile(0.05)) &
        (valuation_data['PE_Ratio'] <= valuation_data['PE_Ratio'].quantile(0.95)) &
        (valuation_data['PB_Ratio'] >= valuation_data['PB_Ratio'].quantile(0.05)) &
        (valuation_data['PB_Ratio'] <= valuation_data['PB_Ratio'].quantile(0.95))
    ]
    
    for sector in valuation_data['Sector'].unique()[:5]:  # Limit to 5 sectors for clarity
        sector_data = valuation_data[valuation_data['Sector'] == sector]
        ax7.scatter(sector_data['PE_Ratio'], sector_data['PB_Ratio'], 
                   label=sector, alpha=0.6, s=50)
    
    ax7.set_title('PE Ratio vs PB Ratio', fontweight='bold')
    ax7.set_xlabel('PE Ratio')
    ax7.set_ylabel('PB Ratio')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary Statistics Table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_metrics = ['PE_Ratio', 'ROE', 'ROA', 'Profit_Margin', 'Debt_to_Equity']
    summary_data = []
    
    for metric in summary_metrics:
        data = latest_data[metric].dropna()
        if len(data) > 0:
            # Filter outliers for statistics
            q99 = data.quantile(0.99)
            q01 = data.quantile(0.01)
            data_filtered = data[(data >= q01) & (data <= q99)]
            
            summary_data.append([
                metric.replace('_', ' '),
                f'{data_filtered.mean():.2f}',
                f'{data_filtered.median():.2f}',
                f'{data_filtered.std():.2f}'
            ])
    
    table = ax8.table(
        cellText=summary_data,
        colLabels=['Metric', 'Mean', 'Median', 'Std Dev'],
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax8.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('Fundamental Data Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(VISUALIZATION_DIR / "fundamentals" / "dashboard.png", bbox_inches='tight')
    print(f"✓ Saved: {VISUALIZATION_DIR / 'fundamentals' / 'dashboard.png'}")
    plt.close()


def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("FUNDAMENTAL DATA VISUALIZATION")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading fundamental data...")
    fundamentals = load_data()
    print(f"✓ Loaded {len(fundamentals)} records for {fundamentals['Ticker'].nunique()} tickers")
    print(f"  Date range: {fundamentals['Date'].min()} to {fundamentals['Date'].max()}")
    print(f"  Sectors: {fundamentals['Sector'].nunique()}")
    
    # Generate visualizations
    plot_time_series(fundamentals)
    plot_distributions(fundamentals)
    plot_correlation_heatmap(fundamentals)
    plot_sector_comparison(fundamentals)
    plot_dashboard(fundamentals)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\n✓ All visualizations saved to: {VISUALIZATION_DIR}")
    print("\nGenerated files:")
    print("  - fundamentals_timeseries.png")
    print("  - fundamentals_distributions.png")
    print("  - fundamentals_correlation.png")
    print("  - fundamentals_sector_comparison.png")
    print("  - fundamentals_dashboard.png")


if __name__ == "__main__":
    main()

