"""
Visualize cointegration monitoring behavior.
Shows when pairs stopped and why.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

#TODO: Move this to make import work for all scripts
# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_DIR, VISUALIZATION_DIR

# Set style
sns.set_style('whitegrid')

# Load results
results_no_mon = pd.read_csv(DATA_DIR / "processed" / "backtest_no_monitoring.csv")
results_with_mon = pd.read_csv(DATA_DIR / "processed" / "backtest_with_monitoring.csv")

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Return comparison
pairs = results_with_mon[['Ticker1', 'Ticker2']].apply(lambda x: f"{x[0]}-{x[1]}", axis=1)
x = range(len(pairs))
width = 0.35

axes[0, 0].bar([i - width/2 for i in x], results_no_mon['Annualized_Return'], 
               width, label='Without Monitoring', alpha=0.7, color='red')
axes[0, 0].bar([i + width/2 for i in x], results_with_mon['Annualized_Return'], 
               width, label='With Monitoring', alpha=0.7, color='green')
axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[0, 0].set_ylabel('Annualized Return')
axes[0, 0].set_title('Return Comparison: With vs Without Monitoring')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(pairs, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Sharpe comparison
axes[0, 1].bar([i - width/2 for i in x], results_no_mon['Sharpe_Ratio'], 
               width, label='Without Monitoring', alpha=0.7, color='red')
axes[0, 1].bar([i + width/2 for i in x], results_with_mon['Sharpe_Ratio'], 
               width, label='With Monitoring', alpha=0.7, color='green')
axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[0, 1].set_ylabel('Sharpe Ratio')
axes[0, 1].set_title('Sharpe Ratio Comparison')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(pairs, rotation=45, ha='right')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Utilization rate
axes[1, 0].bar(x, results_with_mon['Utilization'] * 100, color='steelblue', alpha=0.7)
axes[1, 0].axhline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
axes[1, 0].set_ylabel('Utilization (%)')
axes[1, 0].set_title('Pair Utilization (% of time active)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(pairs, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Summary statistics
summary_data = {
    'Metric': ['Avg Return', 'Avg Sharpe', 'Avg Max DD', 'Avg Trades'],
    'Without Mon': [
        results_no_mon['Annualized_Return'].mean(),
        results_no_mon['Sharpe_Ratio'].mean(),
        results_no_mon['Max_Drawdown'].mean(),
        results_no_mon['Num_Trades'].mean()
    ],
    'With Mon': [
        results_with_mon['Annualized_Return'].mean(),
        results_with_mon['Sharpe_Ratio'].mean(),
        results_with_mon['Max_Drawdown'].mean(),
        results_with_mon['Num_Trades'].mean()
    ]
}
summary_df = pd.DataFrame(summary_data)

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=summary_df.values, 
                         colLabels=summary_df.columns,
                         cellLoc='center',
                         loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
axes[1, 1].set_title('Summary Statistics', pad=20)

plt.tight_layout()
plt.savefig(VISUALIZATION_DIR / "monitoring" / "comparison.png", dpi=150, bbox_inches='tight')
print("✓ Saved: " + str(VISUALIZATION_DIR / "monitoring" / "comparison.png"))

# Plot individual monitoring logs
print("\nGenerating individual pair monitoring plots...")
log_files = list(DATA_DIR / "processed" / "monitoring_log_*.csv")

for log_file in log_files[:3]:  # Plot first 3 as examples
    pair_name = log_file.name.replace('monitoring_log_', '')
    
    log = pd.read_csv(log_file)
    log['Date'] = pd.to_datetime(log['Date'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(log['Date'], log['P_Value'], marker='o', linewidth=2, markersize=4)
    ax.axhline(0.05, color='green', linestyle='--', label='Pass (p<0.05)', alpha=0.7)
    ax.axhline(0.5, color='red', linestyle='--', label='Stop (p>0.5)', alpha=0.7)
    ax.fill_between(log['Date'], 0, 0.05, alpha=0.2, color='green')
    ax.fill_between(log['Date'], 0.5, 1, alpha=0.2, color='red')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cointegration P-Value')
    ax.set_title(f'Cointegration Monitoring: {pair_name.replace("_", "-")}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "monitoring" / f"{pair_name}.png", dpi=150)
    
print(f"✓ Generated {len(log_files[:3])} monitoring plots")
print("\n✓ All visualizations complete!")