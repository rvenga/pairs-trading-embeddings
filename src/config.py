from pathlib import Path

# Configuration
# Paths - ensure we work from project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" 
VISUALIZATION_DIR = PROJECT_ROOT / "data" / "visualizations"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2019-01-01"
END_DATE = "2024-11-01"

# Small test universe - 25 stocks across sectors
TEST_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "META", "NVDA",
    # Finance
    "JPM", "BAC", "GS", "MS", "C",
    # Consumer
    "WMT", "TGT", "COST", "HD", "LOW",
    # Healthcare
    "JNJ", "PFE", "UNH", "CVS", "ABBV",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG"
]