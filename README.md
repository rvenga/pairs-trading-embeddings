# Pairs Trading with Semantic-Structural Embeddings

Multi-modal similarity measures for pairs trading, inspired by spreadsheet template discovery research.

## Setup

### Prerequisites
- Python 3.11+
- Conda
- Poetry

### Installation
```bash
# Create conda environment
conda create -n pairs-trading python=3.11
conda activate pairs-trading

# Install dependencies
pip install poetry
poetry install
```

## Project Structure
```
pairs-trading-embeddings/
├── data/
│   ├── raw/              # Raw data (gitignored)
│   └── processed/        # Processed data (gitignored)
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   └── data_collection.py
├── tests/                # Tests
├── pyproject.toml        # Poetry dependencies
└── README.md
```

## Usage

### Data Collection
```bash
python src/data_collection.py
```

## Progress
- [x] Project setup
- [ ] Data collection
- [ ] Feature engineering
- [ ] Baseline: Traditional cointegration
- [ ] Baseline: Semantic similarity
- [ ] Hybrid approach