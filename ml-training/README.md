# Junction ML

A machine learning project for predicting junction lane suggestions using PyTorch Geometric and junction graph data from OSMScout.

## Project Structure

```
ml-training/
├── src/junction_ml/           # Main package
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # PyTorch Geometric models
│   ├── training/              # Training loops and utilities
│   └── utils/                 # Utility functions
├── notebooks/                 # Jupyter notebooks for experimentation
├── scripts/                   # Training and evaluation scripts
└── tests/                     # Unit tests
```

## Installation

1. Create and activate virtual environment:
```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
poetry install
```

## Usage

The project processes junction graph JSON files from the `../tmp-junctions/` directory to train models that predict:
- `suggestedFrom`: Suggested starting lane
- `suggestedTo`: Suggested ending lane  
- `suggestedTurn`: Suggested turn type

## Features

- Graph Neural Network models using PyTorch Geometric
- Multi-task prediction for lane suggestions
- Comprehensive data preprocessing pipeline
- Training with validation and testing splits
- Model evaluation and visualization tools
