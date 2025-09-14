# 🔧 PHM America 2023 - Vibration Analysis & Health Prediction

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

This project implements a comprehensive machine learning pipeline for the **PHM America 2023 Data Challenge**, focusing on predictive health monitoring through vibration signal analysis. The solution processes industrial vibration data to predict equipment health levels, enabling proactive maintenance strategies.

## 🎯 Key Features

- **Advanced Signal Processing**: Automated parsing and downsampling of high-frequency vibration data
- **Feature Engineering**: Extraction of time-domain and frequency-domain features from multi-axis accelerometer signals
- **Multi-Model Approach**: Implementation of multiple classification algorithms with automated model selection
- **Ensemble Learning**: Voting classifier combining top-performing models for improved accuracy
- **Anomaly Detection**: Integrated outlier detection for robust predictions
- **Automated Pipeline**: End-to-end solution from raw data to submission file generation

## 🏗️ Project Structure

```
phm-america-2023/
│
├── 📁 data/
│   ├── raw/                    # Original PHM2023 dataset
│   │   └── B - PHM America 2023 - Dataset/
│   │       ├── Data_Challenge_PHM2023_training_data/
│   │       └── Data_Challenge_PHM2023_test_data/
│   └── processed/              # Processed data files
│       ├── train_data.pkl
│       ├── test_data.pkl
│       ├── train_data_dowsampled.pkl
│       └── X_with_features.pkl
│
├── 📁 utils/
│   ├── parsing.py              # Data parsing utilities
│   ├── downsampling.py         # Signal downsampling functions
│   ├── feature_extraction.py   # Feature engineering pipeline
│   ├── model_class.py          # Model selection framework
│   ├── submission_generator.py # Competition submission creation
│   └── review_results.py       # Results analysis tools
│
├── 📁 models/                  # Saved trained models
│   └── best_model_classification_*.joblib
│
├── 📓 main.ipynb              # Main analysis notebook
├── 📄 submission.csv          # Final submission file
└── 📄 README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/phm-america-2023.git
cd phm-america-2023
```

2. Place the PHM2023 dataset in the appropriate data folder structure

3. Run the main pipeline:
```bash
jupyter notebook main.ipynb
```

## 📊 Pipeline Workflow

### 1. **Data Parsing** 📥
Processes raw vibration signals from multiple sensors:
- Horizontal acceleration
- Axial acceleration  
- Vertical acceleration
- Tachometer signal
- Speed and torque measurements

### 2. **Downsampling** ⚡
Reduces data dimensionality while preserving signal characteristics:
- Configurable time window (default: 3 seconds)
- Maintains critical frequency components

### 3. **Feature Extraction** 🔍
Comprehensive feature engineering including:
- Statistical features (mean, std, skewness, kurtosis)
- Frequency domain features
- Signal energy metrics
- Multicollinearity removal (threshold: 0.85)

### 4. **Model Training** 🤖
Multi-model approach with automated selection:
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machines
- Neural Networks
- Ensemble voting classifier

### 5. **Evaluation & Optimization** 📈
- Cross-validation with stratified k-fold
- F1-macro scoring for imbalanced classes
- Confusion matrix analysis
- Feature importance ranking
- Ensemble optimization (top-3 models)

### 6. **Prediction & Submission** 🎯
- Confidence-based predictions
- Anomaly detection integration
- Automated submission file generation

## 📈 Results

The solution achieves robust performance through:
- **Ensemble Learning**: Combining multiple models for improved accuracy
- **Confidence Thresholding**: Managing prediction uncertainty
- **Anomaly Detection**: Identifying out-of-distribution samples

Key metrics:
- F1-macro score optimization for balanced performance across all health levels
- Comprehensive validation on held-out test set
- Visual analysis through confusion matrices and feature importance plots

## 🛠️ Key Components

### Model Selection Framework
```python
selector = ModelSelectorClassification(
    scoring="f1_macro",
    cv_folds=3,
    random_state=42
)
selector.fit(X_tr, y_tr, feature_names)
```

### Feature Engineering
```python
X_with_features = expand_features(
    df, 
    array_cols=['horizontal_acceleration', 'axial_acceleration', ...],
    scalar_cols=['velocita', 'torque']
)
```

### Submission Generation
```python
generator = SubmissionGenerator()
submission_df, info = generator.create_submission_simple(
    X_tr, y_tr, X_te, test_df, model,
    conf_thresh=0.6,
    contamination='auto',
    alpha=0.6
)
```

## 📊 Visualization Tools

The project includes comprehensive visualization capabilities:
- Model performance comparison plots
- Confusion matrices (normalized and raw)
- Feature importance rankings
- Prediction confidence distributions
- Results analysis dashboards



## 🤝 Contributing


|Nome | GitHub |
|-----------|--------|
| 🔨 `` | [Click here](https://github.com/) |
| 🪛 `` | [Click here](https://github.com/) |
| 🪛 `` | [Click here](https://github.com/) |

---
