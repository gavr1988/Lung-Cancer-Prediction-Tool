# Lung Cancer Prediction Tool

A machine learning-based tool that predicts the likelihood of lung cancer in patients using a Random Forest Regressor model trained on survey data.

## Overview

This project builds a predictive model to assess lung cancer risk based on patient health attributes. The tool:
- Cleans and preprocesses survey lung cancer data
- Trains a Random Forest model with hyperparameter tuning
- Evaluates model performance on training and test sets
- Makes predictions on new patient data

## Features Used

The model uses the following 15 patient health features:
- **GENDER** (M/F) - Patient gender
- **AGE** - Patient age
- **SMOKING** - Smoking frequency (scale)
- **YELLOW_FINGERS** - Presence of yellowing (scale)
- **ANXIETY** - Anxiety level (scale)
- **PEER_PRESSURE** - Peer pressure influence (scale)
- **CHRONIC_DISEASE** - Chronic disease presence (scale)
- **FATIGUE** - Fatigue level (scale)
- **ALLERGY** - Allergy presence (scale)
- **WHEEZING** - Wheezing frequency (scale)
- **ALCOHOL_CONSUMING** - Alcohol consumption (scale)
- **COUGHING** - Coughing frequency (scale)
- **SHORTNESS_OF_BREATH** - Breathing difficulty (scale)
- **SWALLOWING_DIFFICULTY** - Difficulty swallowing (scale)
- **CHEST_PAIN** - Chest pain level (scale)

**Target:** LUNG_CANCER (YES/NO or 1/0)

## Prerequisites

- Python 3.9+
- pandas
- scikit-learn
- numpy

## Installation

### 1. Create a Virtual Environment

```bash
cd /path/to/Lung-Cancer-Prediction-Tool
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install pandas scikit-learn numpy
```

## Usage

### Running the Script

```bash
python run.py
```

### Output

The script will output:
1. Column headers and data preview
2. Data type information
3. Number of duplicate rows (duplicates are retained for analysis)
4. Missing value summary
5. Cleaned column names
6. **Best hyperparameters** found during grid search
7. **Training accuracy** (R² score)
8. **Testing accuracy** (R² score)
9. **Prediction** for the sample patient data
10. **Likelihood** of lung cancer (0-1 scale)

### Example Output
```
Best Settings: {'model__n_estimators': 30}
Training Accuracy: 0.88
Testing Accuracy: 0.53
Likelihood Patient has lung cancer: 0.85
Summary: Based on the live input data, the model predicts LUNG_CANCER = 0.85
```

## Project Structure

```
Lung-Cancer-Prediction-Tool/
├── run.py                              # Main script
├── survey lung cancer.csv              # Raw survey data
├── cleaned_survey_lung_cancer.csv      # Cleaned data (generated)
├── README.md                           # This file
└── .venv/                              # Virtual environment
```

## Data Processing Steps

1. **Load Data** - Read CSV file
2. **Clean Column Names** - Remove whitespace and replace spaces with underscores
3. **Check Data Quality** - Display column types, duplicates, and missing values
4. **Encode Categorical Variables**:
   - GENDER: M→1, F→2
   - LUNG_CANCER: YES→1, NO→0
5. **Export Cleaned Data** - Save to `cleaned_survey_lung_cancer.csv`

## Model Details

### Algorithm
- **Random Forest Regressor** - Ensemble learning method combining multiple decision trees
- **Hyperparameters Tuned**: Number of estimators (n_estimators: 10, 20, 30)

### Pipeline Steps
1. **StandardScaler** - Normalize features (zero mean, unit variance)
2. **Random Forest Regressor** - Train predictive model

### Cross-Validation
- **CV Folds**: 2 folds
- **Evaluation Metric**: R² Score (coefficient of determination)
  - 1.0 = Perfect prediction
  - 0.0 = No predictive power
  - Negative = Worse than baseline

## Making Predictions

To make a prediction on new patient data, modify the `X_live` DataFrame in `run.py`:

```python
X_live = pd.DataFrame(data={
    'GENDER': [2],                      # 1=M, 2=F
    'AGE': [65],
    'SMOKING': [2],
    'YELLOW_FINGERS': [2],
    'ANXIETY': [1],
    'PEER_PRESSURE': [1],
    'CHRONIC_DISEASE': [2],
    'FATIGUE': [2],
    'ALLERGY': [1],
    'WHEEZING': [2],
    'ALCOHOL_CONSUMING': [2],
    'COUGHING': [2],
    'SHORTNESS_OF_BREATH': [2],
    'SWALLOWING_DIFFICULTY': [1],
    'CHEST_PAIN': [2]
}, index=[0])
```

Then run:
```bash
python run.py
```

## Notes

- **Duplicate Rows**: 33 duplicate rows are retained as they may represent valid data points
- **No Missing Values**: The dataset has complete records with no missing values
- **Model Performance**: The testing accuracy (0.53) is lower than training accuracy (0.88), which may indicate potential overfitting

## Future Improvements

- Feature importance analysis
- Class imbalance handling (if present)
- Cross-validation with more folds
- Other algorithms (Logistic Regression, SVM, Gradient Boosting)
- Feature scaling alternatives
- Hyperparameter grid expansion

## Author

Lung Cancer Prediction Tool - Machine Learning Project
