# Master Thesis: Glucose Prediction Using LSTM

This repository contains the implementation and experiments for a master thesis project focused on predicting glucose levels using Long Short-Term Memory (LSTM).

## Project Overview

This project analyzes patient data including glucose measurements, bolus insulin doses, meal information, and physical activity (steps) to predict future glucose levels. The system processes raw patient data and uses machine learning models to make predictions that could assist in diabetes management.

## Project Structure

```
├── data/
│   ├── raw/                          # Original XML patient data
│   │   ├── train/                    # Training data (6 patients)
│   │   └── test/                     # Testing data (6 patients)
│   └── processed/                    # Processed datasets
│       ├── bolus_meal_steps_data/    # Combined feature datasets
│       ├── train/                    # Training datasets (CSV/Parquet)
│       └── test/                     # Testing datasets (CSV/Parquet)
├── src/
│   ├── main.py                       # Main execution script
│   ├── parser.py                     # XML data parser
│   ├── preprocessing.py              # Data preprocessing utilities
│   ├── data_handling.py              # Data loading and management
│   ├── bolus_feature_engineering.py  # Bolus-related features
│   ├── meals_feature_engineering.py  # Meal-related features
│   ├── steps_feature_engineering.py  # Physical activity features
│   ├── model_handling.py             # Model training and evaluation
│   ├── validation.py                 # Model validation utilities
│   └── tuning/
│       └── hyperparameter_tuning.py  # Hyperparameter optimization
├── models_result/                    # Saved models and results
│   ├── Model Iteration 1/
│   ├── Model Iteration 2/
│   ├── Model Iteration 3/
│   ├── Model Iteration 4/
│   └── Model Iteration 5/
├── Data processing.ipynb             # Jupyter notebook for data exploration
```

## Dataset

The dataset includes data from 6 patients (IDs: 559, 563, 570, 575, 588, 591), split into training and testing sets. Each patient's data contains:

- **Glucose measurements**: Blood glucose readings over time
- **Bolus insulin data**: Insulin dose information
- **Meal information**: Carbohydrate intake and meal timing
- **Physical activity**: Step counts and activity levels

### Data Format

- **Raw data**: XML format (in `data/raw/`)
- **Processed data**: CSV and Parquet formats (in `data/processed/`)

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow/Keras
- NumPy, Pandas
- Scikit-learn
- Keras Tuner (for hyperparameter optimization)

### Setup

```bash
# Clone the repository
git clone https://github.com/Maria-Lavinia/glucose-prediction.git
cd master-thesis

```

## Usage

### 1. Data Processing

### 2. Training the Model

### 3. Hyperparameter Tuning

For all of the above actions please run the main script:

```bash
python src/main.py
```

Results are saved in `src/hyperparameter_tuning/glucose_prediction_lstm/`.

## Model Iterations

The project includes 5 model iterations, each exploring different:
- Feature combinations
- Network architectures
- Hyperparameter configurations
- Training strategies

Results for each iteration are stored in `models_result/Model Iteration X/`.

## Features

The model uses engineered features from multiple data sources:

- **Bolus features**: Insulin dosage patterns and timing
- **Meal features**: Carbohydrate intake, meal timing, and composition
- **Activity features**: Physical activity levels and patterns
- **Temporal features**: Time-based patterns and trends

## Results

Model performance metrics and predictions are saved in the `models_result/` directory, organized by iteration.

## Contributing

This is a thesis project. For questions or suggestions, please contact the authors.

## Authors

- Adelina Radulescu, Master Student at Roskilde University
- Maria Otelea, Master Student at Roskilde University
- Pawel Stepien, Master Student at Roskilde University

## Acknowledgments

Roskilde University (RUC)
