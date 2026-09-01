# 🏠 Housing Price Prediction

A machine learning project that predicts housing prices using **Linear Regression**, built with Python, pandas, and scikit-learn.

## 📌 Project Overview

This project analyzes a housing dataset, performs data cleaning and exploratory analysis, encodes categorical features, and trains a Linear Regression model to predict house prices. Model performance is evaluated using MAE, MSE, RMSE, and R² score, along with visual diagnostics.

## 📂 Repository Structure

```
├── main.py                              # End-to-end pipeline script
├── new.ipynb                            # Notebook version of the analysis
├── Housing.csv                          # Dataset
├── Heat_Map.png                         # Correlation heatmap
├── Actual_VS_Predicted_Scatter_Plot.png # Actual vs predicted prices
├── Residual_Plot.png                    # Residuals vs predicted price
├── Resdiual_Distribution.png            # Distribution of residuals
├── requirements.txt                     # Python dependencies
└── README.md
```
## Dataset

## Dataset

The dataset used in this project is the "Housing Prices Dataset"
created by M Yasser H and obtained from Kaggle.

Source:
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data

The dataset is listed on Kaggle under the CC0: Public Domain license.

## ⚙️ Workflow

1. Load and explore the dataset (shape, info, missing values, duplicates)
2. One-hot encode categorical variables
3. Visualize feature correlations with a heatmap
4. Split data into training and testing sets (80/20)
5. Train a Linear Regression model
6. Evaluate performance (MAE, MSE, RMSE, R²)
7. Visualize actual vs predicted prices, residuals, and residual distribution
8. Inspect feature coefficients and predict a sample price

## 📊 Results

| Metric | Description |
|--------|-------------|
| MAE    | Mean Absolute Error |
| MSE    | Mean Squared Error |
| RMSE   | Root Mean Squared Error |
| R²     | Coefficient of determination |

*(Exact values are printed when you run `main.py`.)*

## 🚀 Getting Started

### Prerequisites
- Python 3.8+

### Installation
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

### Run
```bash
python main.py
```

## 🛠️ Built With
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — model training and evaluation

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
