# Bengaluru House Price Prediction — Multiple Linear Regression

This project builds an end-to-end machine learning pipeline to estimate **housing prices in Bengaluru** using structured real-estate data.

The goal was not only to get good accuracy, but also to:

- clean and understand the dataset
- engineer meaningful features
- build a reproducible pipeline
- check multiple linear regression assumptions
- evaluate the model honestly

---

## Dataset

The dataset contains housing listings with details such as:

- Area type
- Location
- Total square feet
- Number of bedrooms (BHK)
- Number of bathrooms
- Balcony
- Price (target)

 **Target Variable:** `price` (in Lakhs)

---

## Project Workflow

### 1. Problem Understanding
Defined objective: predict house prices and understand feature impact.

### 2. Data Cleaning
- Removed duplicates
- Handled missing values
- Dropped columns with excessive nulls (like `society`)
- Converted `size` → numeric **bhk**
- Cleaned and standardized `total_sqft`

### 3. Feature Engineering
- Created `price_per_sqft`
- Grouped rare locations to avoid high-cardinality noise
- Removed unrealistic values (e.g., extremely small sqft per bhk)

### 4. Outlier Handling
Removed records with clearly unrealistic price-per-sqft outliers while preserving natural variation.

### 5. Train / Test Split
80% training — 20% testing.

### 6. Modeling with Pipelines
Used **ColumnTransformer + Pipeline** so preprocessing happens inside the model:

- One-hot encoding for categorical variables  
- Numeric features passed through  
- Trained **Multiple Linear Regression**

### 7. Evaluation & Assumption Checks
Evaluated using:

- R²
- Adjusted R²
- RMSE
- MAE
- Residual diagnostics
- VIF (multicollinearity)

---

## Model Performance

| Metric | Value |
|-------|------:|
| Train R² | ~0.79 |
| Test R² | ~0.77 |
| Adjusted R² | ~0.77 |
| RMSE | ~Rs.43 Lakhs |
| MAE | ~Rs.20 Lakhs|

Train and test scores are close → **good generalization**.

---

## Regression Assumptions

### Linearity
Residual plot shows no strong curvature.

### Homoscedasticity
Variance increases slightly at higher prices — common in real estate.

### Normality
Residuals have heavy tails, but acceptable for predictive modeling.

### Multicollinearity
All core numeric features had **VIF < 5**.

### Outliers
A few high-end properties remain — intentionally kept to reflect reality.

---

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-Learn
- Statsmodels
- Matplotlib / Seaborn

---

## Possible Future Improvements

- Log-transform target to reduce heteroscedasticity
- Try Ridge / Lasso regularization
- Better handling of luxury outliers
- Cross-validation comparison with other models

---

## What I Learned

- How to clean and structure messy real-estate data
- Importance of feature engineering and outlier control
- Building reproducible ML pipelines
- Evaluating assumptions of linear regression (not just accuracy)
- Writing interpretable results — not just code



