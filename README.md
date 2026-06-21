# Amazon Sales: Price Prediction & Rating Classification

Supervised machine learning on the Amazon Sales Dataset (1,465 product listings, 9 categories) to address two business questions:

1. **Regression** — what will a product's discounted selling price be, given its attributes?
2. **Classification** — will a product earn a high customer rating (≥ 4.0)?

## Results

| Task | Best model | Key metric |
|---|---|---|
| Price prediction | Random Forest Regressor | R² = 0.987, MAE = ₹123.9 |
| High-rating classification | Random Forest Classifier | AUC-ROC = 0.790, F1 = 0.856 |

Five models were benchmarked per task (Linear/Logistic Regression, SVM/SVR, Decision Tree, Random Forest, KNN), with log-transform feature engineering, class-imbalance handling (3:1 imbalance on the classification target), and a stratified 80/20 train/test split.

**Headline finding:** for price, the list price and discount depth explain ~95% of the variance — pricing on Amazon is close to mechanical. For *rating*, the strongest signal isn't price at all — it's review count, suggesting customer satisfaction is better proxied by product popularity/longevity on the platform than by price positioning.

## Contents

- `amazon_price_rating_models.ipynb` — full analysis: EDA, feature engineering, 10 trained models, evaluation, feature-importance plots.

## Team

Ibtisam Kayani · Faseeh Ul Hassan · Abdullah Mehmood · Ahsan Ul Haq · Salman Jamal
Machine Learning for Business Analytics, FAST-NUCES

## Stack

Python · pandas · scikit-learn · matplotlib/seaborn
