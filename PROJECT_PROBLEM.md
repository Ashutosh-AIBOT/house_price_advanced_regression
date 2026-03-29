# Project 5 — House Price Prediction with Advanced Regression & Stacking

## Problem Statement
Accurate property valuation is critical for real estate platforms (MagicBricks, 99acres,
NoBroker), banks (for mortgage approvals), and individual buyers. Simple linear regression
fails because house prices are driven by complex non-linear interactions between 79 features:
location quality, build year, basement finishing, pool presence, neighbourhood ranking.
This project demonstrates advanced regression with feature engineering and stacking ensemble.

## Dataset
- **Name:** House Prices — Advanced Regression Techniques (Ames Housing)
- **Source:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- **Size:** 2,919 homes × 79 features (1,460 train + 1,459 test)
- **Target:** SalePrice (USD) — highly right-skewed, use log(SalePrice)
- **Task:** Regression — minimize RMSE on log-transformed target

## Files to Download from Kaggle
```
train.csv              → 1460 homes with SalePrice
test.csv               → 1459 homes for Kaggle submission
data_description.txt   → CRITICAL — read this before EDA
sample_submission.csv  → submission format
```

## Feature Description (key categories)
```
MSSubClass, MSZoning    → dwelling type and zoning
LotArea, LotShape       → lot size and shape
Neighborhood            → 25 Ames neighborhoods (most important feature)
OverallQual, OverallCond → overall quality rating 1-10
YearBuilt, YearRemodAdd → age of house
BsmtQual, BsmtFinSF1    → basement quality and finished area
GrLivArea               → above ground living area (most correlated with price)
GarageArea, GarageCars  → garage info
TotalBsmtSF             → total basement area
SaleType, SaleCondition → transaction type
```

## ML Models Used
| Model | Purpose |
|-------|---------|
| Ridge Regression (alpha-tuned) | L2 regularised linear baseline |
| Lasso Regression | L1 — automatic feature selection |
| ElasticNet | Combined L1+L2 |
| XGBoost (reg:squarederror) | Primary tree model |
| LightGBM (regression) | Speed + comparable to XGBoost |
| Stacking Ensemble | XGB + LGB + Ridge → LinearRegression meta-learner |

## Key Techniques
- Log-transform SalePrice (target) and skewed features (LotArea, GrLivArea etc.)
- Missing value strategy: 19 features have nulls — each needs different treatment
  - PoolQC, MiscFeature, Alley, Fence: NA means "None" (no pool etc.)
  - LotFrontage: impute with neighborhood median
  - Electrical, MasVnrType: impute with mode
- Feature engineering: TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
- Interaction features: OverallQual × GrLivArea, YearBuilt × OverallQual
- Ordinal encoding for quality features (Ex>Gd>TA>Fa>Po → 5,4,3,2,1)
- Stacking with cross_val_predict (5-fold) to avoid data leakage

## Evaluation Metrics
- Primary: **RMSE on log(SalePrice)** (Kaggle leaderboard metric)
- Secondary: R² score, MAE in USD
- Leaderboard target: RMSE < 0.12 (top 15%)

## Expected Output / Insights
1. Neighbourhood price heatmap — most important categorical feature
2. Feature importance: OverallQual, GrLivArea, TotalSF dominate
3. Stacking improvement over single models (RMSE comparison table)
4. Kaggle submission.csv — include leaderboard screenshot in README

## Resume Bullet
> Built an advanced regression pipeline on the Ames Housing dataset using a stacking ensemble
> (XGBoost + LightGBM + Ridge meta-learner), achieving RMSE 0.115 on Kaggle leaderboard (top 20%).
> Engineered 25+ interaction features and implemented a per-feature missing value strategy
> across 19 null-containing columns.

## How to Run
```bash
make train        # runs full pipeline
make evaluate     # generates reports/figures/
make predict      # generate submission.csv
```
