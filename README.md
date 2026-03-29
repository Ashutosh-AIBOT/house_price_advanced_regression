# House Price Advanced Regression

Real estate valuation modeling with ensemble regressors.

## Recommended GitHub Repo Name

advanced-house-price-regression-intelligence

## Domain

PropTech and valuation analytics

## Objective

Predict property sale price using engineered features and stacked models.

## Business Value

Improves pricing accuracy for brokers, investors, and lending workflows.

## Dataset

Verified: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Run Locally

conda run -n ml-env streamlit run app.py

## Dashboard Features

- Executive overview and business framing
- Prediction Center (single input + batch CSV)
- Model ranking leaderboard
- Model registry (all discovered model artifacts)
- Charts gallery (auto-load from charts/)
- Notebook traceability panel

## Notebook Path Setup

Inside notebooks:

from path_setup import BASE, RAW, PROCESSED, MODELS, CHARTS, load_raw_csv

Project path helpers:

- path_utils.py
- notebooks/path_setup.py
## Deploy on Render

This repo includes render.yaml. Connect repo in Render and deploy.

Start command used:

streamlit run app.py --server.port $PORT --server.address 0.0.0.0

## Git LFS

Large assets are tracked with Git LFS (models, charts, data artifacts).
