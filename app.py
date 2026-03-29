import os

from dashboard_core import ProjectConfig, run_project_app


CONFIG = ProjectConfig(
    key="Project 05",
    title="House Price Advanced Regression",
    subtitle="Real estate valuation modeling with ensemble regressors",
    icon="🏠",
    domain="PropTech and valuation analytics",
    objective="Predict property sale price using engineered features and stacked models.",
    business_value="Improves pricing accuracy for brokers, investors, and lending workflows.",
    prediction_label="Price Estimation",
    highlights=[
        "Implements advanced feature engineering and ensemble-based valuation methods.",
        "Supports rapid what-if property analysis through interactive prediction flow.",
        "Provides benchmark transparency across linear, tree, and stacking approaches.",
        "Connects residual and metrics analysis to valuation confidence discussions.",
        "Shows strong end-to-end delivery from experimentation to deployable interface.",
    ],
)


run_project_app(CONFIG, os.path.dirname(__file__))
