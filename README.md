# ReneWind Predictive Maintenance

**Domain:** Renewable Energy · Industrial ML  
**Tools:** Python, scikit-learn, XGBoost, imbalanced-learn  

---

## The Problem

Wind turbine generator failures are expensive — replacement costs dwarf repair costs, and both dwarf the cost of inspection. ReneWind provided sensor data from 20,000 turbines (40 features, ciphered) and asked for a model that catches failures before they happen.

The catch: only 5.5% of the training data represents actual failures. A model that predicts "no failure" every time is 94.5% accurate and completely useless.

---

## The Approach

**Metric:** Recall — because a missed failure (false negative) costs far more than a false inspection (false positive).

**Class imbalance:** Compared three strategies — original data, SMOTE oversampling, and random undersampling — across seven model types (Logistic Regression, Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost). That's 21 configurations before tuning.

**Tuning:** Top four performers by validation recall were hyperparameter-tuned using RandomizedSearchCV. XGBoost with undersampling consistently led.

**Production:** Final model wrapped in a sklearn Pipeline with median imputation for deployment readiness.

---

## Results

| Model | Recall (Test) |
|---|---|
| XGBoost + Undersampling (final) | **0.89** |
| Random Forest + Undersampling | 0.89 |
| GBM + Oversampling | 0.88 |
| XGBoost + Oversampling | 0.87 |

The selected model catches 89% of real failures. Precision is 24% — meaning inspections will sometimes find nothing. Given the cost structure, that tradeoff is deliberate and documented.

---

## Key Finding

The top predictive features (V36, V15, V14, V26, V18) align with the variables showing strongest correlation with failure in EDA — but correlation alone undersells their combined predictive value in the ensemble model.

---

## What This Demonstrates

- Cost-aware metric selection and model evaluation
- Class imbalance handling (SMOTE, undersampling) with honest comparison
- Systematic model selection across algorithm families
- Production pipeline design with no data leakage
- Willingness to document tradeoffs rather than just report the winning number
