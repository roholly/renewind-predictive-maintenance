# ReneWind Predictive Maintenance

**Domain:** Renewable Energy · Industrial ML  
**Tools:** Python, scikit-learn, XGBoost, imbalanced-learn  

---

## Why This Problem Matters

There is something about wind turbines that I find genuinely moving. Growing up in Oklahoma, I was the kid on road trips pointing out the window at every windmill we passed. That hasn't changed. I have followed the wind energy sector for years, and around 2014 drove to the Canadian Valley Technology Center in Yukon, Oklahoma with my father specifically to learn more about how turbines work and how technicians are trained to maintain them. The number one mechanical failure issue at the time was the gearbox.

Becoming a wind turbine technician is not a desk job. Technicians train like firefighters. They climb towers that can exceed 300 feet, work in extreme weather, and perform physically demanding repairs on equipment that costs millions of dollars to replace. Every unnecessary climb is a risk. Every missed failure is a replacement rather than a repair, and a much bigger bill.

This is the problem predictive maintenance is designed to solve: catch failures before they happen, so the right people go up the tower at the right time for the right reason.

---

## The Problem

ReneWind provided sensor data from 20,000 turbines (40 features, ciphered for confidentiality) and asked for a model that identifies failures before they occur.

The cost structure is asymmetric and specific. A missed failure means full replacement, not repair. A false positive means an unnecessary inspection. Replacement costs dwarf repair costs, and both dwarf inspection costs. The model needs to reflect that hierarchy.

The catch: only 5.5% of the training data represents actual failures. A model that predicts no failure every time is 94.5% accurate and completely useless.

---

## The Approach

**Metric:** Recall, because a missed failure costs far more than a false inspection.

**Class imbalance:** Compared three strategies (original data, SMOTE oversampling, and random undersampling) across seven model types (Logistic Regression, Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost). That's 21 configurations before tuning.

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

The selected model catches 89% of real failures. Precision is 24%, meaning inspections will sometimes find nothing. Given the cost structure, that tradeoff is deliberate and documented.

---

## What This Demonstrates

This project connected two things I find genuinely interesting: renewable energy and the kind of infrastructure monitoring and alert-response work I spent years doing at the IRS. The underlying question is the same one I worked with for years in a different context. What signals in the data indicate something is about to go wrong, and how do you surface that signal reliably enough to act on it before the cost becomes much larger?

The sensor data here is obfuscated, so there's no way to know which features map to which physical components. But the modeling discipline is the same regardless: cost-aware metric selection, honest handling of class imbalance, and documenting the tradeoffs rather than just reporting the winning number.
