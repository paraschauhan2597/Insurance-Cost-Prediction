# 🏥 Health Insurance Premium Predictor: Risk Assessment & Pricing Model

## 📊 Executive Summary

This project develops a complete, end-to-end Machine Learning pipeline designed to predict health insurance premiums based on individual patient demographics and health histories. By utilizing predictive modeling, this tool assists in financial risk assessment, allowing insurance providers to optimize underwriting processes, price policies accurately, and maintain healthy profit margins while remaining competitive.

## 🎯 Business Problem

In the health insurance and NBFC sectors, inaccurate premium pricing leads to either adverse selection (underpricing high-risk profiles) or customer churn (overpricing low-risk profiles). The objective of this model is to move beyond generalized pricing tables and leverage non-linear algorithmic predictions to assign a precise financial value to an individual's unique health risk profile.

## 🛠️ Methodology & Technical Approach

### 1. Exploratory Data Analysis (EDA)

- Conducted rigorous data validation, missing value imputation, and univariate/bivariate analysis.
- Visualized target distributions to understand the spread and variance of premium pricing.

### 2. Statistical Hypothesis Testing

To ensure mathematical rigor before modeling, statistical tests were conducted to prove the significance of key health factors:

- **Independent T-Test:** Confirmed a statistically significant difference ($p < 0.05$) in premium costs for individuals with chronic diseases versus those without.
- **One-Way ANOVA:** Confirmed a statistically significant variance ($p < 0.05$) in premiums based on a patient's history of major surgeries.

### 3. Predictive Modeling (Machine Learning)

Multiple regression algorithms were trained and evaluated to capture both linear and non-linear interactions between health factors:

- **Baseline Model:** Linear Regression ($R^2$: 0.71)
- **Gradient Boosting Regressor:** ($R^2$: 0.85)
- **Champion Model:** Random Forest Regressor ($R^2$: 0.88, MAE: ~1,034)
- _The Random Forest model successfully reduced the average prediction error by over 60% compared to the baseline, providing highly reliable estimates for premium underwriting._

## 💡 Key Business Insights (Feature Importance)

Extracted feature importances from the Random Forest model revealed the primary drivers of premium costs:

1. **Age:** The most dominant factor in risk pricing.
2. **Weight:** A critical secondary factor compounding overall health risk.
3. **History of Transplants:** The highest-impact specific medical condition influencing the algorithmic pricing tier.

## 🚀 Interactive Web Application

The predictive model was deployed into a fully functional, interactive web application using **Streamlit**. This allows non-technical stakeholders or customer-facing agents to input a prospective client's health profile and receive a real-time, mathematically backed premium estimate.

### Tech Stack Used:

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Statistical Testing:** SciPy
- **Machine Learning:** Scikit-Learn
- **Web Deployment:** Streamlit, Joblib
