# Examining Hospital Readmissions

Hospital Readmissions are challenging, since these increase hospital's costs and represent underperforming healthcare service. Therefore, being able to predict such readmissions can lead to improved patient care and substantial cost savings.

As such, in this Machine Learning project, we used use an hospital's dataset, to predict based on historical data, if someone will be readmitted. In this project, we produced two classification models: 

* a **Binary Classification**, which predicts if a patient will be readmitted to the hospital within 30 days of being discharged.

* a **Multiclass Classification**, which predicts if a patient will not be readmitted, if the patient will be readmitted within 30 days of the encounter or if the patient will be readmitted 30 days after the encounter.

We created a separate Jupyter Notebook for each classification task, one for binary classification and another for multiclass classification, allowing for a more organized and focused analysis of each model type. Both Jupyter Notebooks can be found in this repository.

**Kaggle Competition:** https://www.kaggle.com/competitions/predicting-hospital-readmissions/leaderboard

## Methodology

### Data Preparation

The first step involved exploring the dataset to identify inconsistencies, outliers, and missing values. These issues were addressed using appropriate data cleaning techniques based on the nature and context of each case.

New features were engineered from the original variables to enrich the dataset. Numerical features were standardized using StandardScaler, while categorical variables were encoded using One-Hot Encoding.  Even though there is a notebook for each classification problem, the data transformations performed by us were the same on both notebooks, up until **feature engineering**.

Since the dataset originally contained 28 variables, it was essential to apply feature selection techniques to reduce dimensionality and enhance model performance. To achieve this, we employed the following methods:

* Spearman Correlation

* Recursive Feature Elimination (RFE)

* ANOVA

* Mutual Information

* Lasso Regression

* Ridge Regression

* Chi-squared Test


To address class imbalance in both target variables, we applied **SMOTE** (Synthetic Minority Over-sampling Technique) and the using **Weighted Values**, which follows the class weighting approach, which adjusts the model’s class weight parameter, to assign greater importance to underrepresented classes. The goal was to evaluate and compare each model’s performance using each of these imbalance-handling techniques.


### Modeling

The metric that was used to assess the model's performance was the **F1 Score**. 

On an initial round, the following models were implemented to predict both target variables: **Logistic Regression**, **Gaussian Naïve Bayes**, **Artificial Neural Networks**, **K-Nearest Neighbors (KNN)**, **Random Forest**, **Gradient Boosting**, **Support Vector Machines (SVM)**, **Decision Trees**, **Bagging Classifier**, and **AdaBoost**.

On the final round, we applied **Grid Search** to the best performing models, which were: **Logistic Regression**, **Random Forest**, **Artificial Neural Networks**,**Gradient Boosting** and a **Stacking Classifier** with **Logistic Regression** and **Random Forest**.

The best solution, for the **Binary Classification** was **Random Forest**, with the Weighted Values approach. For the **Multiclass Classification** was **Artificial Neural Networks**, with the Weighted Values approach. 


