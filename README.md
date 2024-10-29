## Customer-Churn-Prediction

# Overview

This project aims to analyze customer churn for a telecommunications company and develop predictive models to identify customers at risk of leaving. The project involves data preprocessing, exploratory data analysis (EDA), model training, evaluation, and feature importance analysis.

# Customer Churn Meaning
Customer churn refers to the loss of clients or customers. In this project, we aim to predict which customers are likely to churn based on their usage patterns, demographics, and service plans. By identifying these customers, the company can take proactive measures to retain them.

# Dataset Description
The dataset used for this project contains information about customers, including features such as:

Account length
Area code
Call plans (International plan, Voice mail plan)
Various metrics (e.g., total day minutes, total night calls, etc.)
Churn label (True/False)
The dataset was split into training and test sets to validate the performance of the predictive models.

# Data Preprocessing
The preprocessing steps included:

Loading the dataset and checking for missing values.
Encoding categorical variables using one-hot encoding.
Scaling numerical features using StandardScaler for normalization.
Splitting the data into features (X) and target (y).

# Exploratory Data Analysis (EDA)
Exploratory data analysis was conducted to understand churn patterns and relationships among features:

Visualization of churn rates across different features (e.g., call plans, area codes).
Statistical analysis to identify significant trends in the data.

# Model Selection
Five different machine learning models were implemented:

Logistic Regression
Random Forest
Decision Tree
Support Vector Machine (SVM)
XGBoost
Each model was trained using the training dataset, and performance metrics were collected for comparison.

# Model Evaluation
The models were evaluated using:

# Confusion Matrix
Classification Report (precision, recall, F1-score)
ROC AUC Score
The Random Forest and XGBoost models performed the best, with Random Forest showing the highest accuracy and AUC score.

# Feature Importance Analysis
Feature importance was investigated for the top-performing models (Random Forest and XGBoost) to understand which features were the most influential in predicting customer churn. This analysis provides valuable insights for improving customer retention strategies.

# Final Model Testing
The selected models were tested on a reserved test dataset to validate their performance. The results were as follows:

# Random Forest:
Accuracy: 94%
ROC AUC Score: 0.94
XGBoost:
Accuracy: 93%
ROC AUC Score: 0.88

# Recommendations
Based on the insights gathered from the analysis, the following recommendations can be made to reduce churn rates:

Targeted Retention Campaigns: Focus marketing efforts on customers identified as high-risk for churn based on model predictions. Tailor retention strategies that address their specific concerns.

Service Improvement: Enhance the quality of services and features that are significant predictors of churn (e.g., customer service quality, call plan features).

Customer Feedback: Implement regular feedback mechanisms to understand customer satisfaction and address issues proactively.

Personalized Offers: Use the model's insights to create personalized offers or discounts for customers who may be considering leaving.

Monitoring and Updates: Continuously monitor churn rates and model performance. Regularly update the model with new data to ensure accuracy over time.

# Conclusion
The project successfully identified key drivers of customer churn and developed robust predictive models. The insights gained from the feature importance analysis can guide the company in implementing targeted customer retention strategies.

# Technologies Used

Python
Pandas
NumPy
Matplotlib / Seaborn (for visualizations)
Scikit-learn (for machine learning)
