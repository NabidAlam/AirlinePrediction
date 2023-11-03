# Customer Satisfaction in Airline

This Kaggle project involves a dataset with information on 129,880 customers, and it encompasses various data points such as class, flight distance, and inflight entertainment. The primary objective of this project is to develop a predictive model to determine whether a customer is likely to be satisfied with their flight experience. By leveraging machine learning and data analysis techniques, the project aims to provide valuable insights for airlines to enhance customer satisfaction and overall service quality.

This Python project seems to be focused on analyzing airline customer satisfaction based on various attributes and then building machine learning models for predictive analysis. Here's a brief overview of what has been done:

1. Data Preprocessing:
   - The project begins by importing necessary libraries, including pandas, seaborn, numpy, and matplotlib.
   - The dataset is loaded from a CSV file named 'Invistico_Airline.csv'.
   - The project analyzes and preprocesses the data, including counting the number of unique values for object data types, and visualizing categorical columns.

2. Exploratory Data Analysis (EDA):
   - EDA is conducted to explore the dataset further.
   - The names of all columns with the 'object' data type are retrieved, excluding 'satisfaction'.
   - Countplots are created for the top 6 values of each categorical variable using Seaborn.
   - Boxplots are generated for numerical variables.
   - Various other data visualizations are created to understand the data better.

3. Label Encoding:
   - Label encoding is applied to object data types to convert them into numerical form for machine learning.

4. Decision Tree Classifier:
   - A Decision Tree classifier is implemented with hyperparameter tuning using grid search.
   - The model's accuracy is calculated, and various classification metrics are evaluated, including F1 Score, Precision, Recall, Jaccard Score, and Log Loss.
   - Feature importance is visualized using bar plots.
   - SHAP (SHapley Additive exPlanations) values are used to explain the model's predictions.
   - A confusion matrix and ROC curve are plotted for model evaluation.

5. Random Forest Classifier:
   - A Random Forest classifier is implemented with hyperparameter tuning using grid search.
   - The model's accuracy is calculated, and various classification metrics are evaluated, similar to the Decision Tree Classifier.
   - Feature importance, SHAP values, and model evaluation metrics are also analyzed for the Random Forest model.

6. Data Preprocessing Part 2:
   - Missing values are checked and handled by removing rows with missing data in the 'Arrival Delay in Minutes' column.

The project demonstrates a comprehensive analysis of airline customer satisfaction data and showcases the use of machine learning models to predict customer satisfaction. It also uses various data visualization techniques for better insights into the dataset.
