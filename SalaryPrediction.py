# -*- coding: utf-8 -*-
"""
@author: Ramesh Manickavel
Creation: 20-March-2023

Purpose: Leveraging Streamlit for salary prediction machine learning algorithm
High Level Overview:
    1.

"""
# Importing required libraries
import pandas as pd  # for data frames
import numpy as np  # for array reshaping
import seaborn as sn  # for plots
import matplotlib.pyplot as plt  # for plots
from sklearn.model_selection import train_test_split  # to do test & train data split
from sklearn.linear_model import LinearRegression  # for linear regression
from sklearn import metrics  # for the error metrics
from sklearn.metrics import mean_squared_error, r2_score  # for error calculation & the r sequared value
import statsmodels.api as sm  # Linear regression with statsmodels
import streamlit as st  # for the web app & deployment
from io import StringIO  # reading file
from math import sqrt

st.title("Machine Learning on Salary Prediction")

# Let us start with file upload
uploaded_file = st.file_uploader("Choose a file which contains dataset for salary prediction")
if uploaded_file is not None:
    # Reading the file content and loading it as data frame
    df = pd.read_csv(uploaded_file)
    st.subheader("Descriptive Analytics")
    st.write("Visualizing the dataset with line chart")
    st.line_chart(df, x="YearsExperience", y="Salary")
    st.write("Understanding the key elements of the data set")
    st.write("Shape (rows, columns): ", df.shape)
    # create DataFrame using data
    st.write("Data types of columns: ", pd.DataFrame(df.dtypes, columns=['Data Types']))
    st.write("Total null values in the dataset: ", pd.DataFrame(df.isnull().sum(), columns=['Number of rows with null values']))
    st.write("Descriptive statistics of the dataset: ", df.describe().T)
    st.subheader("Data Visualization")
    st.write("Below plot depict pairwise relationships between variables within our dataset")
    fig = sn.pairplot(df)
    st.pyplot(fig)
    st.write("Histogram for Salary")
    fig, ax = plt.subplots()
    ax.hist(df["Salary"], bins=5)
    st.pyplot(fig)
    st.write("Histogram for Years of Experience")
    fig, ax = plt.subplots()
    ax.hist(df["YearsExperience"], bins=5)
    st.pyplot(fig)
    st.subheader("Diagnostic Analytics")
    st.write("Correlation of variables using Heatmap")
    # Plotting using heatmap
    sn.heatmap(df.corr(), ax=ax, annot=True, cmap='coolwarm')
    st.write(fig)
    # Correlation using kendall method
    st.write("Correlation of variables using kendall method")
    st.write(df.corr(method='kendall'))
    # scatter plot
    st.write("Scatter plot to understand the relationship between variables")
    fig, ax = plt.subplots()
    df.plot.scatter(x='YearsExperience', y='Salary', ax=ax, s=100)
    st.write(fig)
    st.subheader("Predictive Analytics")
    # Splitting the dataset into train & test
    x = df['YearsExperience']
    y = df['Salary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # Since we have single feature we need to convert into a 2-D array with reshape (since model training requires 2-D array)
    # This step is NOT required if you have more than one features for prediction
    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)
    st.write("Visualizing the training dataset (Y)")
    st.write(y_train)
    st.write("Visualizing the training dataset (X)")
    st.write(x_train)
    # Linear Regression Model
    model = LinearRegression()

    response = model.fit(x_train, y_train)
    pred = model.predict(x_test)

    mse = mean_squared_error(y_test, pred)
    rmse = sqrt(mse)

    st.markdown(f"""
    Linear Regression model trained :
    	- MSE: {mse}
    	- RMSE: {rmse}
    """)
    st.success('Model trained successfully')
    # Getting the coefficient
    coeff = response.coef_
    # Getting the intercept
    intercept = response.intercept_
    st.write("The coefficient is: %d and the intercept is: %d" % (coeff, intercept))
    r2 = r2_score(y_test, pred)
    st.write("R-squared: ", r2)
    # Adjusted r squared is calculated by dividing the residual Mean Square Error (MSE) by total Mean Square Error & subtracted from 1.
    N = y_test.shape[0]  # number of records in the data set.
    p = 1  # number of independent variables (in our case it would be 'years of experience')
    x = (1 - r2)
    y = (N - 1) / (N - p - 1)
    adj_rsquared = (1 - (x * y))
    st.write("Adjusted R-Squared : ", adj_rsquared)

    st.write("Plotting predicted and actual responses (Ypred & Yactuals)")
    fig, ax = plt.subplots()
    plt.scatter(y_test, pred)
    st.write(fig)

    st.write("Model Fitting using statsmodels")
    # Here we are creating array of dependent (x) and indpendent variables again
    x = df['YearsExperience']
    y = df['Salary']
    x, y = np.array(x), np.array(y)
    # We need to column of ones to the inputs if you want statsmodels to calculate the intercept
    x = sm.add_constant(x)
    # Create a linear regression
    model = sm.OLS(y, x)
    # Fitting the model
    response = model.fit()
    # Printing the summary of fitment
    st.write("Summary of model fitment: ")
    st.write(response.summary())

else:
    st.write("No file is selected")

