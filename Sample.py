import streamlit as st
from datetime import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

import numpy as np

sns.set_style("darkgrid")

st.title("Advertisement and Sales Data")

st.markdown("""
	The data set contains information about money spent on advertisement and their generated sales. Money
	was spent on TV, radio and newspaper ads.
	## Problem Statement
	Sales (in thousands of units) for a particular product as a function of advertising budgets (in thousands of
	dollars) for TV, radio, and newspaper media. Suppose that in our role as statistical consultants we are
	asked to suggest.
	Here are a few important questions that you might seek to address:
	- Is there a relationship between advertising budget and sales?
	- How strong is the relationship between the advertising budget and sales?
	- Which media contribute to sales?
	- How accurately can we estimate the effect of each medium on sales?
	- How accurately can we predict future sales?
	- Is the relationship linear?
	We want to find a function that given input budgets for TV, radio and newspaper predicts the output sales
	and visualize the relationship between the features and the response using scatter plots.
	The objective is to use linear regression to understand how advertisement spending impacts sales.

	### Data Description
	TV
	Radio
	Newspaper
	Sales
""")
st.sidebar.title("Operations on the Dataset")

# st.subheader("Checkbox")
w1 = st.sidebar.checkbox("show table", False)
plot = st.sidebar.checkbox("show plots", False)
plothist = st.sidebar.checkbox("show hist plots", False)
trainmodel = st.sidebar.checkbox("Train model", False)
dokfold = st.sidebar.checkbox("DO KFold", False)
distView = st.sidebar.checkbox("Dist View", False)
_3dplot = st.sidebar.checkbox("3D plots", False)
linechart = st.sidebar.checkbox("Linechart", False)


# st.write(w1)


@st.cache_data
def read_data():
    return pd.read_csv("C:\Ramesh\Scripts\General\Advertising.csv")[["TV", "Radio", "Newspaper", "Sales"]]


df = read_data()

# st.write(df)
if w1:
    st.dataframe(df, width=2000, height=500)
if linechart:
    st.subheader("Line chart")
    st.line_chart(df)
if plothist:
    st.subheader("Distributions of each columns")
    options = ("TV", "Radio", "Newspaper", "Sales")
    sel_cols = st.selectbox("select columns", options, 1)
    st.write(sel_cols)
    # f=plt.figure()
    fig = go.Histogram(x=df[sel_cols], nbinsx=50)
    st.plotly_chart([fig])

#    plt.hist(df[sel_cols])
#    plt.xlabel(sel_cols)
#    plt.ylabel("sales")
#    plt.title(f"{sel_cols} vs Sales")
# plt.show()
#    st.plotly_chart(f)

if plot:
    st.subheader("correlation between sales and Ad compaigns")
    options = ("TV", "Radio", "Newspaper", "Sales")
    w7 = st.selectbox("Ad medium", options, 1)
    st.write(w7)
    f = plt.figure()
    plt.scatter(df[w7], df["Sales"])
    plt.xlabel(w7)
    plt.ylabel("Sales")
    plt.title(f"{w7} vs Sales")
    # plt.show()
    st.plotly_chart(f)

if distView:
    st.subheader("Combined distribution viewer")
    # Add histogram data

    # Group data together
    hist_data = [df["TV"].values, df["Radio"].values, df["Newspaper"].values]

    group_labels = ["TV", "Radio", "Newspaper"]

    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])

    # Plot!
    st.plotly_chart(fig)

if _3dplot:
    options = st.multiselect(
        'Enter columns to plot', ('TV', 'Radio'), ('TV', 'Radio', 'Newspaper', 'Sales'))
    st.write('You selected:', options)
    st.subheader("TV & Radio vs Sales")
    hist_data = [df["TV"].values, df["Radio"].values, df["Newspaper"].values]

    # x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 400).transpose()
    trace1 = go.Scatter3d(
        x=hist_data[0],
        y=hist_data[1],
        z=df["Sales"].values,
        mode="markers",
        marker=dict(
            size=8,
            # color=df['Sales'],  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            #        opacity=0.,
        ),
    )

    data = [trace1]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=data, layout=layout)
    st.write(fig)

# trainmodel= st.checkbox("Train model", False)

if trainmodel:
    st.header("Modeling")
    y = df.Sales
    X = df[["TV", "Radio", "Newspaper"]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lrgr = LinearRegression()
    lrgr.fit(X_train, y_train)
    pred = lrgr.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    rmse = sqrt(mse)

    st.markdown(f"""
	Linear Regression model trained :
		- MSE:{mse}
		- RMSE:{rmse}
	""")
    st.success('Model trained successfully')

if dokfold:
    st.subheader("KFOLD Random sampling Evalution")
    st.empty()
    my_bar = st.progress(0)

    from sklearn.model_selection import KFold

    X = df.values[:, -1].reshape(-1, 1)
    y = df.values[:, -1]
    # st.progress()
    kf = KFold(n_splits=10)
    # X=X.reshape(-1,1)
    mse_list = []
    rmse_list = []
    r2_list = []
    idx = 1
    fig = plt.figure()
    i = 0
    for train_index, test_index in kf.split(X):
        #	st.progress()
        my_bar.progress(idx * 10)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lrgr = LinearRegression()
        lrgr.fit(X_train, y_train)
        pred = lrgr.predict(X_test)

        mse = mean_squared_error(y_test, pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, pred)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        plt.plot(pred, label=f"dataset-{idx}")
        idx += 1
    plt.legend()
    plt.xlabel("Data points")
    plt.ylabel("Predictions")
    plt.show()
    st.plotly_chart(fig)

    res = pd.DataFrame(columns=["MSE", "RMSE", "r2_SCORE"])
    res["MSE"] = mse_list
    res["RMSE"] = rmse_list
    res["r2_SCORE"] = r2_list

    st.write(res)
    st.balloons()
