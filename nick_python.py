import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import random


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

font = {'size'   : 12}
matplotlib.rc('font', **font)
sns.set(style="ticks")
def read_data():
    df = pd.read_csv("data/merged_data.csv")
    df.fillna(0, inplace = True)
    df.drop(set(df[df["HIVdiagnoses"] == 0].index), axis=0, inplace=True)
    cols = ["HIVdiagnoses", "HIVprevalence", "PLHIV", "Population"]
    X = pd.DataFrame(index=df["ADULTMEN"].index, columns=cols)
    for col in cols:
        X[col] = df[col]
    y = np.array(df["HIVincidence"])
    return df, X, y

def plot_alpha_vals(X_train, X_test, y_train, y_test, alphas):
    r_values = []
    mse_test = []
    for a in alphas:
        model = linear_model.Ridge(alpha = a)
        fitted = model.fit(X_train, y_train)
        results = fitted.predict(X_test)
        stats_of_result = stats.linregress(results, y_test)
        #print("R value is {}".format(stats_of_result[2]))
        r_values.append(stats_of_result[2]) #[2] is r-value
        mse_test.append(metrics.mean_squared_error(y_test, results))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(alphas, mse_test)

def plot_residuals(X_train, X_test, y_train, y_test):
    model_log = sm.OLS(y_train, X_train)
    results_log = model_log.fit_regularized(method='elastic_net', alpha=0.5, L1_wt=0.0)
    hiv_resids_log = results_log.outlier_test()['student_resid']
    cols = list(X_train.columns)
    fig, axs = plt.subplots(len(X_train.columns), 1, figsize=(8,20))
    for idx, ax in enumerate(axs.flatten()):
        ax.scatter(X_train[cols[idx]], hiv_resids_log)
        ax.hlines(0,
              X_train[cols[idx]].min(), 
              X_train[cols[idx]].max(), 
              'k', linestyle='dashed')
        ax.set_xlabel(cols[idx])
        ax.set_ylabel('studentized residuals');

if __name__ == "__main__":
    hiv_data, X, y = read_data()
    y = np.log(y)
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(X, y, random_state=13)
    model = linear_model.Ridge(alpha = 0.5)
    fitted = model.fit(X_train_full, y_train_full)
    results = fitted.predict(X_holdout)
    l_results = stats.linregress(results, y_holdout) #[2] is r-value