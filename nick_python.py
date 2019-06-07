import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
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

if __name__ == "__main__":
    hiv_data, X, y = read_data()
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(X, y)
    model = linear_model.Ridge(alpha = 0.5)
    fitted = model.fit(X_train_full, y_train_full)
    results = fitted.predict(X_holdout)
    print(stats.linregress(results, y_holdout))