#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:41:35 2019

@author: seth
"""

#Lasso

import pandas as pd
import scipy as sp
import numpy as np
from sklearn import model_selection as ms
from sklearn import linear_model as lm
from yellowbrick.regressor import ResidualsPlot

df = pd.read_csv('data/merged_data.csv')
df.columns = [x.lower() for x in df.columns]
df.drop(set(df[df["hivdiagnoses"] == 0].index), axis=0, inplace=True)
X = df[['hivdiagnoses', 'hivprevalence', 'plhiv', 'population']]
y = np.log(df['hivincidence'])

X_train, X_test, y_train, y_test = ms.train_test_split(
        X, y, test_size =.25)

model = lm.Lasso(alpha = .5,fit_intercept=True,tol=10)
visualizer = ResidualsPlot(model)
model.fit(X_train,y_train)


pred = model.predict(X_train)
r = sp.stats.linregress(pred,y_train)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()
