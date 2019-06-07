#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:41:35 2019

@author: seth
"""

#Lasso

import pandas as pd
import scipy as sp
from sklearn import model_selection as ms
from sklearn import linear_model as lm

df = pd.read_csv('data/merged_data.csv')
df.columns = map(str.lower,df.columns)
df.fillna
X = df[['household_income','poverty_rate','unemployment_rate','adultmen','msm12mth',
        'hivdiagnoses','hivprevalence','drugdep']]
y = df['hivincidence']

X_train, X_test, y_train, y_test = ms.train_test_split(
        X, y, test_size =.25,random_state=13)



model = lm.Lasso(fit_intercept=True)
model.fit(X_train,y_train)
print(model.coef_)
print('\n\n\n\n\n')
print(model.intercept_)

pred = model.predict(X_train)


r = sp.stats.linregress(pred,y_train)