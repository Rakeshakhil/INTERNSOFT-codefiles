# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:09:29 2020

@author: rakei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



adv=pd.read_csv('advertising.csv')
adv.head()
fig,axs=plt.subplots(1,3,sharey=True)
adv.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
adv.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
adv.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])

fea=['TV']
x=adv[fea]
y=adv.Sales

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)

res=6.9748214882298925+0.05546477*50
print(res)

x_new=pd.DataFrame({'TV':[adv.TV.min(),adv.TV.max()]})
x_new.head()

pred=lr.predict(x_new)
pred
adv.plot(kind='scatter', x='TV', y='Sales')
plt.plot(x_new, pred, c='red', linewidth=1)
import statsmodels.formula.api as smf
lr = smf.ols(formula='Sales ~ TV', data=adv).fit()
lr.conf_int()

lr.pvalues

lr.rsquared



fea=['TV','Radio','Newspaper']
x=adv[fea]
y=adv.Sales
lr=LinearRegression()
lr.fit(x,y)
print(lr.intercept_)
print(lr.coef_)

lr=smf.ols(formula='Sales~ TV+Radio+Newspaper',data=adv).fit()
lr.conf_int()
lr.summary()

lr = smf.ols(formula='Sales ~ TV + Radio', data=adv).fit()
lr.rsquared

lr = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=adv).fit()
lr.rsquared


