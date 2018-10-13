# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:24:51 2018

@author: rtake
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(100, 1) # 100個の乱数を作る。範囲は0～1
x = x * 4 -2 # xを2～2に変換する

y = 3* x -2 #一次関数
y += np.random.rand(100, 1)

plt.scatter(x,y, marker='+')    #プロット
plt.show()  #表示

#%%

from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(x,y)

print(model.coef_)
print(model.intercept_)
r2 = model.score(x,y)
print(r2)

#%%二次関数の場合

x = np.random.rand(100,1)
x = x * 4 -2
y = 3 * x ** 2 -2 
y += np.random.rand(100, 1)

model = linear_model.LinearRegression()
model.fit(x**2,y)

plt.scatter(x,y, marker='+')    #プロット
plt.scatter(x,y, marker='o')    #プロット
plt.show()  #表示
print(model.coef_)
print(model.intercept_)
r2 = model.score(x**2,y)
print(r2)
