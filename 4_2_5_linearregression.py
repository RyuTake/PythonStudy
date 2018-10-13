# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:48:48 2018

@author: rtake
"""

import matplotlib.pyplot as plt
import numpy as np

x1 = np.random.rand(100,1)
x1 = x1 * 4 -2

x2 = np.random.rand(100,1)
x2 = x2 * 4 -2

y = 3 * x1 - 2 * x2 +1

plt.subplot(1,2,1)
plt.scatter(x1, y, marker='+')
plt.xlabel('x1')
plt.ylabel('y')

plt.subplot(1,2,2)
plt.scatter(x2, y, marker='+')
plt.xlabel('x2')
plt.ylabel('y')

plt.tight_layout()
plt.show()

from sklearn import linear_model

x1_x2 = np.c_[x1, x2] #2つの配列をひとまとめに [[x1_1, x2_1],[x1_2, x2_2], ... [x1_100, x2_100]]

model = linear_model.LinearRegression()
model.fit(x1_x2, y)

y_ = model.predict(x1_x2) # 回帰式で予測

plt.subplot(1,2,1)
plt.scatter(x1, y, marker='+')
plt.scatter(x1, y_, marker='o')
plt.xlabel('x1')
plt.ylabel('y')

plt.subplot(1,2,2)
plt.scatter(x2, y, marker='+')
plt.scatter(x2, y_, marker='o')
plt.xlabel('x1')
plt.ylabel('y')

plt.tight_layout()
plt.show()

print(model.coef_)
print(model.intercept_)

print(model.score(x1_x2, y))