# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:00:08 2018

@author: rtake
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(100, 1)
x = x * 2 - 1

y = 4 * x**3 -3 * x **2 + 2 * x - 1
y += np.random.randn(100, 1) # 標準正規分布（平均0, 標準偏差1)の乱数を加える
