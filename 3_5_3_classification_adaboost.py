# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:35:29 2018

@author: rtake
"""

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

# digitsデータをロード
digits = datasets.load_digits()

# 画像を２行５列に表示
for label, img in zip(digits.target[:10], digits.images[:10]):
    plt.subplot(2,5,label+1)                                             # 行列配置で表示をする
    plt.axis('off')                                                      # 軸の非表示
    plt.imshow(img, cmap = plt.cm.gray_r, interpolation = 'nearest')     # 画像表示をする
    plt.title('Digit:[0]'.format(label))                                 # 画像のラベルを「Digit:○」の形で表示する
    
plt.show()

#%% 分類器を作る

#3と8のデータ一を求める
flag_3_8 = (digits.target == 3) + (digits.target == 8)

#3と8のデータ取得
images = digits.images[flag_3_8]
labels = digits.target[flag_3_8]

#行列の形を一次元に変更
images = images.reshape(images.shape[0], -1)

#%% 分類器を生成する
from sklearn import tree
from sklearn import ensemble

n_samples = len(flag_3_8[flag_3_8])             #サンプル数 配列の長さからサンプル数を読み出す
train_size = int (n_samples * 3 / 5)            #サンプル数のうち、学習データを全体の6割とする
#classifier = tree.DecisionTreeClassifier()      #分類器を生成 決定木というやつらしい
estimator = tree.DecisionTreeClassifier(max_depth=3)
classifier = ensemble.AdaBoostClassifier(base_estimator=estimator, n_estimators=20)
classifier.fit(images[:train_size], labels[:train_size])    #分類器に学習データを与え、学習させる

#%% 分類器の性能評価
from sklearn import metrics

expected = labels[train_size:]              #サンプルデータ移行のデータを性能評価に使う
predicted = classifier.predict(images[train_size:])    #分類器に性能評価用データを読み込ませ、回答を出させる

print('Accuracy:\n', metrics.accuracy_score(expected, predicted))   #正答率を計算する
print('\nConfusion matrix:\n', metrics.confusion_matrix(expected, predicted))   #混合行列の表示(予測と実際の正誤表)
print('\nPrecision:\n', metrics.precision_score(expected, predicted, pos_label=3))    #label3の適合率
print('\nRecall:\n', metrics.recall_score(expected, predicted, pos_label=3)) #label3の再現率
print('\nF-measure:\n',metrics.f1_score(expected, predicted, pos_label=3))   #label3のF値