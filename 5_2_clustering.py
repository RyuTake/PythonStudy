# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:47:57 2018

@author: rtake
"""

import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets

#irisデータを読み込み
iris = datasets.load_iris()
data = iris['data']

# 学習→クラスタの生成
model = cluster.KMeans(n_clusters=3) #学習、クラスタを3つにわける
model.fit(data) # 学習結果からクラスタを生成

#学習結果のラベル取得
labels = model.labels_

#グラフの描画
#三つのクラスタのうち、ラベル０のクラスタをプロットする
ldata = data[labels == 0] 
plt.scatter(ldata[:,2], ldata[:,3], c='black', alpha= 0.3, s=100, marker="o")

#三つのクラスタのうち、ラベル１のクラスタをプロットする
ldata = data[labels == 1] 
plt.scatter(ldata[:,2], ldata[:,3], c='black', alpha= 0.3, s=100, marker="^")

#三つのクラスタのうち、ラベル２のクラスタをプロットする
ldata = data[labels == 2] 
plt.scatter(ldata[:,2], ldata[:,3], c='black', alpha= 0.3, s=100, marker="*")

plt.xlabel(iris['feature_names'][2])
plt.ylabel(iris['feature_names'][3])

plt.show()

#%% クラスタリング結果を6つの図を一括で表示する
MARKERS = ["v", "^", "+", "x", "d", "p","s","1","2"]

#指定されたインデックスのfeature値で散布図を作成する関数
#defで関数定義
def scatter_by_features(feat_idx1, feat_idx2):
    for lbl in range(labels.max() + 1):
        clustered = data[labels == lbl]
        plt.scatter(clustered[:, feat_idx1], clustered[:, feat_idx2], c = 'black', alpha=0.3, s=100, marker = MARKERS[lbl], label='label{}'.format(lbl))
        
    plt.xlabel(iris['feature_names'][feat_idx1], fontsize='xx-large')
    plt.ylabel(iris['feature_names'][feat_idx2], fontsize='xx-large')
    
plt.figure(figsize=(16,16))

#feature "sepal length" & "sepal width"
plt.subplot(3,2,1)
scatter_by_features(0,1)

#feature "sepal length" & "petal length"
plt.subplot(3,2,2)
scatter_by_features(0,2)

#feature "sepal length" & "petal width"
plt.subplot(3,2,3)
scatter_by_features(0,3)

#feature "sepal width" & "petal length"
plt.subplot(3,2,4)
scatter_by_features(1,2)

#feature "sepal width" & "petal width"
plt.subplot(3,2,5)
scatter_by_features(1,3)

#feature "petal length" & "petal width"
plt.subplot(3,2,6)
scatter_by_features(2,3)

plt.tight_layout()
plt.show()

#%% こたえあわせ
from sklearn import metrics
print(metrics.confusion_matrix(iris['target'], model.labels_))
iris['target_names']