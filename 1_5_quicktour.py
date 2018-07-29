# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:10:50 2018

@author: RyusukeTakeuchi
"""

#################################################
#####  MEMO                                 #####
#####   F5 to run                           #####
#####   #%% to create cell                  #####
#####   trl+Enter to run current cell       #####
#####   numpy.arrange to create progression #####
#####   numpy.reshape to reshape array      #####
#################################################



####データの準備
import matplotlib.pyplot as plt
import numpy as np

###定義

#x軸
x_max = 1
x_min = -1

#y軸
y_max = 2
y_min = -1

#スケール(1単位に何点使うか)
SCALE = 50

#train/testでTestデータの割合を指定
TEST_RATE = 0.3


##データ生成
data_x = np.arange(x_min, x_max, 1 / float(SCALE)).reshape(-1, 1)

data_ty = data_x ** 2 #ノイズが乗る前の値
data_vy = data_ty + np.random.randn(len(data_ty), 1) * 0.5 #ノイズを乗せる


### 学習データ/テストデータに分割(分類問題、回帰問題で使用するため)
# 学習データ _train
# テストデータ _test

# 学習データ/テストデータの分割処理
# def～returnで関数の定義っぽいぞ
def split_train_test(array):
    # 配列サイズ取得
    length = len(array)
    # 学習データ点数取得
    n_train = int(length * (1 - TEST_RATE))
    
    #indices indexの複数
    indices = list(range(length))
    np.random.shuffle(indices )
    idx_train = indices[:n_train]
    idx_test =  indices[n_train:]
    
    return sorted(array[idx_train]), sorted(array[idx_test])

#インデックスリストを分割
indices = np.arange(len(data_x)) #インデックスのリスト値
#上で定義した関数をコール
idx_train, idx_test = split_train_test(indices)

#学習データ
x_train = data_x[idx_train]
y_train = data_vy[idx_train]

#テストデータ
x_test = data_x[idx_test]
y_test = data_vy[idx_test]


##グラフ描画
#分析対象の散布図
plt.scatter(data_x, data_vy, label = 'target')

#もとの線を表示
plt.scatter(data_x, data_ty, linestyle=':', label = 'non noise curve')

#軸の範囲を設定
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#凡例の表示位置指定
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

#グラフを表示
plt.show()


#%%
######################################
#### 分類問題

##分類ラベル作成
CLASS_RADIUS = 0.6

#近い/遠いでクラス分け 近いとTrue, 遠いとFalse
#右辺の判断式の結果配列を格納
labels = (data_x**2+data_vy**2) < CLASS_RADIUS**2
#確認用
#print(labels)

#学習データ/テストデータに分割
label_train = labels[idx_train]
label_test = labels[idx_test]


#グラフ描画
#近い/遠いクラス、学習/テストの4種類の散布図を重ねる

#学習データ内のラベルがTrueのものだけを出力
plt.scatter(x_train[label_train], y_train[label_train], 
            c='black', s=30, marker='*', label ='near train')
#学習データ内のラベルがTrueでないものを出力
plt.scatter(x_train[label_train != True], y_train[label_train != True], 
            c='black', s=30, marker='+', label ='far train')
#テストデータ内のラベルがTrueのものだけを出力
plt.scatter(x_test[label_test], y_test[label_test], 
            c='black', s=30, marker='^', label ='near test')
#テストデータ内のラベルがTrueでないものを出力
plt.scatter(x_test[label_test != True], y_test[label_test != True], 
            c='black', s=30, marker='x', label ='far test')

#もとのグラフを表示
plt.plot(data_x, data_ty, linestyle = ':', label = 'non noise curve')

#クラスのぶん離縁
circle = plt.Circle((0,0), CLASS_RADIUS, alpha = 0.1, label='near area')
ax = plt.gca()
ax.add_patch(circle)

#軸の範囲を設定
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#凡例の表示位置指定
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

#グラフを表示
plt.show()


#%%
######################################
#### 回帰問題

from sklearn import linear_model

###一次式で回帰
# x値
X1_TRAIN = x_train
X1_TEST = x_test

# 学習
model = linear_model.LinearRegression()
model.fit(X1_TRAIN, y_train)

#グラフに描画
plt.plot(x_test, model.predict(X1_TEST), linestyle='-.', label='poly deg 1')


### 二次式で回帰
# x値
X2_TRAIN = np.c_[x_train**2, x_train]
X2_TEST = np.c_[x_test**2, x_test]

# 学習
model = linear_model.LinearRegression()
model.fit(X2_TRAIN, y_train)

#グラフに描画
plt.plot(x_test, model.predict(X2_TEST), linestyle='--', label='poly deg 2')


### 九次式で回帰
# x値
X9_TRAIN = np.c_[x_train**9, x_train**8, x_train**7, x_train**6, x_train**5,
                 x_train**4, x_train**3, x_train**2, x_train]
X9_TEST = np.c_[x_test**9, x_test**8, x_test**7, x_test**6, x_test**5, 
                x_test**4, x_test**3, x_test**2, x_test]

# 学習
model = linear_model.LinearRegression()
model.fit(X9_TRAIN, y_train)

#グラフに描画
plt.plot(x_test, model.predict(X9_TEST), linestyle='-', label='poly deg 9')


### データの表示
plt.scatter(x_train, y_train, c='black', s=30, marker='v', label='train')
plt.scatter(x_test, y_test, c='black', s=30, marker='x', label='train')

#もとのグラフを表示
plt.plot(data_x, data_ty, linestyle = ':', label = 'non noise curve')


#軸の範囲を設定
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#凡例の表示位置指定
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

#グラフを表示
plt.show()

#%%
######################################
#### クラスタリング

from sklearn import cluster

# x, y データを結合
data = np.c_[data_x, data_vy]

# 3つのクラスタに分類
model = cluster.KMeans(n_clusters=3)
model.fit(data)

# dataの分類結果(0～(n_clusters -1)の番号がつけられている)
labels = model.labels_

plt.scatter(data_x[labels == 0], data_vy[labels == 0], c='black', s = 30, 
            marker = 'v', label = 'cluster 0')
plt.scatter(data_x[labels == 1], data_vy[labels == 1], c='black', s = 30, 
            marker = 'x', label = 'cluster 1')
plt.scatter(data_x[labels == 2], data_vy[labels == 2], c='black', s = 30, 
            marker = '*', label = 'cluster 2')

#もとのグラフを表示
plt.plot(data_x, data_ty, linestyle = ':', label = 'non noise curve')


#軸の範囲を設定
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#凡例の表示位置指定
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

#グラフを表示
plt.show()
