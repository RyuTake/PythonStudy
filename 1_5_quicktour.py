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