# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

###########################################
#####  MEMO                           #####
#####   F5 to run                     #####
#####   #%% to create cell            #####
#####   trl+Enter to run current cell #####
###########################################

# numpyをインポート
import numpy as np
# matplotlibをインポート
import matplotlib.pyplot as plt

#%%
# 二次元配列
# 配列の定義
x = np.array([1, 2, 3])
# 出力
print(x)

# 行列の定義
A = np.array([[1, 2, 3],[4, 5, 6]])
print(A)

# 行列の大きさM×Nを表示
print(A.shape)
# 行列の型を表示
print(A.dtype)
# ひとつ要素にアクセス
print(A[1, 2])
# 0行目を抽出
print(A[0])
# スライシング(配列の一部を配列として取り出す)
# 二次元配列のまま(2, 1)
print(A[:, 0:1])
# 一次元配列として取り出す(2,)
print(A[:,0])
# 転置を取る
print(A.T)
#転置を行うとshapeも変更される
print(A.T.shape)

#%%
#３次元配列を作成
B=np.array([[[1,2],[3,4],[5,6]]])
print(B)
print(B.T)
print(B.shape)
print(B.T.shape)

#%%形状を変える
# 要素数を変えない範囲でndarrayの形状を変えられる
print(A.reshape(1, 6))

# 要素数が一致しない場合エラーを吐く
print(A.reshape(1,7))
#ValueError cannot reshape array of size 6 into shape (1,7)と言われた。ストレートなエラー文だね。

#%% 配列の連結
#rであれば行(row)の連結
# Cはすべて1の配列
C = np.ones((2,3))
print(np.r_[A,C])
#cであれば列columnの連結
print(np.c_[A,C])


#%% 四則演算
#要素ごとの演算結果が得られる
#和
print(A + C)
#差
print(A - C)
#積(*を使っても行列計算にはならないよ)
print(A * C)
#商
print(C / A)
#行列積
D = np.ones([3,2])
print(np.dot(A, D))
print(A.dot(D))

#%%
#サイズの一致しない演算は片方の次元の長さが0or1の場合は拡張されて計算される
#次元の長さが2以上だと拡張されない？
E=np.array([[1,2,3]])
print(E)
print(A+E)
F=np.array([1,2,3])
print(F)
print(A+F)

#%%グラフプロット
# x軸の領域と精度を設定し、x値を用意
# -3～3 精度0.1
x = np.arange(-3, 3, 0.1)
#各方程式のy値を用意
y_sin = np.sin(x)
x_rand = np.random.rand(100) * 6 - 3
y_rand = np.random.rand(100) * 6 - 3
# figure オブジェクトを作成
plt.figure()
#一つのグラフで表示する設定
plt.subplot(1, 1, 1)

#各方程式の線形とマーカー、ラベルを設定しプロット
plt.plot(x, y_sin, marker='o', markersize = 5, label = 'line')
#散布図
plt.scatter(x_rand, y_rand, label = 'scatter')

#判例表示設定
plt.legend()

#グリッド線表示
plt.grid(True)

#グラフ表示
plt.show()