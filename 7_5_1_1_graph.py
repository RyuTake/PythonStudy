# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 19:50:55 2018

@author: rtake
"""

import pandas as pd

# 四国電力の電力消費量データ読み込み
ed = [pd.read_csv(
    'shikoku_electlicity_%d.csv' % year,
    skiprows=3,
    names=['DATE', 'TIME', 'consumption'],
    parse_dates={'date_hour': ['DATE', 'TIME']},
    index_col='date_hour')
    for year in [2012, 2013, 2014, 2015, 2016]
]

elec_data = pd.concat(ed)

#気象データ読み込み
tmp = pd.read_csv(
    u'47891_高松.csv'
    ,parse_dates={'date_hour': ["日時"]}
    ,index_col = "date_hour"
    ,na_values="×"
    ,engine = 'python' #このバージョンの不具合らしく、エンジンを指定しないとバグる
    ,encoding="utf-8" #ファイルがutf-8なのでエンコード
        )

del tmp["時"]  # 「時」の列は使わないので、削除


# 列の名前に日本語が入っているとよくないので、これから使う列の名前のみ英語に変更
columns = {
    "降水量(mm)": "rain",
    "気温(℃)": "temperature",
    "日照時間(h)": "sunhour",
    "湿度(％)": "humid",
}
tmp.rename(columns=columns, inplace=True)

#データ処理
#気象データと電力消費量データを一旦統合し、時間軸をあわせた後再分離
takamatsu = elec_data.join(tmp["tem@erature"]).dropna().as_matrix()

takamatsu_elec = takamatsu[:, 0:1]
takamatsu_wthr = takamatsu[:, 1:]

# -- 可視化 --
import matplotlib.pyplot as plt

# 画像のサイズを設定する
plt.figure(figsize=(10, 6))

# ヒストグラム生成

plt.scatter(takamatsu_wthr['temperature'], takamatsu_elec['consumption'], s=0.1)
plt.xlabel('Temperature(C degree)')
plt.ylabel('electricity consumption(*1000 kw)')

# グラフ保存
plt.savefig('7-5-1-1-graph_1.png')