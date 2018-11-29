# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:00:40 2018

@author: rtake
"""
#%% お試し_csv読み込み
import pandas as pd
pd.read_csv(
        'shikoku_electricity_2012.csv',
        skiprows=3,
        names=['DATE','TIME','consumption'],
        parse_dates={'date_hour':['DATE', 'TIME']},
        index_col='date_hour')

#%% 複数ファイル読み込み～可視化
import pandas as pd

# csv読み込み
ed = [pd.read_csv(
        'shikoku_electricity_%d.csv' % year,
        skiprows=3,
        names=['DATE','TIME','consumption'],
        parse_dates={'date_hour':['DATE','TIME']},
        index_col="date_hour"
        )
    for year in [2012, 2013, 2014, 2015, 2016]
]

elec_data = pd.concat(ed)


# visualisation
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))

delta = elec_data.index - pd.to_datetime('2012/07/01 00:00:00')
elec_data['time'] = delta.days + delta.seconds / 3600.0 / 24.0

plt.scatter(elec_data['time'], elec_data['consumption'], s = 0.1)
plt.xlabel('days from 2012/7/1')
plt.ylabel('electricity consumption(*10000 kWh)')

plt.savefig('7_4_1_1_graph.png')

# ヒストグラム作成
plt.figure(figsize=(10,6))

plt.hist(elec_data['consumption'],bins=50, color = "gray", )
plt.savefig('7-4-1-2-graph.png')