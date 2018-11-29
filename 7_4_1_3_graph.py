# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:04:30 2018

@author: rtake
"""

import pandas as pd

#気象データ読み込み
tmp = pd.read_csv(
        u'47891_高松.csv',
        parse_dates={'date_hour':["日時"]},
        index_col="date_hour",
        na_values="×",
        engine = 'python') #OSエラーの対応として読み込みエンジンを指定

del tmp["時"] #時の列は扱わないので削除

#カラム名を英語で統一
columns = {
        "降水量(mm)":"rain",
        "気温(℃)":"temperature",
        "日照時間(h)":"sunhour",
        "湿度(%)":"humid",
}
tmp.rename(columns=columns, inplace=True)