# -*- coding: utf-8 -*-
import sys
from collections import Counter

import time

reload(sys)
sys.setdefaultencoding('utf-8')

import os
import util
import pandas as pd
import numpy as np

file_name = 'dealed_data/target_order_info.csv'
# 这里面是提取order_info表里面的特征
print('......读取表并合并表')
train_target = pd.read_csv('data/train_target.csv')
train_order_info = pd.read_csv('data/train_order_info.csv')
test_target = pd.read_csv('data/test_list.csv')
test_order_info = pd.read_csv('data/test_order_info.csv')
target = pd.concat([train_target, test_target], axis=0)
order_info = pd.concat([train_order_info, test_order_info], axis=0)
if os.path.exists(file_name):
    target_order_info = pd.read_csv(file_name, encoding='utf-8', low_memory=False)
else:
    target_order_info = pd.merge(target, order_info, on='id', how='left')
    target_order_info.to_csv(file_name, encoding='utf-8', index=False)

print('......读取完毕')

print('......特征提取\n')
feature = pd.DataFrame()

def count_order_info(row):
    row = np.array(row)
    if (np.isnan(row[0]))and(len(row) == 1):
        return 0
    return len(row)
# 对应id的order_info个数
feature_count_order_info = target_order_info.groupby('id', sort=False)['amt_order'].agg(count_order_info)
feature_count_order_info = feature_count_order_info.reset_index()
del feature_count_order_info['id']
feature_count_order_info = feature_count_order_info.rename({'amt_order': 'count_order_info'}, axis=1)
feature = pd.concat([feature, feature_count_order_info], axis=1)
print('......order_info个数特征提取完毕')


print('......保存特征')
feature.to_csv('feature/feature_order_info.csv', index=False)
print('......结束')
