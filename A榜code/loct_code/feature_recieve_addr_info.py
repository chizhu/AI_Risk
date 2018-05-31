# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import numpy as np
import pandas as pd

file_name = 'dealed_data/target_recieve_addr_info_dealed.csv'
# 这里面是提取recieve_addr_info表里面的特征
print('......读取表并合并表')
train_target = pd.read_csv('data/train_target.csv')
train_recieve_addr_info = pd.read_csv('data/train_recieve_addr_info.csv')
test_target = pd.read_csv('data/test_list.csv')
test_recieve_addr_info = pd.read_csv('data/test_recieve_addr_info.csv')
target = pd.concat([train_target, test_target], axis=0)
recieve_addr_info = pd.concat([train_recieve_addr_info, test_recieve_addr_info], axis=0)
if os.path.exists(file_name):
    target_recieve_addr_info = pd.read_csv(file_name, encoding='utf-8', low_memory=False)
else:
    target_recieve_addr_info = pd.merge(target, recieve_addr_info, on='id', how='left')
    target_recieve_addr_info.to_csv(file_name, encoding='utf-8', index=False)
print('......读取完毕')

print('......特征提取\n')
feature = pd.DataFrame()

# 对应id的reg_info个数
def count_rec_info(row):
    # addr_id, region, phone, fix_phone, receiver_md5
    row = np.array(row)
    if (np.isnan(row[0]))and(len(row) == 1):
        return 0
    return len(row)
feature_count_rec_info = target_recieve_addr_info.groupby('id', sort=False).agg(count_rec_info)
feature_count_rec_info = feature_count_rec_info.reset_index()
del feature_count_rec_info['id']
del feature_count_rec_info['target']
feature_count_order_info = feature_count_rec_info.rename({'addr_id': 'count_rec_info'}, axis=1)
feature = pd.concat([feature, feature_count_order_info], axis=1)
print('......购买个数')

# 地区
def order_name_fre(name):
    name = list(name)
    if len(name) < 3:
        if len(name) <= 1:
            return -1
        if name[-1] == name[-2]:
            return 0
        else:
            return 1
    else:
        if name[-1] == name[-3]:
            return 2
        else:
            return 3
temp = target_recieve_addr_info.groupby('id', sort=False)['region'].agg(order_name_fre)
feature['region_fre'] = list(temp)
print('......购买是否换地区')

# 座机填写次数比上总次数
def pos_count_div_sum_count(name):
    name = name.fillna(0)
    name = list(name)
    count = 0
    for i in name:
        if i == 0:
           count = count + 1
    return float(count)/len(name)
temp = target_recieve_addr_info.groupby('id', sort=False)['fix_phone'].agg(pos_count_div_sum_count)
feature['fix_phone_fre'] = list(temp)
print('......固话有填写吗')


print('......保存特征')
feature.to_csv('feature/feature_recieve_addr_info.csv', index=False)
print('......结束')




