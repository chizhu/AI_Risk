# -*- coding: utf-8 -*-
import sys

import os
import numpy as np
import pandas as pd
import util

file_name = 'dealed_data/Btarget_recieve_addr_info_dealed.csv'
# 这里面是提取recieve_addr_info表里面的特征
print('......读取表并合并表')
train_target = pd.read_csv('data/train_target.csv')
train_recieve_addr_info = pd.read_csv('data/train_recieve_addr_info.csv')
test_target = pd.read_csv('data/Btest_list.csv')
test_recieve_addr_info = pd.read_csv('data/Btest_recieve_addr_info.csv')
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
def region_fre(name):
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
temp = target_recieve_addr_info.groupby('id', sort=False)['region'].agg(region_fre)
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



# 清理脏数据
print('......开始清理脏数据')
deal_file_name = 'dealed_data/Btarget_order_info_dealed.csv'
if os.path.exists(deal_file_name):
    target_order_info_dealed = pd.read_csv(deal_file_name, encoding='utf-8', low_memory=False)
else:
    train_target = pd.read_csv('data/train_target.csv')
    train_order_info = pd.read_csv('data/train_order_info.csv')
    test_target = pd.read_csv('data/Btest_list.csv')
    test_order_info = pd.read_csv('data/Btest_order_info.csv')
    target = pd.concat([train_target, test_target], axis=0)
    order_info = pd.concat([train_order_info, test_order_info], axis=0)
    target_order_info = pd.merge(target, order_info, on='id', how='left')
    # 对时间归整并排序
    target_order_info_dealed = target_order_info
    def time_deal_fun(row):
        try:
            float(row)
            util.timestamp_to_time(float(row))
        except ValueError:
            return row
    target_order_info_dealed['time_order'] = target_order_info.apply(lambda row: time_deal_fun(row['time_order']), axis=1)
    target_order_info_dealed = target_order_info_dealed.sort_index(by='time_order')
    target_order_info_dealed = pd.merge(target, target_order_info_dealed, on='id', how='left')
    del target_order_info_dealed['appl_sbm_tm_y']
    del target_order_info_dealed['target_y']
    target_order_info_dealed.to_csv(deal_file_name, encoding='utf-8', index=False)

print('......清理完毕')


def order_name_fre(name):
    return float(len(np.unique(name)))/len(name)
feature['order_name_fre'] = target_order_info_dealed.groupby('id', sort=False)['name_rec_md5'].agg(order_name_fre)
print('......出现fre特征提取完毕')



file_name = 'dealed_data/Btarget_auth_info.csv'
# 这里面是提取auth_info表里面的特征
print('......读取表并合并表')
train_target = pd.read_csv('data/train_target.csv')
train_auth_info = pd.read_csv('data/train_auth_info.csv')
test_target = pd.read_csv('data/Btest_list.csv')
test_auth_info = pd.read_csv('data/Btest_auth_info.csv')
target = pd.concat([train_target, test_target], axis=0)
auth_info = pd.concat([train_auth_info, test_auth_info], axis=0)
if os.path.exists(file_name):
    target_auth_info = pd.read_csv(file_name, encoding='utf-8', low_memory=False)
else:
    target_auth_info = pd.merge(target, auth_info, on='id', how='left')
    target_auth_info.to_csv(file_name, encoding='utf-8', index=False)
print('......读取完毕')

print('......特征提取\n')
id_card = target_auth_info['id_card'].fillna(0)
id_card_dealed = []
for i in id_card:
    if i==0:
        id_card_dealed.append(-1)
    else:
        id_card_dealed.append(i[0])
feature['id_card_first'] = id_card_dealed
print('......特征提取完毕')

print('......保存特征')
feature['id'] = target_auth_info['id']
feature.to_csv('data/feature_loct.csv', index=False)
print('......结束')


