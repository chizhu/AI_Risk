# -*- coding: utf-8 -*-
import sys
from collections import Counter

reload(sys)
sys.setdefaultencoding('utf-8')

import os
import re

import pandas as pd
import numpy as np
import util

# bankcard_info 不是相对应的一个表！！！每个人可能存在多个bankcard_info
file_name = 'dealed_data/target_bankcard_info.csv'
# 这里面是提取bankcard_info表里面的特征
print('......读取表并合并表')
train_target = pd.read_csv('data/train_target.csv')
train_bankcard_info = pd.read_csv('data/train_bankcard_info.csv')
test_target = pd.read_csv('data/test_list.csv')
test_bankcard_info = pd.read_csv('data/test_bankcard_info.csv')
target = pd.concat([train_target, test_target], axis=0)
bankcard_info = pd.concat([train_bankcard_info, test_bankcard_info], axis=0)
# pd.merge太慢，merge后我先保存就只merge一次
# id列表每个表的id顺序不一样要注意处理顺序
if os.path.exists(file_name):
    target_bankcard_info = pd.read_csv(file_name, encoding='utf-8', low_memory=False)
else:
    target_bankcard_info = pd.merge(target, bankcard_info, on='id', how='left')
    target_bankcard_info.to_csv(file_name, encoding='utf-8', index=False)
print('......读取完毕')


print('......特征提取\n')
feature = pd.DataFrame()

# 拥有几张银行卡
feature['bank_length'] = target_bankcard_info.groupby('id', sort=False).size()
print('......拥有几张银行卡特征提取完毕')

# 每种银行的有几张以及频率
def count_card(card_type):
    card_type_size = Counter(card_type)
    save_card_count = card_type_size[u'储蓄卡']
    credit_card_count = card_type_size[u'信用卡']
    save_card_fre = float(save_card_count)/len(card_type)
    credit_card_fre = float(credit_card_count)/len(card_type)
    return [save_card_count, credit_card_count, save_card_fre, credit_card_fre]
card_type_feature = target_bankcard_info.groupby(['id'], sort=False)['card_type'].agg(count_card)
card_type_feature = pd.DataFrame(list(card_type_feature)).rename({0:'save_card_count', 1:'credit_card_count',2:'save_card_fre',3:'credit_card_fre'}, axis=1)
feature = feature.reset_index()
del feature['id']
feature = pd.concat([feature, card_type_feature], axis=1)
print('......每种银行有几张及频率特征提取完毕')

# 是否含有多个号码
def phone_deal(phone):
    phone = np.array(phone)
    phone_count = len(np.unique(phone))
    return [phone_count]
phone_number = target_bankcard_info.groupby(['id'], sort=False)['phone'].agg(phone_deal)
phone_number_feature = pd.DataFrame(list(phone_number)).rename({0:'phone_count'}, axis=1)
feature = pd.concat([feature, phone_number_feature], axis=1)
print('......电话特征提取完毕')

# 每个商标在里面出现的次数
bank_name_single = np.unique(target_bankcard_info['bank_name'])
def count_brand(brand):
    brand_size = Counter(brand)
    brand_list = []
    for i in bank_name_single:
        brand_list.append(brand_size[i])
    return brand_list
count_brand_feature = target_bankcard_info.groupby(['id'], sort=False)['bank_name'].agg(count_brand)
count_brand_feature = pd.DataFrame(list(count_brand_feature))
for i,value in enumerate(bank_name_single):
    count_brand_feature = count_brand_feature.rename({i: str(value) + 'count'}, axis=1)
feature = pd.concat([feature, count_brand_feature], axis=1)
print('......每种银行次数特征提取完毕')

# 每个商标在里面出现的频率
bank_name_single = np.unique(target_bankcard_info['bank_name'])
def count_brand(brand):
    brand_size = Counter(brand)
    brand_list = []
    for i in bank_name_single:
        brand_list.append(float(brand_size[i])/len(brand))
    return brand_list
count_brand_feature = target_bankcard_info.groupby(['id'], sort=False)['bank_name'].agg(count_brand)
count_brand_feature = pd.DataFrame(list(count_brand_feature))
for i,value in enumerate(bank_name_single):
    count_brand_feature = count_brand_feature.rename({i: str(value) + 'fre'}, axis=1)
feature = pd.concat([feature, count_brand_feature], axis=1)
print('......每种银行频率特征提取完毕')

print('......保存特征')
feature.to_csv('feature/feature_bankcard_info.csv', index=False)
print('......结束')
