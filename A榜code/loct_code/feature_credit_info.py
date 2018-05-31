# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os

import pandas as pd

file_name = 'dealed_data/target_credit_info.csv'
# 这里面是提取credit_info表里面的特征
print('......读取表并合并表')
train_target = pd.read_csv('data/train_target.csv')
train_credit_info = pd.read_csv('data/train_credit_info.csv')
test_target = pd.read_csv('data/test_list.csv')
test_credit_info = pd.read_csv('data/test_credit_info.csv')
target = pd.concat([train_target, test_target], axis=0)
credit_info = pd.concat([train_credit_info, test_credit_info], axis=0)
if os.path.exists(file_name):
    target_credit_info = pd.read_csv(file_name, encoding='utf-8', low_memory=False)
else:
    target_credit_info = pd.merge(target, credit_info, on='id', how='left')
    target_credit_info.to_csv(file_name, encoding='utf-8', index=False)
print('......读取完毕')

print('......特征提取\n')
feature = pd.DataFrame()

feature['credit_score'] = target_credit_info['credit_score'].fillna(-1)
print('......网购平台信用评分特征提取完毕')
feature['quota'] = target_credit_info['quota'].fillna(-1)
print('......网购平台信用额度特征提取完毕')
feature['overdraft'] = target_credit_info['overdraft'].fillna(-1)
print('......网购平台信用额度使用值特征提取完毕')
feature['overdraft_div_quota'] = feature['overdraft']/feature['quota']
feature['overdraft_div_quota'].fillna(-1)
print('......网购平台信用额度使用率特征提取完毕\n')

print('......特征提取完毕')

print('......保存特征')
feature.to_csv('feature/feature_credit_info.csv', index=False)
print('......结束')


