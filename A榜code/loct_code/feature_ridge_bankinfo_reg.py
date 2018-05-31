# -*- coding: utf-8 -*-
import sys
from collections import Counter

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

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
feature = pd.concat([feature, count_brand_feature], axis=1)
print('......每种银行次数特征提取完毕')

# 每个商标在里面出现的频率
bank_name_single = np.unique(target_bankcard_info['bank_name'])
def fre_count_brand(brand):
    brand_size = Counter(brand)
    brand_list = []
    for i in bank_name_single:
        brand_list.append(float(brand_size[i])/len(brand))
    return brand_list
count_brand_feature = target_bankcard_info.groupby(['id'], sort=False)['bank_name'].agg(fre_count_brand)
count_brand_feature = pd.DataFrame(list(count_brand_feature))
feature = pd.concat([feature, count_brand_feature], axis=1)
print('......每种银行频率特征提取完毕')

X = np.array(feature[:len(train_target)])
test = np.array(feature[len(train_target):])
y = np.array(train_target['target'])
n_flods = 5
kf = KFold(n_splits=n_flods,shuffle=True,random_state=1017)
kf = kf.split(X)

# ######################## ridge reg #########################3
cv_pred = []
stack = np.zeros((len(y),1))
stack_te = np.zeros((len(test),1))
model_1 = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=250, normalize=False, tol=0.01,random_state=1017)
for i ,(train_fold,test_fold) in enumerate(kf):
     X_train, X_validate, label_train, label_validate = X[train_fold, :], X[test_fold, :], y[train_fold], y[test_fold]
     model_1.fit(X_train, label_train)
     val_ = model_1.predict(X=X_validate)
     stack[test_fold] = np.array(val_).reshape(len(val_),1)
     cv_pred.append(model_1.predict(test))
s = 0
for i in cv_pred:
    s = s+i
s = s/n_flods
print(stack)
print(s)
df_stack1 = pd.DataFrame(stack)
df_stack2 = pd.DataFrame(s)
df_stack = pd.concat([df_stack1,df_stack2
                ], axis=0)
df_stack = df_stack.rename({0:'ridge_reg'}, axis=1)
df_stack.to_csv('feature/feature_ridge_reg.csv', encoding='utf8', index=None)

