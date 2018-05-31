# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import re

import pandas as pd
import numpy as np
import util

file_name = 'dealed_data/target_user_info.csv'
# 这里面是提取user_info表里面的特征
print('......读取表并合并表')
train_target = pd.read_csv('data/train_target.csv')
train_user_info = pd.read_csv('data/train_user_info.csv')
test_target = pd.read_csv('data/test_list.csv')
test_user_info = pd.read_csv('data/test_user_info.csv')
target = pd.concat([train_target, test_target], axis=0)
user_info = pd.concat([train_user_info, test_user_info], axis=0)
# pd.merge太慢，merge后我先保存就只merge一次
# id列表每个表的id顺序不一样要注意处理顺序
if os.path.exists(file_name):
    target_user_info = pd.read_csv(file_name, encoding='utf-8', low_memory=False)
else:
    target_user_info = pd.merge(target, user_info, on='id', how='left')
    target_user_info.to_csv(file_name, encoding='utf-8', index=False)
print('......读取完毕')

print('......特征提取\n')
feature = pd.DataFrame()

# 性别 男:1 女:2 nan:3 保密:4
def sex_dict(var):
    return {
            u'男': 1,
            u'女': 2,
            np.nan: 0,
            u'保密': -1,
    }.get(var,-1)
sex_to_number = []
sex = target_user_info['sex']
for i in sex:
    sex_to_number.append(sex_dict(i))
feature['sex'] = sex_to_number
print('......性别特征提取完毕')

# 婚否
def merriage_dict(var):
    return {
            u'已婚': 1,
            u'未婚': 2,
            np.nan: 0,
            u'保密': -1,
    }.get(var,-1)
merriage_to_number = []
merriage = target_user_info['merriage']
for i in merriage:
    merriage_to_number.append(merriage_dict(i))
feature['merriage'] = merriage_to_number
print('......婚否特征提取完毕')

# 年龄
def age_dict(var):
    try:
        year = var.split('-')[0]
        if len(year.split('后')) <= 1:
            age = 2018 - int(year)
        else:
            year = year.split('后')[0]
            age = 2018 - (int(year) + 1900)
        return age
    except ValueError:
        return 0
age_to_number = []
birthday = target_user_info['birthday'].fillna('2019-01-01')
for i in birthday:
    age_to_number.append(age_dict(str(i)))
feature['age'] = age_to_number
print('......年龄特征提取完毕')

# 爱好的长度
def hobby_length(var):
    if (var == '这里没有hobby'):
        return 0
    return len(re.split('[;/]', var))
hobby_length_number = []
hobby = target_user_info['hobby'].fillna('这里没有hobby')
for i in hobby:
    hobby_length_number.append(hobby_length(i))
feature['hobby_length'] = hobby_length_number
print('......hobby个数提取完毕')

# 爱好是否随便乱输入 是乱输入:1 不是：2 nan:0
def hobby_incorrect(var):
    if (var == '这里没有hobby'):
        return 0
    list_hobby = re.split('[;/]', var)
    try:
        float(list_hobby[0])
        return 1
    except ValueError:
        return 2
is_hobby_correct = []
hobby = target_user_info['hobby'].fillna('这里没有hobby')
for i in hobby:
    is_hobby_correct.append(hobby_incorrect(i))
feature['hobby_is_correct'] = is_hobby_correct
print('......hobby是否乱输入提取完毕')

# 收入水平 nan:0 2000以下:1 2000-3999:2 4000-5999:3 6000-7999:4 8000以上:5
def income_dict(var):
    return {
            u'2000元以下': 1,
            u'2000-3999元': 2,
            u'4000-5999元': 3,
            u'6000-7999元': 4,
            u'8000元以上':5,
    }.get(var, 0)
income_list = []
income = target_user_info['income']
for i in income:
    income_list.append(income_dict(i))
feature['income_level'] = income_list
print('......income水平特征提取完毕')

# 学历 nan:0 其他:1 中专:2 初中:3 高中:4 大专:5 本科:6 硕士:7 博士:8
def degree_dict(var):
    return {
            u'其他': 1,
            u'中专': 2,
            u'初中': 3,
            u'高中': 4,
            u'大专': 5,
            u'本科': 6,
            u'硕士': 7,
            u'博士': 8,
    }.get(var, 0)
degree_list = []
degree = target_user_info['degree']
for i in degree:
    degree_list.append(degree_dict(i))
feature['degree_level'] = degree_list
print('......degree水平特征提取完毕')

# 所在行业 太多了写代码映射成字典
industry_list = []
industry = target_user_info['industry']
industry_single = np.unique(industry)
industry_code = np.arange(1, len(industry_single) + 1, 1)
industry_pro_dict = dict(zip(industry_single, industry_code))
def industry_dict(var):
    return industry_pro_dict.get(var, 0)
for i in industry:
    industry_list.append(industry_dict(i))
feature['industry'] = industry_list
print('......industry编码特征提取完毕')

def bound_qq_weixin(var):
    return {
            u'已绑定': 2,
            u'未绑定': 1,
    }.get(var, 0)

# 是否绑定qq
qq_bound_list = []
qq_bound = target_user_info['qq_bound']
for i in qq_bound:
    qq_bound_list.append(bound_qq_weixin(i))
feature['qq_bound'] = qq_bound_list
print('......qq是否绑定特征提取完毕')

# 是否绑定微信
chat_bound_list = []
chat_bound = target_user_info['wechat_bound']
for i in chat_bound:
    chat_bound_list.append(bound_qq_weixin(i))
feature['chat_bound'] = chat_bound_list
print('......微信是否绑定特征提取完毕')

# 会员级别
def viplevel_dict(var):
    return {
            u'注册会员': 1,
            u'铜牌会员': 2,
            u'银牌会员': 3,
            u'金牌会员': 4,
            u'钻石会员': 5,
    }.get(var, 0)
viplevel_list = []
viplevel = target_user_info['account_grade']
for i in viplevel:
    viplevel_list.append(viplevel_dict(i))
feature['vip_level'] = viplevel_list
print('......vip级别特征提取完毕')

# 据身份证获得的性别(倒数第二位是奇数还是偶数) 男:1 女:2
def id_card_sex(var):
    if var == 0:
        return 0
    else:
        num = int(var[-2])
        if (num % 2) == 0:
            return 2
        else:
            return 1
id_card_sex_list = []
id_card = target_user_info['id_card'].fillna(0)
for i in id_card:
    id_card_sex_list.append(id_card_sex(i))
feature['id_card_sex'] = id_card_sex_list
print('......身份证号码得到的性别特征提取完毕')

# 据身份证获得的城市(前两位的数字)
def id_card_city(var):
    if var == 0:
        return 0
    else:
        return int(var[:2])
id_card_city_list = []
id_card = target_user_info['id_card'].fillna(0)
for i in id_card:
    id_card_city_list.append(id_card_city(i))
feature['id_card_city'] = id_card_city_list
print('......身份证号码得到的城市特征提取完毕')

def auth_timestamp(var, app_tm):
    try:
        year = var.split('-')[0]
        if len(year.split('后')) <= 1:
            year = int(year)
        else:
            year = year.split('后')[0]
            year = (int(year) + 1900)
    except ValueError:
        return -1
    var = str(year)+ '-' + var.split('-')[1] + '-' + var.split('-')[2]
    app_tm = app_tm.split('.')[0]
    try:
        util.time_to_timestamp(var + ' 00:00:00')
    except ValueError:
        return -1
    return  util.time_to_timestamp(app_tm) - util.time_to_timestamp(var + ' 00:00:00')
brithday_cut_apptm = []
birthday = target_user_info['birthday'].fillna('2019-01-01')
appl_sbm_tm = target_user_info['appl_sbm_tm'].fillna('2019-01-01 00:00:00')
for i,value in enumerate(birthday):
    brithday_cut_apptm.append(auth_timestamp(value, appl_sbm_tm[i]))
feature['bir_cut_sbm_tm'] = brithday_cut_apptm
print('......借贷时间减去生日特征提取完毕\n')
# print(target_user_info.groupby('account_grade').describe())
# 检查一下特征是否有问题
# print(feature.ix[8563])
# print(target_user_info.ix[8563])
# print(feature.groupby('id_card_city').describe())
print('......特征提取完毕')

print('......保存特征')
feature.to_csv('feature/feature_user_info.csv', index=False)
print('......结束')


