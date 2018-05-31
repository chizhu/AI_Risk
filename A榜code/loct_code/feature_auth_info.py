# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import util
import pandas as pd

file_name = 'dealed_data/target_auth_info.csv'
# 这里面是提取auth_info表里面的特征
print('......读取表并合并表')
train_target = pd.read_csv('data/train_target.csv')
train_auth_info = pd.read_csv('data/train_auth_info.csv')
test_target = pd.read_csv('data/test_list.csv')
test_auth_info = pd.read_csv('data/test_auth_info.csv')
target = pd.concat([train_target, test_target], axis=0)
auth_info = pd.concat([train_auth_info, test_auth_info], axis=0)
if os.path.exists(file_name):
    target_auth_info = pd.read_csv(file_name, encoding='utf-8', low_memory=False)
else:
    target_auth_info = pd.merge(target, auth_info, on='id', how='left')
    target_auth_info.to_csv(file_name, encoding='utf-8', index=False)
print('......读取完毕')

# 中国电信2G / 3G号段：133，153， 180，181，189
# 4G号段：173， 177
# 中国联通2G / 3G号段：130，131，132，155，156，185，186
# 3G上网卡：145
# 4G号段：176，185
# 中国移动2G / 3G号段：134，135，136，137，138，139，150，151，152，158，159，182，183，184
# 3G上网卡：147
# 4G号段：178

print('......特征提取\n')
feature = pd.DataFrame()

# 中国电信:1 中国联通:2 中国移动:3 未知：4
def phone_where(var):
    if var == 0:
        return 0
    pos = float(var[:3])
    if (pos==133)or(pos==153)or(pos==180)or(pos==181)or(pos==189)or(pos==173)or(pos==177):
        return 1
    if (pos==130)or(pos==131)or(pos==132)or(pos==155)or(pos==156)or(pos==185)or(pos==186)or(pos==145)or(pos==176)or(pos==185):
        return 2
    if (pos==134)or(pos==135)or(pos==136)or(pos==137)or(pos==138)or(pos==139)or(pos==150)or(pos==151)or(pos==152)or(pos==158)or(pos==159)or(pos==182)or(pos==183)or(pos==184)or(pos==147)or(pos==178):
        return 3
    return 4
phone_list = []
phone = target_auth_info['phone'].fillna(0)
for i in phone:
    phone_list.append(phone_where(i))
feature['number_pos'] = phone_list
print('......电话公司特征提取完毕')

# 2G/3G:1 3G上网卡:2 4G:3
def phone_g(var):
    if var == 0:
        return 0
    pos = float(var[:3])
    if (pos==173)or(pos==177)or(pos==176)or(pos==185)or(pos==178):
        return 3
    if (pos==147)or(pos==145):
        return 2
    return 1
phone_glist = []
phone = target_auth_info['phone'].fillna(0)
for i in phone:
    phone_glist.append(phone_g(i))
feature['number_g'] = phone_glist
print('......电话是什么网段特征提取完毕')


# 认证时间转换成时间戳
def auth_timestamp(var):
    if var == '0000-00-00':
        return -1
    return util.time_to_timestamp(var + ' 00:00:00')
auth_time_list = []
auth_time = target_auth_info['auth_time'].fillna('2019-01-01')
for i in auth_time:
    auth_time_list.append(auth_timestamp(i))
feature['auth_time'] = auth_time_list
print('......认证时间特征提取完毕')

def auth_timestamp(var, app_tm):
    app_tm = app_tm.split('.')[0]
    if var == '0000-00-00':
        return -1
    return util.time_to_timestamp(var + ' 00:00:00') - util.time_to_timestamp(app_tm)
auth_time_list = []
auth_time = target_auth_info['auth_time'].fillna('2019-01-01')
appl_sbm_tm = target_auth_info['appl_sbm_tm'].fillna('2019-01-01 00:00:00')
for i,value in enumerate(auth_time):
    auth_time_list.append(auth_timestamp(value, appl_sbm_tm[i]))
feature['auth_time_cut_sbm_tm'] = auth_time_list
print('......借贷时间减去认证时间特征提取完毕\n')

print('......特征提取完毕')

print('......保存特征')
feature.to_csv('feature/feature_auth_info.csv', index=False)
print('......结束')


