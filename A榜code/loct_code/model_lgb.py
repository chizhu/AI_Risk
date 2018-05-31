# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import numpy as np
from sklearn import model_selection

print('开始处理特征......')
# 读入测试集和训练集
train_target = pd.read_csv('data/train_target.csv', encoding='utf-8')
test_target = pd.read_csv('data/test_list.csv', encoding='utf-8')

# 读入特征
df_user_info = pd.read_csv('feature/feature_user_info.csv', encoding='utf-8')
df_credit_info = pd.read_csv('feature/feature_credit_info.csv', encoding='utf-8')
df_auth_info = pd.read_csv('feature/feature_auth_info.csv', encoding='utf-8')
df_bankcard_info = pd.read_csv('feature/feature_bankcard_info.csv', encoding='utf-8')
df_ridge_reg = pd.read_csv('feature/feature_ridge_reg.csv', encoding='utf-8')
df_order_info = pd.read_csv('feature/feature_order_info.csv', encoding='utf-8')
df_order_info_dealed = pd.read_csv('feature/feature_order_info_dealed.csv', encoding='utf-8')
df_recieve_addr_info = pd.read_csv('feature/feature_recieve_addr_info.csv', encoding='utf-8')

feature = pd.concat([df_user_info,df_credit_info,df_auth_info,df_bankcard_info,
                     df_ridge_reg,df_order_info,df_recieve_addr_info,df_order_info_dealed,
                     ], axis=1)

train_feature = feature[:len(train_target)]
test_feature = feature[len(train_target):]
label = list(train_target['target'])

# 删除特征
# feature_importance = np.loadtxt('learn/feature_importance')
# pos = np.argsort(feature_importance)
# delete_head = pos[0:5]
# train_feature.drop(train_feature.columns[delete_head], axis=1,inplace=True)
# test_feature.drop(test_feature.columns[delete_head], axis=1,inplace=True)

# 切分训练集
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, label, test_size=0.2, random_state=1017)
# 注掉就是不切分
# train_feature = X_train
# label = Y_train
print('特征处理完毕......')

########### T-POT 构建特征 ############
# train_feature[np.isnan(train_feature)] = 0
# X_test[np.isnan(X_test)] = 0
# tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
# tpot.fit(X_train, Y_train)
# print(tpot.score(X_test, Y_test))

# 计算logloss
from sklearn import metrics

def CaculateAUC(test_pre,test_label):
    result = metrics.auc(test_label, test_pre)
    return result

###################### lgb ##########################
import lightgbm as lgb

print('载入数据......')

lgb_train = lgb.Dataset(train_feature, label)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
print('开始训练......')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'}
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=81,
                valid_sets=lgb_eval
                )
gbm.save_model('model/lgb_model.txt')
preds_prob = gbm.predict(test_feature)

# print('验证集auc:' +  str(CaculateAUC(gbm.predict(X_test), Y_test)))
print ("特征重要性:" + str(gbm.feature_importance()))
print('特征名称:' + train_feature.columns)
feature_name = train_feature.columns
feature_imp = gbm.feature_importance()
feature_name = feature_name[np.argsort(feature_imp)]
feature_imp = feature_imp[np.argsort(feature_imp)]
for i,value in enumerate(feature_name):
    print(value + ':' + str(feature_imp[i]))
print("多少个大于0.5的数据:" + str(np.sum(preds_prob >= 0.5)))

# auc只关心排序结果
################### 自己线下的预测的结果与标准结果保存 ###################
see_res = pd.DataFrame()
see_res['self'] = gbm.predict(X_test)
see_res['real'] = Y_test
see_res['id'] = np.array(train_target['id'])[X_test.reset_index()['index']]
see_res.to_csv('another/a_sr.csv', index=False)


########################## 保存结果 ############################
print('保存结果......')
# np.savetxt('learn/feature_importance',gbm.feature_importance())
df_result = pd.DataFrame()
df_result['ID'] = test_target['id']
df_result['PROB'] = preds_prob
df_result.to_csv('result/lgb_result.csv', header=True, index=False)
