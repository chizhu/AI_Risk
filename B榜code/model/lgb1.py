import pandas as pd
import lightgbm as lgb
target=pd.read_csv('data/target.csv',low_memory=False)
auth=pd.read_csv('data/auth.csv',low_memory=False)
auth=auth.drop(['auth_time','id_card','phone','real_name','auth_time_year','auth_time_month','appl_sbm_tm','target'],1)
user=pd.read_csv('data/user.csv',low_memory=False)
credit=pd.read_csv('data/credit.csv',low_memory=False)
order=pd.read_csv('data/order_feature.csv',low_memory=False)
order_dup=pd.read_csv('data/order_dup.csv',low_memory=False)
orderpred=pd.read_csv('data/orderpred.csv',low_memory=False)
ordernamefre=pd.read_csv('data/order_name_fre.csv',low_memory=False)
receive=pd.read_csv('data/receive.csv',low_memory=False)
province=pd.read_csv('data/province.csv',low_memory=False)
bank=pd.read_csv('data/bankfeature1.csv',low_memory=False)
bankpred=pd.read_csv('data/bankpred.csv',low_memory=False)
loan_count=pd.read_csv('data/loan_count.csv',low_memory=False)
bankrate=pd.read_csv('data/bank_rate_feature.csv',low_memory=False)
order_time_dur=pd.read_csv('data/order_time_dur.csv',low_memory=False)
feature_loc=pd.read_csv('data/feature_loct.csv',low_memory=False)


data=pd.merge(target,auth,on='id',how='left')
data=pd.merge(data,user,on='id',how='left')
data=pd.merge(data,credit,on='id',how='left')
data=pd.merge(data,order,on='id',how='left')
data=pd.merge(data,order_dup,on='id',how='left')
data=pd.merge(data,orderpred,on='id',how='left')
data=pd.merge(data,ordernamefre,on='id',how='left')
data=pd.merge(data,receive,on='id',how='left')
data=pd.merge(data,province,on='id',how='left')
data=pd.merge(data,bank,on='id',how='left')
data=pd.merge(data,bankpred,on='id',how='left')
data=pd.merge(data,loan_count,on='id',how='left')
data=pd.merge(data,bankrate,on='id',how='left')
data=pd.merge(data,order_time_dur,on='id',how='left')
data=data.merge(feature_loc,on='id',how='left')

data['ordernum_minus']=data['before_count']-data['dup_num']
data['ordernum_divd']=data['dup_num']/data['before_count']

data=data[['id','target','auth_time_day', 'auth_time_weekday', 'id_card_exist',
       'auth_time_exist', 'phone_exist', 'phone_head3', 'appl_auth_time',
       'bankpred', 'orderpred', 'credit_score', 'overdraft', 'quota', 'remain',
       'use_rate', 'bank_count', 'chuxuka_count', 'xinyongka_count', 'sex',
       'qq_bound', 'wechat_bound', 'account_grade', 'age', 'null_count',
       'loan_count', 'overtime_delta', 'amt_order_sum', 'amt_order_mean',
       'amt_order_std', 'amt_order_skew', 'phone_num', 'order_count',
       'dur_day', 'morning_order_count', 'afternoon_order_count',
       'evening_order_count', 'night_order_count', 'weekday_count',
       'weekend_count', 'weekday_mode', 'product_id_null_num',
       'order_null_avg', 'time_dur', 'bank_rate_avg', 'dup_num',
       'before_count', 'order_name_fre_x', 'id_len', 'ordernum_minus',
       'ordernum_divd', 'idcard_first', 'count_rec_info', 'region_fre',
       'fix_phone_fre']]

traindata=data[data.target!=2]
testdata=data[data.target==2]
X_train=traindata.iloc[:,2:]
y_train=traindata.target
X_test=testdata.iloc[:,2:]
y_test=testdata.target

lgb_train = lgb.Dataset(X_train, y_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': { 'auc'},
    'num_leaves': 36,
    'learning_rate': 0.01,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'random_state':1024,
}
print('Start training...')
# train
gbm = lgb.train(params,  lgb_train,  num_boost_round=1300
               )
print('Start predict...')
predict= gbm.predict(X_test)
submit = pd.DataFrame({'ID': testdata.id, 'PROB': predict})
submit.PROB=submit.PROB.apply(lambda x:round(x,4))
submit.to_csv('data/submitB_all_feature.csv',index=False)