import lightgbm as lgb
import pandas as pd

data=pd.read_csv('best_data.csv')

traindata=data[data.target!=2]
testdata=data[data.target==2]
X_train=traindata.iloc[:,5:]
y_train=traindata.target
X_test=testdata.iloc[:,5:]
y_test=testdata.target

lgb_train = lgb.Dataset(X_train, y_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': { 'auc'},
    'num_leaves': 36,
    'learning_rate': 0.01,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.7,
    'random_state':1024,
}
print('Start training...')
# train
gbm = lgb.train(params,  lgb_train,  num_boost_round=950
               )

print('Start predict...')
predict= gbm.predict(X_test)
submit = pd.DataFrame({'ID': testdata.id, 'PROB': predict})
submit.PROB=submit.PROB.apply(lambda x:round(x,4))
submit.to_csv('submit83587.csv',index=False)