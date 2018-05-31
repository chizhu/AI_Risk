import pandas as pd

lgb83587=pd.read_csv('submit83587.csv')
lgb832=pd.read_csv('submit832.csv')

#归一化
min1= lgb83587['PROB'].min()
max1= lgb83587['PROB'].max()
lgb83587['PROB'] = lgb83587['PROB'].map(lambda x:(x-min1)/(max1-min1))

min2= lgb832['PROB'].min()
max2= lgb832['PROB'].max()
lgb832['PROB'] = lgb832['PROB'].map(lambda x:(x-min2)/(max2-min2))

## 简单加权融合
submit=pd.DataFrame()
submit['ID']=lgb83587.ID
submit['PROB'] = 0.7*lgb83587['PROB'] + 0.3*lgb832['PROB']

min3= submit['PROB'].min()
max3= submit['PROB'].max()
submit['PROB'] = submit['PROB'].map(lambda x:(x-min3)/(max3-min3))
submit.PROB=submit.PROB.apply(lambda x:round(x,4))

submit.to_csv('data/submit_final.csv',index=False)