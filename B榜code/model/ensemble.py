import pandas as pd

lgb_all=pd.read_csv('data/submitB_all_feature.csv')
lgb_some=pd.read_csv('data/submitB_some_feature.csv')

#归一化
min1= lgb_all['PROB'].min()
max1= lgb_all['PROB'].max()
lgb_all['PROB'] = lgb_all['PROB'].map(lambda x:(x-min1)/(max1-min1))

min2= lgb_some['PROB'].min()
max2= lgb_some['PROB'].max()
lgb_some['PROB'] = lgb_some['PROB'].map(lambda x:(x-min2)/(max2-min2))

## 简单加权融合
submit=pd.DataFrame()
submit['ID']=lgb_all.ID
submit['PROB'] = 0.7*lgb_all['PROB'] + 0.3*lgb_some['PROB']

min3= submit['PROB'].min()
max3= submit['PROB'].max()
submit['PROB'] = submit['PROB'].map(lambda x:(x-min3)/(max3-min3))
submit.PROB=submit.PROB.apply(lambda x:round(x,4))

submit.to_csv('data/submit_final.csv',index=False)