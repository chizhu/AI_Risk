import pandas as pd
import datetime
import os
import time
from config import path, train_dir, test_dir
import util

#统计银行卡和信用卡的比例
def get_bank_info(one):
    res = pd.Series()
    res['bank_count'] = len(one.bank_name.unique())
    res['chuxuka_count'] = len(one[one.card_type == '储蓄卡']) / len(one)
    res['xinyongka_count'] = len(one[one.card_type == '信用卡']) / len(one)
    return res


def order_name_fre(one):
    res = pd.Series()
    res['order_name_fre'] = len(one.name_rec_md5.unique())/len(one)
    return res


def get_dup_info(one):
    res = pd.Series()
    res['dup_num'] = len(one[one.is_dup == True])
    res['before_count'] = len(one)
    return res

def get_orderpred():
    ##产生orderpred
    orderfeature = pd.read_csv(path+'order_feature.csv', low_memory=False)
    target = pd.read_csv(path+'target.csv', low_memory=False)
    order = orderfeature.merge(target, on='id', how='left')
    trainorder = order[order.target != 2]
    testorder = order[order.target == 2]
    trainorder = trainorder.reset_index(drop=True)
    testorder = testorder.reset_index(drop=True)

    from sklearn.model_selection import StratifiedKFold

    from sklearn import linear_model
    from sklearn.tree import DecisionTreeClassifier
    skf = StratifiedKFold(n_splits=5, random_state=1111)
    X = trainorder.iloc[:, 1:95]
    y = trainorder.target
    trainorder['orderpred'] = 1
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #     clf = linear_model.Ridge(alpha=1)
        clf = DecisionTreeClassifier(random_state=0, max_depth=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]
        trainorder.loc[test_index, 'orderpred'] = y_pred
    testorder['orderpred'] = clf.predict_proba(testorder.iloc[:, 1:95])[:, 1]
    pd.concat([trainorder, testorder])[['id', 'orderpred']].to_csv(
        path+'orderpred.csv', index=False, encoding='utf-8')

def get_order_infomation(one):
    res = pd.Series()
    res['type_pay_0'] = one[one.type_pay == '在线支付'].shape[0]
    res['type_pay_1'] = one[one.type_pay == '货到付款'].shape[0]
    res['type_pay_2'] = one[one.type_pay == 'nan'].shape[0]
    res['type_pay_3'] = one[one.type_pay == '混合支付'].shape[0]
    res['type_pay_4'] = one[one.type_pay == '上门自提'].shape[0]
    res['type_pay_5'] = one[one.type_pay == '在线'].shape[0]
    res['type_pay_6'] = one[one.type_pay == '白条支付'].shape[0]
    res['type_pay_7'] = one[one.type_pay == '分期付款'].shape[0]
    res['type_pay_8'] = one[one.type_pay == '在线+京豆'].shape[0]
    res['type_pay_9'] = one[one.type_pay == '前台自付'].shape[0]
    res['type_pay_10'] = one[one.type_pay == '余额'].shape[0]
    res['type_pay_11'] = one[one.type_pay == '在线+限品东券'].shape[0]
    res['type_pay_12'] = one[one.type_pay == '京豆支付'].shape[0]
    res['type_pay_13'] = one[one.type_pay == '京豆混合支付'].shape[0]
    res['type_pay_14'] = one[one.type_pay == '邮局汇款'].shape[0]
    res['type_pay_15'] = one[one.type_pay == '东券混合支付'].shape[0]
    res['type_pay_16'] = one[one.type_pay == '公司转账'].shape[0]
    res['type_pay_17'] = one[one.type_pay == '在线+东券支付'].shape[0]
    res['type_pay_18'] = one[one.type_pay == '在线+定向东券'].shape[0]
    res['type_pay_19'] = one[one.type_pay == '定向京券支付'].shape[0]
    res['type_pay_20'] = one[one.type_pay == '京豆'].shape[0]
    res['type_pay_21'] = one[one.type_pay == '在线+东券'].shape[0]
    res['type_pay_22'] = one[one.type_pay == '在线预付'].shape[0]
    res['type_pay_23'] = one[one.type_pay == '京豆东券混合支付'].shape[0]
    res['type_pay_24'] = one[one.type_pay == '定向东券'].shape[0]
    res['type_pay_25'] = one[one.type_pay == '在线+全品东券'].shape[0]
    res['type_pay_26'] = one[one.type_pay == '京券混合支付'].shape[0]
    res['type_pay_27'] = one[one.type_pay == '在线+余额'].shape[0]
    res['type_pay_28'] = one[one.type_pay == '余额+限品东券'].shape[0]
    res['type_pay_29'] = one[one.type_pay == '在线+限品京券'].shape[0]
    res['type_pay_30'] = one[one.type_pay == '在线+全品京券'].shape[0]
    res['type_pay_31'] = one[one.type_pay == '在线+京券支付'].shape[0]
    res['type_pay_32'] = one[one.type_pay == '积分支付'].shape[0]
    res['type_pay_33'] = one[one.type_pay == '定向京券'].shape[0]
    res['type_pay_34'] = one[one.type_pay == '全品京券'].shape[0]
    res['type_pay_35'] = one[one.type_pay == '京券全额支付'].shape[0]
    res['type_pay_36'] = one[one.type_pay == '限品京券'].shape[0]
    res['type_pay_37'] = one[one.type_pay == '在线+余额+限品东券'].shape[0]
    res['type_pay_38'] = one[one.type_pay == '高校代理-自己支付'].shape[0]
    res['type_pay_39'] = one[one.type_pay == '高校代理-代理支付'].shape[0]
    res['type_pay_40'] = one[one.type_pay == '分期付款(招行)'].shape[0]

    res['sts_order_0'] = one[one.sts_order == '等待收货'].shape[0]
    res['sts_order_1'] = one[one.sts_order == '完成'].shape[0]
    res['sts_order_2'] = one[one.sts_order == 'nan'].shape[0]
    res['sts_order_3'] = one[one.sts_order == '已完成'].shape[0]
    res['sts_order_4'] = one[one.sts_order == '充值失败退款成功'].shape[0]
    res['sts_order_5'] = one[one.sts_order == '已取消'].shape[0]
    res['sts_order_6'] = one[one.sts_order == '订单取消'].shape[0]
    res['sts_order_7'] = one[one.sts_order == '充值成功'].shape[0]
    res['sts_order_8'] = one[one.sts_order == '未抢中'].shape[0]
    res['sts_order_9'] = one[one.sts_order == '商品出库'].shape[0]
    res['sts_order_10'] = one[one.sts_order == '出票成功'].shape[0]
    res['sts_order_11'] = one[one.sts_order == '退款完成'].shape[0]
    res['sts_order_12'] = one[one.sts_order == '正在出库'].shape[0]
    res['sts_order_13'] = one[one.sts_order == '等待付款确认'].shape[0]
    res['sts_order_14'] = one[one.sts_order == '充值失败'].shape[0]
    res['sts_order_15'] = one[one.sts_order == '等待审核'].shape[0]
    res['sts_order_16'] = one[one.sts_order == '付款成功'].shape[0]
    res['sts_order_17'] = one[one.sts_order == '退款成功'].shape[0]
    res['sts_order_18'] = one[one.sts_order == '等待付款'].shape[0]
    res['sts_order_19'] = one[one.sts_order == '正在处理'].shape[0]
    res['sts_order_20'] = one[one.sts_order == '配送退货'].shape[0]
    res['sts_order_21'] = one[one.sts_order == '出票失败'].shape[0]
    res['sts_order_22'] = one[one.sts_order == '已晒单'].shape[0]
    res['sts_order_23'] = one[one.sts_order == '等待处理'].shape[0]
    res['sts_order_24'] = one[one.sts_order == '已入住'].shape[0]
    res['sts_order_25'] = one[one.sts_order == '等待退款'].shape[0]
    res['sts_order_26'] = one[one.sts_order == '抢票已取消'].shape[0]
    res['sts_order_27'] = one[one.sts_order == '请上门自提'].shape[0]
    res['sts_order_28'] = one[one.sts_order == '已收货'].shape[0]
    res['sts_order_29'] = one[one.sts_order == '缴费成功'].shape[0]
    res['sts_order_30'] = one[one.sts_order == '已退款'].shape[0]
    res['sts_order_31'] = one[one.sts_order == '订单已取消'].shape[0]
    res['sts_order_32'] = one[one.sts_order == '预约完成'].shape[0]
    res['sts_order_33'] = one[one.sts_order == '发货中'].shape[0]
    res['sts_order_34'] = one[one.sts_order == '失败退款'].shape[0]
    res['sts_order_35'] = one[one.sts_order == '已确认'].shape[0]
    res['sts_order_36'] = one[one.sts_order == '预订结束'].shape[0]
    res['sts_order_37'] = one[one.sts_order == '下单失败'].shape[0]
    res['sts_order_38'] = one[one.sts_order == '部分充值成功'].shape[0]
    res['sts_order_39'] = one[one.sts_order == '未入住'].shape[0]
    res['sts_order_40'] = one[one.sts_order == '正在送货（暂不能上门自提）'].shape[0]
    res['sts_order_41'] = one[one.sts_order == '过期放弃'].shape[0]
    res['sts_order_42'] = one[one.sts_order == '购买成功'].shape[0]
    res['sts_order_43'] = one[one.sts_order == '已取消订单'].shape[0]
    res['sts_order_44'] = one[one.sts_order == '正在充值'].shape[0]
    res['sts_order_45'] = one[one.sts_order == '等待揭晓'].shape[0]
    res['sts_order_46'] = one[one.sts_order == '预订中'].shape[0]
    res['sts_order_47'] = one[one.sts_order == '等待厂商处理'].shape[0]
    res['sts_order_48'] = one[one.sts_order == '退款中'].shape[0]
    res['sts_order_49'] = one[one.sts_order == '商品退库'].shape[0]
    res['sts_order_50'] = one[one.sts_order == '等待分期付款'].shape[0]
    res['sts_order_51'] = one[one.sts_order == '过期关闭'].shape[0]
    res['sts_order_52'] = one[one.sts_order == '支付失败'].shape[0]
    return res

# 将time_order 转化成时间 并且提取 年 月 日
def timestamp_datetime(value):
    for_mat = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(for_mat, value)
    return dt

#统计特征

def get_order_feature(one):
    res = pd.Series()
    #订单金额
    res['amt_order_sum'] = one.amt_order.sum()
    res['amt_order_mean'] = one.amt_order.mean()
    res['amt_order_std'] = one.amt_order.std()
    res['amt_order_skew'] = one.amt_order.skew()

    ##手机号码个数
    res['phone_num'] = len(one.phone.unique())

    #订单计数特征
    length = len(one)
    res['order_count'] = length
    res['dur_day'] = one.time_order.max()-one.time_order.min()   # 最大订单时间间隔
    res['morning_order_count'] = len(
        one[(one.hour < 11) & (one.hour > 6)]) / length  # 早上次数
    res['afternoon_order_count'] = len(
        one[(one.hour < 17) & (one.hour >= 11)])/length  # 下午次数
    res['evening_order_count'] = len(
        one[(one.hour <= 23) & (one.hour >= 17)])/length  # 晚上次数
    res['night_order_count'] = len(
        one[(one.hour == 24) | (one.hour <= 6)])/length  # 半夜次数
    res['weekday_count'] = len(one[one.weekday < 5])/length  # 工作日的次数
    res['weekend_count'] = len(one[one.weekday >= 5])/length  # 周末的次数
    res['weekday_mode'] = one.weekday.mode().iloc[0]  # 周几的众数

    #获取每个id里面的product_id为空的次数
    res['product_id_null_num'] = len(one[one.product_id_md5.isnull()])/len(one)
    #获取缺失值的比例
    res['order_null_avg'] = one.null_count.sum()/len(one)

    return res



def get_auth_feat():
    print('auth')
    train_auth_info = pd.read_csv(
        path+train_dir+'train_auth_info.csv', low_memory=False)


    test_auth_info = pd.read_csv(
        path+test_dir+'Btest_auth_info.csv', low_memory=False)
    train_auth_info.auth_time = train_auth_info.auth_time.replace(
        '0000-00-00', '2017-01-01')
    test_auth_info.auth_time = test_auth_info.auth_time.replace(
        '0000-00-00', '2017-01-01')
    target_info = pd.read_csv(path+'target.csv', low_memory=False)

    auth_info = pd.concat([train_auth_info, test_auth_info], ignore_index=True)


    #年
    auth_info['auth_time_year'] = -1
    auth_info.loc[auth_info.auth_time.notnull(), 'auth_time_year'] = auth_info[auth_info.auth_time.notnull(
    )]['auth_time'].apply(lambda x: int(x.split('-')[0]))
    #月
    auth_info['auth_time_month'] = -1
    auth_info.loc[auth_info.auth_time.notnull(), 'auth_time_month'] = auth_info[auth_info.auth_time.notnull(
    )]['auth_time'].apply(lambda x: int(x.split('-')[1]))
    #日
    auth_info['auth_time_day'] = -1
    auth_info.loc[auth_info.auth_time.notnull(), 'auth_time_day'] = auth_info[auth_info.auth_time.notnull(
    )]['auth_time'].apply(lambda x: int(x.split('-')[2]))
    #星期几
    auth_info['auth_time_weekday'] = -1
    auth_info.loc[auth_info.auth_time.notnull(), 'auth_time_weekday'] = auth_info[auth_info.auth_time.notnull(
    )]['auth_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').weekday())  # 星期几
    #身份证是否存在
    auth_info['id_card_exist'] = 1
    auth_info.loc[auth_info.id_card.isnull(), 'id_card_exist'] = 0
    #认证时间是否存在
    auth_info['auth_time_exist'] = 1
    auth_info.loc[auth_info.auth_time.isnull(), 'auth_time_exist'] = 0
    # 电话是否存在
    auth_info['phone_exist'] = 1
    auth_info.loc[auth_info.phone.isnull(), 'phone_exist'] = 0

    #身份证号码的第一位
    auth_info['id_card'].fillna(0, inplace=True)
    auth_info['idcard_first'] = auth_info['id_card'].apply(lambda x: str(x)[:1])

    #id的长度
    auth_info['id_len'] = auth_info.id.apply(lambda x: len(str(x)), 1)

    #讲电话号码前三位转换成运营生id     1移动，2联通  3.电信
    phone_dict = pd.read_csv(path+'phone_dict.csv', index_col='number')
    auth_info['phone_head3'] = 0
    auth_info.loc[auth_info.phone.notnull(
    ), 'phone_head3'] = auth_info[auth_info.phone.notnull()]['phone'].apply(lambda x: x[:3])
    auth_info['phone_head3'] = auth_info['phone_head3'].astype(
        'int').map(phone_dict.to_dict()['type'])

    #appl时间和 auth 差值得天数特征
    auth_info = pd.merge(auth_info, target_info, on='id', how='left')
    auth_info['appl_sbm_tm'] = auth_info['appl_sbm_tm'].apply(
        lambda x: str(x)[:-10])
    auth_info['appl_sbm_tm'] = pd.to_datetime(auth_info['appl_sbm_tm'])
    auth_info['auth_time'] = pd.to_datetime(auth_info['auth_time'])
    auth_info['appl_auth_time'] = (auth_info['appl_sbm_tm']-auth_info['auth_time'])
    auth_info['appl_auth_time'] = auth_info['appl_auth_time'].apply(
        lambda x: x.days)

    auth_info.idcard_first = auth_info.idcard_first.astype('int')
    auth_info.idcard_first.replace(0, -1, inplace=True)
    auth_info.to_csv(path+'auth.csv', index=False)
    print('auth finish')

def get_bank_feat():
    train_bank = pd.read_csv(
        path+train_dir+'train_bankcard_info.csv', low_memory=False)
    test_bank = pd.read_csv(
        path+test_dir+'Btest_bankcard_info.csv', low_memory=False)
    #B 榜多出来一个owner_name字段
    # test_bank.drop('owner_name',1,inpladatce=True)
    target=pd.read_csv(path+'target.csv',low_memory=False)

    bank=pd.concat([train_bank,test_bank],ignore_index=False)
    bank_dict=pd.read_csv(path+"bank_dict.csv",header=None)  ##中间数据，把同一个银行的不同名称统一
    bank_dict=dict(zip(bank_dict[0].values,bank_dict[1].values))
    bank.bank_name.replace(bank_dict,inplace=True)
    bank_name = bank.bank_name.unique()
    bank_name_dict = dict(zip(bank_name, range(len(bank_name))))
    bank['bank_id']=bank.bank_name.replace(bank_name_dict)


    bank=bank.drop_duplicates()
    bank_feature1=bank.groupby('id').apply(get_bank_info)
    bank_feature1.to_csv(path+'bankfeature1.csv',index=True)

    bank=pd.merge(bank,pd.get_dummies(bank['bank_name']),left_index=True,right_index=True)
    bank_onehot=bank.groupby('id')['上海农村商业银行',
    '上海银行',
    '东莞农村商业银行',
    '东营银行',
    '中信银行',
    '中国邮政',
    '中国银行',
    '临商银行',
    '九江银行',
    '云南省农村信用社',
    '交通银行',
    '光大银行',
    '兰州银行',
    '兴业银行',
    '内蒙古农村信用社',
    '农业银行',
    '包商银行',
    '北京农村商业银行',
    '北京银行',
    '华夏银行',
    '华融湘江银行',
    '南京银行',
    '南昌银行',
    '厦门银行',
    '台州银行',
    '吉林省农村信用社',
    '吉林银行',
    '吴江农村商业银行',
    '哈尔滨银行',
    '四川省农村信用社',
    '大连银行',
    '天津农商银行',
    '威海市商业银行',
    '宁夏银行',
    '宁波银行',
    '山西省农村信用社',
    '工商银行',
    '平安银行',
    '平顶山银行',
    '广东南粤银行',
    '广发银行',
    '广州农村商业银行',
    '广州市农村信用社',
    '广西农村信用社',
    '广西省农村信用社',
    '建设银行',
    '张家口银行',
    '张家港农商银行',
    '德州银行',
    '德阳银行',
    '徽商银行',
    '恒丰银行',
    '成都农商银行',
    '成都银行',
    '承德银行',
    '抚顺银行',
    '招商银行',
    '攀枝花市商业银行',
    '新疆农村信用社联合社',
    '日照银行',
    '昆仑银行',
    '昆山农商银行',
    '晋中银行',
    '晋城银行',
    '未知',
    '杭州银行',
    '桂林银行',
    '武汉农村商业银行',
    '民生银行',
    '汉口银行',
    '江南农村商业银行',
    '江苏农村商业银行',
    '江苏省农村信用社联合社',
    '江苏银行',
    '江苏长江银行',
    '江西银行',
    '江阴农商银行',
    '河北省农村信用社',
    '河南省农村信用社',
    '泉州银行',
    '泰安市商业银行',
    '洛阳银行',
    '济宁银行',
    '浙商银行',
    '浙江泰隆银行',
    '浦发银行',
    '海南省农村信用社',
    '深圳农村商业银行',
    '深圳发展银行',
    '渤海银行',
    '温州银行',
    '湖北省农村信用社',
    '湖北省农村信用社联合社',
    '湖北银行',
    '湖南农村信用社',
    '潍坊银行',
    '烟台银行',
    '甘肃省农村信用社',
    '盘锦银行',
    '盛京银行',
    '莱商银行',
    '营口银行',
    '西安银行',
    '贵州省农村信用社',
    '贵阳银行',
    '辽宁农村信用社',
    '邢台银行',
    '郑州银行',
    '鄂尔多斯银行',
    '鄞州银行',
    '重庆农村商业银行',
    '铁岭银行',
    '锦州银行',
    '长安银行',
    '陕西省农村信用社',
    '陕西省农村信用社联合社',
    '青岛银行',
    '青海省农村信用社',
    '黄河农村商业银行',
    '黑龙江农村信用社',
    '齐商银行',
    '齐鲁银行',
    '龙江银行',
    '韩亚银行',
    '浙江农信银行',
    '中原银行',
    'bokl'].sum()
    bank_onehot['id']=bank_onehot.index
    bank_onehot.reset_index(drop=True,inplace=True)

    bank_onehot=pd.merge(bank_onehot,target,on='id',how='left')
    bank_onehot.drop('appl_sbm_tm',1,inplace=True)
    trainbank=bank_onehot[bank_onehot.target!=2]
    testbank=bank_onehot[bank_onehot.target==2]
    trainbank=trainbank.reset_index(drop=True)
    testbank=testbank.reset_index(drop=True)

    #生成bankpred
    from sklearn.model_selection import StratifiedKFold
    from sklearn import linear_model
    from sklearn.tree import DecisionTreeClassifier
    skf = StratifiedKFold(n_splits=5,random_state=1111)
    X=trainbank.iloc[:,:-2]
    y= trainbank.target
    trainbank['bankpred']=0
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
    #     clf = linear_model.Ridge(alpha=1)
        clf=DecisionTreeClassifier(random_state=0,max_depth=7)
        clf.fit(X_train,y_train)
        y_pred=clf.predict_proba(X_test)[:,1]
        trainbank.loc[test_index,'bankpred']=y_pred
    testbank['bankpred']=clf.predict_proba(testbank.iloc[:,:-2])[:,1]
    pd.concat([trainbank,testbank])[['id','bankpred']].to_csv(path+'bankpred.csv',index=False,encoding='utf-8')


    ##计算该贷款id是第几次贷款
    train_bank=train_bank.drop(train_bank[train_bank.bank_name.isnull()].index)
    train_bank=train_bank.drop_duplicates()
    a=train_bank.groupby(['bank_name','tail_num','card_type','phone'],as_index=False)['id'].count()
    t1=pd.merge(train_bank,a,on=['bank_name','tail_num','card_type','phone'],how='left')
    t1=t1.drop_duplicates(['id_x','id_y'])
    t1=t1.sort_values(by='id_y')
    t1=t1.drop_duplicates('id_x',keep='last')

    test_bank=test_bank.drop(test_bank[test_bank.bank_name.isnull()].index)
    test_bank=test_bank.drop_duplicates()
    b=test_bank.groupby(['bank_name','tail_num','card_type','phone'],as_index=False)['id'].count()
    t2=pd.merge(test_bank,b,on=['bank_name','tail_num','card_type','phone'],how='left')
    t2=t2.drop_duplicates(['id_x','id_y'])
    t2=t2.sort_values(by='id_y')
    t2=t2.drop_duplicates('id_x',keep='last')


    t3 = pd.concat([t1,t2],ignore_index=True)
    t3.rename(columns={'id_x':'id','id_y':'loan_count'},inplace=True)
    t3=t3[['id','loan_count']]

    t3=pd.merge(target,t3,on='id',how='left')
    t3=t3.fillna(0)
    t3[['id','loan_count']].to_csv('data/loan_count.csv',index=False)

def get_order_feat():
    order_train = pd.read_csv(path+train_dir+'train_order_info.csv')


    order_test = pd.read_csv(path+test_dir+'Btest_order_info.csv')
    order_info = pd.concat([order_train, order_test], ignore_index=True)

    target = pd.read_csv(path+'target.csv', low_memory=False)

    #将target合并在order表里面
    order_info = pd.merge(order_info, target, on='id', how='left')

    #将 申请时间 小于 订单时间 的数据剔除掉

    order_info = order_info.drop(order_info[order_info.time_order.isnull()].index)

    order_info['time_length'] = order_info.time_order.apply(
        lambda x: len(x.split(' ')))
    order_info1 = order_info[order_info.time_length == 1]
    order_info2 = order_info[order_info.time_length == 2]
    order_info1 = order_info1[order_info1.time_order != '0']

    order_info1['appl_sbm_tm'] = order_info1['appl_sbm_tm'].apply(lambda x: str(x)[
                                                                :-2])
    order_info1['appl_sbm_tm'] = order_info1['appl_sbm_tm'].apply(
        lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    order_info1['time_order'] = order_info1['time_order'].apply(lambda x: int(x))

    order_info2['appl_sbm_tm'] = order_info2['appl_sbm_tm'].apply(lambda x: str(x)[
                                                                :-2])
    order_info2['time_order'] = order_info2['time_order'].apply(
        lambda x: time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S")))
    order_info2['appl_sbm_tm'] = order_info2['appl_sbm_tm'].apply(
        lambda x: time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S")))

    order_info1 = order_info1[order_info1.time_order < order_info1.appl_sbm_tm]
    order_info2 = order_info2[order_info2.time_order < order_info2.appl_sbm_tm]

    order_info = pd.concat([order_info1, order_info2], ignore_index=True)

    #将时间小于2010年的数据剔除掉
    order_info = order_info[order_info.time_order > 1262275200]

    order_feature = order_info.groupby('id').apply(get_order_infomation)


    order_feature.to_csv(path+'order_feature.csv', index=True)
    ##产生orderpred
    get_orderpred()

    order_name_frequency = order_info.groupby('id').apply(order_name_fre)


    order_name_frequency.to_csv(path+'order_name_fre.csv', index=True)


    #计算重复次数
    order_train['is_dup'] = order_train.duplicated(
        ['amt_order', 'type_pay', 'time_order', 'phone', 'no_order_md5'])
    order_test['is_dup'] = order_test.duplicated(
        ['amt_order', 'type_pay', 'time_order', 'phone', 'no_order_md5'])
    
    train_order_f = order_train.groupby('id').apply(get_dup_info)


    test_order_f = order_test.groupby('id').apply(get_dup_info)

    order_dup = pd.concat([train_order_f, test_order_f], ignore_index=False)
    order_dup['id'] = order_dup.index
    order_dup.to_csv(path+'order_dup.csv', index=False)


    #去掉重复的数据
    order_info = order_info.drop_duplicates(
        ['amt_order', 'time_order', 'phone', 'id', 'no_order_md5'])
    #每条记录缺失值个数
    order_info['null_count'] = order_info.isnull().sum(1).astype(int)

    ###计算最后一次消费时间和申请时间的间隔
    order_info['time_dur'] = order_info['appl_sbm_tm']-order_info['time_order']
    order_time_dur = order_info.groupby('id')['time_dur'].min()
    order_time_dur = pd.DataFrame(order_time_dur)
    order_time_dur['id'] = order_time_dur.index
    order_time_dur.to_csv(path+'order_time_dur.csv', index=False)

    order_info['time_order'] = order_info.time_order.apply(
        lambda x: timestamp_datetime(int(x)))


    order_info['year'] = order_info.time_order.apply(lambda x: int(str(x)[0:4]))
    order_info['month'] = order_info.time_order.apply(lambda x: int(str(x)[5:7]))
    order_info['day'] = order_info.time_order.apply(lambda x: int(str(x)[8:10]))
    order_info['hour'] = order_info.time_order.apply(lambda x: int(str(x)[11:13]))
    order_info['time_order'] = order_info['time_order'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    order_info['weekday'] = order_info.time_order.apply(lambda x: x.weekday())
    order_feature = order_info.groupby('id').apply(get_order_feature)
    order_feature.to_csv(path+'order_feature.csv', index=True)
def get_user_credit_feat():
    print('user')


    ##
    #  USER_INFO表
    ##
    train_user_info = pd.read_csv(
        path+train_dir+'train_user_info.csv', low_memory=False)
    test_user_info = pd.read_csv(
        path+test_dir+'Btest_user_info.csv', low_memory=False)
    user_info = pd.concat([train_user_info, test_user_info], ignore_index=True)

    #年龄  这是中间文件
    age = pd.read_csv(path+'age.csv', low_memory=False)
    user_info['sex'] = user_info['sex'].map({'保密': 0, '男': 1, '女': 2})
    user_info['qq_bound'] = user_info['qq_bound'].map({'已绑定': 1, '未绑定': 0})
    user_info['wechat_bound'] = user_info['wechat_bound'].map({'已绑定': 1, '未绑定': 0})
    user_info['account_grade'] = user_info['account_grade'].map(
        {'注册会员': 0, '铜牌会员': 1, '银牌会员': 2, '金牌会员': 3, '钻石会员': 4})

    user_info = pd.merge(user_info, age, on='id', how='left')
    user_info['null_count'] = user_info.isnull().sum(1)

    user_info = user_info[['id', 'sex', 'qq_bound',
                        'wechat_bound', 'account_grade', 'age', 'null_count']]
    user_info.to_csv(path+'user.csv', index=False)
    print('user finish')


    print('credit')
    ###
    #
    #CREDIT 表
    ###
    import pandas as pd
    train_credit_info = pd.read_csv(path+train_dir+'train_credit_info.csv')
    test_credit_info = pd.read_csv(path+test_dir+'Btest_credit_info.csv')

    credit_info = pd.concat(
        [train_credit_info, test_credit_info], ignore_index=True)
    credit_info['remain'] = credit_info['quota']-credit_info['overdraft']
    credit_info['use_rate'] = credit_info['overdraft'] / credit_info['quota']
    credit_info.to_csv(path+'credit.csv', index=False, encoding='utf-8')
    print('credit finish')



                
