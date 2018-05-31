import pandas as pd
import numpy as np
path = "../data/"
train_dir = "AI_Risk_Train_V3.0/"
test_dir = "AI_Risk_data_Btest_V2.0/"
import os

# train
receive = pd.read_csv(path+train_dir+"train" +
                      "_recieve_addr_info.csv", encoding="utf-8")
# receive_group = receive.groupby('id', as_index=False).count()[['id', 'phone']]
receive['region'] = receive.region.apply(lambda x: str(x)[:2])
receive['region'].replace("na", 0, inplace=True)

receive.fillna(0, inplace=True)
receive.region.replace(0, np.nan, inplace=True)
print("sss")
receive_temp = receive.groupby(['id', 'region'], as_index=False).count()
receive_temp = receive_temp.sort_values(by=["id", "addr_id"])

receive_temp = receive_temp.groupby('id', as_index=False).last()[
    ['id', 'region']]
# receive_group = pd.merge(receive_group, receive_temp, on='id', how="left")
receive_group=receive_temp

user_info = pd.read_csv(path+train_dir+"train"+"_user_info.csv")
auth_info = pd.read_csv(path+train_dir+"train"+"_auth_info.csv")

info = pd.merge(user_info, auth_info, on=['id'], how="left")
info.id_card_x.fillna(0, inplace=True)

info.id_card_y = info.apply(
    lambda x: x['id_card_x'] if x['id_card_x'] != 0 else x['id_card_y'], 1)

info = pd.merge(info, receive_group, on='id', how="left")
info['id_card_y'] = info['id_card_y'].fillna(0)

info['id_len'] = info.id_card_y.apply(lambda x: len(str(x).split("*")[0]), 1)


info['id_card_y'] = info.id_card_y.apply(lambda x: str(x).split("*")[0][:2], 1)

info['region'] = info.apply(lambda x: int(
    x['id_card_y'])if x['id_len'] >= 2 else x['region'], 1)

id_card_dict = pd.read_csv("utils_file/id_card.csv", header=None)
id_card_dict = dict(zip(id_card_dict[1].values, id_card_dict[0].values))
info.region.replace(id_card_dict, inplace=True)

train_target = pd.read_csv(path+train_dir+"train"+"_target.csv")
info = pd.merge(info, train_target, on='id', how="left")

tm = info.groupby('region', as_index=False).sum()[['region', 'target']]
tm['count'] = info.groupby('region', as_index=False).count()['target']

tm['overtime_delta'] = tm['target']/tm['count']

pro_train = pd.merge(info, tm, on='region', how="left")[
    ['id', 'region', 'overtime_delta']]
pro_train.to_csv("province_train.csv", index=False)
print("train done")


# test


receive = pd.read_csv(path+test_dir+"Btest"+"_recieve_addr_info.csv")
receive_group = receive.groupby('id', as_index=False).count()[['id', 'phone']]
receive['region'] = receive.region.apply(lambda x: str(x)[:2])
receive['region'].replace("na", 0, inplace=True)

receive.fillna(0, inplace=True)
receive.region.replace(0, np.nan, inplace=True)

receive_temp = receive.groupby(['id', 'region'], as_index=False).count()

receive_temp = receive_temp.sort_values(by=["id", "addr_id"])

receive_temp = receive_temp.groupby('id', as_index=False).last()[
    ['id', 'region']]
receive_group = pd.merge(receive_group, receive_temp, on='id', how="left")

user_info = pd.read_csv(path+test_dir+"Btest"+"_user_info.csv")
auth_info = pd.read_csv(path+test_dir+"Btest"+"_auth_info.csv")

info = pd.merge(user_info, auth_info, on=['id'], how="left")

info.id_card_x.fillna(0, inplace=True)

info.id_card_y = info.apply(
    lambda x: x['id_card_x'] if x['id_card_x'] != 0 else x['id_card_y'], 1)

info = pd.merge(info, receive_group, on='id', how="left")
info['id_card_y'] = info['id_card_y'].fillna(0)

info['id_len'] = info.id_card_y.apply(lambda x: len(str(x).split("*")[0]), 1)


info['id_card_y'] = info.id_card_y.apply(lambda x: str(x).split("*")[0][:2], 1)

info['region'] = info.apply(lambda x: int(
    x['id_card_y'])if x['id_len'] >= 2 else x['region'], 1)

id_card_dict = pd.read_csv("utils_file/id_card.csv", header=None)
id_card_dict = dict(zip(id_card_dict[1].values, id_card_dict[0].values))
info.region.replace(id_card_dict, inplace=True)

pro_test = pd.merge(info, tm, on='region', how="left")[
    ['id', 'region', 'overtime_delta']]

pro_test.to_csv("province_test.csv", index=False)

print("test done")
province=pd.concat([pro_train,pro_test])
province.to_csv("../data/province.csv",index=False)


