import pandas as pd
import numpy as np
path = "../data/"
train_dir = "AI_Risk_Train_V3.0/"
test_dir = "AI_Risk_data_Btest_V2.0/"


def get_age_feat(train_dir, train):
    user_info = pd.read_csv(path+train_dir+train +
                            "_user_info.csv", low_memory=False)
    birthday_dict = pd.read_csv("utils_file/birthday.csv", header=None)
    user_info.birthday = user_info.birthday.apply(
        lambda x: str(x).split("-")[0], 1)
    user_info.birthday = user_info.birthday.apply(
        lambda x: str(x).split("鍚")[0], 1)
    birthday_dict = dict(zip(birthday_dict[0].values, birthday_dict[1].values))
    user_info.birthday.replace(birthday_dict, inplace=True)
    user_info.birthday = user_info.birthday.replace("null", 0)
    user_info.birthday = user_info.birthday.replace("nan", 0)
    user_info.birthday = user_info.birthday.replace("", 0)
    user_info.birthday = user_info.birthday.astype(int)
    user_info['age'] = user_info.birthday.apply(
        lambda x: 2017-x if x != 0 else 0, 1)
    user_info.drop('birthday', 1, inplace=True)
    return user_info[['id','age']]


train_age = get_age_feat(train_dir , "train")

test_age = get_age_feat(test_dir, "Btest")
age=pd.concat([train_age,test_age])
age.to_csv("../data/age.csv",index=False)
