#/usr/bin/python
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from pandas import DataFrame,Series


data_train = pd.read_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/train.csv")
print data_train

data_train.info()          #告诉数据中各个属性的信息。使得属性数据是否齐全一目了然

#print data_train[u'Pclass'].mea

#获救的人大概占总数的比例
data_train_Survived = data_train['Survived'].mean()
print data_train_Survived