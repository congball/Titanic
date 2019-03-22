#/usr/bin/python
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt

fig = plt.figure()
fig.set(alpha = 0.2)
data_train = pd.read_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/train.csv")

plt.subplot2grid((3,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"获救情况")
plt.ylabel(u"人数")

plt.subplot2grid((3,3),(0,1))
data_train.Pclass.value_counts().plot(kind = 'bar')
plt.title(u"乘客等级分布")
plt.ylabel(u"人数")

plt.subplot2grid((3,3),(0,2))
data_train.Sex.value_counts().plot(kind = 'bar')
plt.title(u"乘客性别统计")
plt.ylabel(u"人数")

plt.subplot2grid((3,3),(1,0))
plt.scatter( data_train.Age, data_train.Survived)
plt.xlabel(u"年龄")
plt.title(u"根据性别看获救情况")
plt.grid(b=True,which='major', axis='y')

plt.subplot2grid((3,3),(1,1),colspan=2)
data_train.Embarked.value_counts().plot(kind = 'bar')
plt.title(u"各登船口统计情况")
plt.ylabel(u"人数")


plt.subplot2grid((3,3),(2,0),colspan=3)
data_train.Age[data_train.Embarked == 'S'].plot(kind = 'kde')
data_train.Age[data_train.Embarked == 'C'].plot(kind = 'kde')
data_train.Age[data_train.Embarked == 'Q'].plot(kind = 'kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各港口乘客的年龄分布情况")
plt.legend((u'S港口',u'C港口',u'Q港口'),loc = 'best')

plt.show()



