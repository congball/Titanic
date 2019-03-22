#/usr/bin/python
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

data_train = pd.read_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/train.csv")

# #查看乘客等级的获救情况
# fig = plt.figure()
# fig.set(alpha = 0.2)
#
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
# df.plot(kind = 'bar',stacked = True)      #累计柱状图
# plt.title(u"各乘客等级的获救情况")
# plt.xlabel(u"乘客等级")
# plt.ylabel(u"人数")
#
# plt.show()


# #查看各性别的获救情况
# fig = plt.figure()
# fig.set(alpha = 0.8)
#
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df = pd.DataFrame({u'男性':Survived_m,u'女性':Survived_f})
# df.plot(kind = 'bar',stacked = True)
# plt.title(u'不同性别的获救情况')
# plt.xlabel(u'性别')
# plt.ylabel(u'人数')
# plt.show()

# #查看各种舱级别情况下各性别的获救情况
# fig = plt.figure()
# fig.set(alpha = 0.65)
# plt.title(u"根据舱级别和性别的获救情况")
#
# ax1 = fig.add_subplot(141)          #画子图,141表示行、列、选取第几个绘图区域,可用'，'分开
# data_train.Survived[data_train.Sex =='female'][data_train.Pclass !=3].value_counts().plot(kind = 'bar')
# ax1.set_xticklabels([u"获救",u"未获救"],rotation = 0)      #设置x轴标签文字，rotation表示翻转的角度
# ax1.legend([u"女性/高级舱"],loc = 'best')
#
# ax2 = fig.add_subplot(142)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind = 'bar')
# ax2.set_xticklabels([u'获救',u'未获救'],rotation = 0)
# ax2.legend([u'女性/低级舱'],loc = 'best')
#
# ax3 = fig.add_subplot(143)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind = 'bar')
# ax3.set_xticklabels([u'获救',u'未获救'],rotation =0)
# ax3.legend([u'男性/高级舱'],loc = 'best')
#
# ax4 = fig.add_subplot(144)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind ='bar')
# ax4.set_xticklabels([u'获救',u'未获救'],rotation =0)
# ax4.legend([u'男性/低级舱'],loc = 'best')


# #各登船港口的获救情况
# fig = plt.figure()
# fig.set(alpha = 0.2)
#
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
# df.plot(kind = 'bar',stacked = True)
# plt.title(u'更登陆港口乘客的获救情况')
# plt.xlabel(u'登陆港口')
# plt.ylabel(u'人数')
#
# plt.show()
#
# #堂兄弟/妹人数对获救情况的影响
# g = data_train.groupby(['SibSp','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print df
# print ''''''
#
# #孩子/父母人数对获救的影响
# g = data_train.groupby(['Parch','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print df


#print data_train.Cabin.value_counts()

fig = plt.figure()
fig.set(alpha = 0.2)

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({u'有':Survived_cabin,u'无':Survived_nocabin}).transpose()                #transpose()
df.plot(kind = 'bar',stacked = True)
plt.title(u'按Cabin有无看获救情况')
plt.xlabel(u'Cabin有无')
plt.ylabel(u'人数')
plt.show()
