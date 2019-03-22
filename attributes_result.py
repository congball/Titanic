#/usr/bin/python
#-*- coding:utf-8 -*-

#乘客属性分布
#用图显示属性和结果之间的关系

import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt


fig = plt.figure()
fig.set(alpha = 0.2)     #设定图表颜色alpha参数
data_train = pd.read_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/train.csv")

plt.subplot2grid((2,3),(0,0))               #在一张大图里面分列几个小图
data_train.Survived.value_counts().plot(kind = 'bar')       #柱状图
plt.title(u"获救情况(1为获救)")        #标题
plt.ylabel(u"人数")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title(u"乘客等级分布")
plt.ylabel(u"人数")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived,data_train.Age)          #散点图
plt.title(u"按年龄看获救分布（1为获救）")
plt.ylabel(u"年龄")
plt.grid(b=True,which='major',axis='y')     #网格

plt.subplot2grid((2,3),(1,0),colspan=2)     #colspan定义单元格横跨的列数
data_train.Age[data_train.Pclass == 1].plot(kind = 'kde')           #密度图
data_train.Age[data_train.Pclass == 2].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 3].plot(kind = 'kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱',u'2等舱',u'3等舱'),loc='best')   #legend()显示图例，loc表示图例的位置

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind = 'bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")

plt.show()



