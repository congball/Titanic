#/usr/bin/python
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data_train = pd.read_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/train.csv")

#使用RandomForestClassifier填补缺失的年龄属性
def set_missing_ags(df):
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]            #把已有的数据型特征取出来放在Random Forest Regressor中

    #乘客的年龄分为已知、未知两个部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    #y为目标特征
    y = known_age[:,0]
    #x为特征属性值
    X = known_age[:,1:]

    #fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)

    #用得到的结果进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:,1::])

    #用预测值填补原缺失数据
    df.loc[(df.Age.isnull()),'Age'] = predictedAges

    return df,rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()),'Cabin'] = "No"
    return df

data_train,rfr = set_missing_ags(data_train)
data_train = set_Cabin_type(data_train)

#get_dummies(）完成类目型的特征因子化
#把属性全部转换成0、1属性
dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix='Pclass')
df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis=1)
df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace = True)
print df



#将Age和Fare两个属性的数值进行缩放，控制在[-1,1]之间
import sklearn.preprocessing as preprocessing

from sklearn.preprocessing import StandardScaler
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1),age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1),fare_scale_param)
#试着将转换结果存到指定的csv文件中
outfile = './data/data_preprocessing.csv'
df.to_csv(outfile)
df['Age_scaled'].to_csv(outfile,mode='a')
df['Fare_scaled'].to_csv(outfile,mode='a')                 #mode='a'表示在表格中追加，其默认为w


#逻辑回归建模

from sklearn import linear_model

#用正则取出所需的属性值

train_df = df.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin._*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

#y表示Survived结果
y = train_np[:,0]

#X即特征属性值
X = train_np[:,1:]

#fit到RandomForesetRegressor中
clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(X,y)

print clf


#对test.csv中的数据进行建模

data_test = pd.read_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/test.csv")
data_test.loc[(data_test.Fare.isnull()),'Fare'] = 0
#做特征变换
tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()

#根据特征属性X预测年龄并补上
X = null_age[:,1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()),'Age'] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'],prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'],prefix='Pclass')

df_test = pd.concat([data_test,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
df_test.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace = True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1),age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1),fare_scale_param)
outfile1 = './data/test_data_preprocessing.csv'
df_test.to_csv(outfile1)
print df_test

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
#做预测
test = df_test.filter(regex = 'Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/logistic_regression_predictions.csv",index=False)

#优化
#将得到的model系数和feature关联起来，方便查看规律
relation = pd.DataFrame({"columns":list(train_df.columns)[1:],"coef":list(clf.coef_.T)})     #结果为正则为正相关、为负则为负相关
print relation
relation.to_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/model_feature_relation.csv")

#交叉验证 cross validation
#把train.csv分成两部分，一部分用于训练所需模型，一部分用于看预测算法的效果
from sklearn.model_selection import cross_validate

#简单看看打分情况
clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
all_data = df.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin._*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.as_matrix()[:,1:]
y = all_data.as_matrix()[:,0]
#cross_validate(clf,X,y,cv=5).to_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/cross_result.csv")
print cross_validate(clf,X,y,cv=5)

from sklearn.model_selection import train_test_split

#分割数据，按照 训练数据：cv数据 = 7：3的比例
split_train,split_cv = train_test_split(df,test_size=0.3,random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin._*|Embarked_.*|Sex_.*|Pclass_.*')            #用于训练model
#生成模型
clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])

#对交叉验证数据进行预测
cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin._*|Embarked_.*|Sex_.*|Pclass_.*')              #用于评定和选择模型
predictions = clf.predict(cv_df.as_matrix()[:,1:])

origin_data_train = pd.read_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/train.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
print bad_cases
bad_cases.to_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/bad_cases.csv")

#
# #使用learning curve判断目前模型的状态
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
#
# #用learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
# def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):
#     """
#     画出data在模型上的learning curve
#     :param estimator: 使用的分类器
#     :param title: 表格的标题
#     :param X: 输的的feature,numpy类型
#     :param y: 输入的target vector
#     :param ylim: tuple格式的（ymin,ymax）,设定图像中纵坐标的最低点和最高点
#     :param cv:做交叉验证的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
#     :param n_jobs:并行的任务数（默认1）
#     :param train_sizes:
#     :param verbose:
#     :param plot:
#     :return:
#     """
#     train_sizes,train_scores,test_scores = learning_curve(
#         estimator, X, y, cv = cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose
#     )
#     train_scores_mean = np.mean(train_scores,axis=1)
#     train_scores_std = np.std(train_scores,axis=1)
#     test_scores_mean = np.mean(test_scores,axis=1)
#     test_scores_std = np.std(test_scores,axis=1)
#
#     if plot:
#         plt.figure()            #创建一个画板
#         plt.title(title)
#         if ylim is not None:
#             plt.ylim(*ylim)
#         plt.xlabel(u"训练样本数")
#         plt.ylabel(u"得分")
#         plt.gca().invert_yaxis()        #返回当前的axes  gcf()表示返回当前的数据
#         plt.grid()
#
#         plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color = 'b')
#         plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean +test_scores_std,alpha = 0.1, color = 'r')
#
#         plt.plot(train_sizes, train_scores_mean, 'o-', color = "b", label = u"训练集上得分")     #画图，并制定line的属性、图例。'o-'点线标记
#         plt.plot(train_sizes, test_scores_mean, 'o-', color = 'r', label = u"交叉验证上得分")
#
#         plt.legend(loc = "best")
#
#         plt.draw()
#         plt.show()
#         plt.gca().invert_yaxis()
#
#     midpoint = ((train_scores_mean[-1] + train_scores_std[-1] +test_scores_mean[-1] + test_scores_std[-1])) / 2
#     diff = (test_scores_mean[-1] + train_scores_std[-1] - test_scores_mean[-1] -test_scores_std[-1])
#     return midpoint,diff
# plot_learning_curve(clf,u"学习曲线",X,y)

print '>>>>>>>>>>'

from sklearn.ensemble import BaggingRegressor
train_df = df.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin._*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()

#y为Survival结果
y = train_np[:,0]

#X为特征属性值
X = train_np[:,1:]

#fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X,y)

test = df_test.filter(regex = 'Age_.*|SibSp|Parch|Fare_.*|Cabin._*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})

print result
result.to_csv("/Users/liucong/Desktop/node/Data-anlysis/Titanic/data/logistic_regression_bagging_predictions.csv",index=False)













