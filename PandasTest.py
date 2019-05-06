#pandas　核心数据结构－Series－可以理解为一个一维数组
#pandas 数据结构－Dataframe

import pandas as pd
import numpy as np
from datetime import datetime
#1.series创建
s1 = pd.Series(data=[3,-5,7,4],index=list('ABCD'))#index:行索引　可以写成index＝［＇Ａ＇，＇Ｂ＇，＇Ｃ＇，＇Ｄ＇］
#__series查找
test_data1 = pd.Series(data=[90,86,70],index=['leo','kate','john'])
#＿查找＿通过绝对位置
print("leo的成绩：",test_data1[0])
#＿查找＿通过标签
print("leo的成绩：",test_data1['leo'])
#＿查找＿通过列表
print("leo ,kate 的成绩：\n",test_data1[['leo','kate']])
#＿查找＿通过表达式
print("成绩>80：\n",test_data1[test_data1>80])

#print(s1,type(s1))

#2.Dataframe 创建
data =[
    ['Belglum','Brussels',11190846],
    ['India','New Delhi',1303171035],
    ['Brazil','Brasilia',207847528],
]
df = pd.DataFrame(data=data,index=[1,2,3],columns=['Country','Capital','Population'])
#print(df)

##########练习
test_data2 = np.random.normal(1,2,5)
test_data2 = pd.Series(data=test_data2,index=list('ABCDE'))
test_data3 = pd.Series(data=[6],index=[0])#pd.Series(6)

data2 = np.random.normal(1,2,24).reshape(6,4)
#data2 = np.random.randn(6,4)
date_index = pd.date_range(start='2019-01-01',end='2019-01-6',freq='D')#
#date_index = pd.date_range(start='2019-01-01',periods=6)
test_data4 = pd.DataFrame(data=data2,index=date_index,columns=list('ABCD'))
#print(test_data4)