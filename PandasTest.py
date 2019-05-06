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
#Dataframe　操作
data2 = np.random.normal(1,2,24).reshape(6,4)#data2 = np.random.randn(6,4)
date_index = pd.date_range(start='2019-01-01',end='2019-01-6',freq='D')#date_index = pd.date_range(start='2019-01-01',periods=6)
df2 = pd.DataFrame(data=data2,index=date_index,columns=list('ABCD'))
#__基础属性＿＿
print('__形状__shape',df.shape)
#
print('__维数__ndim',df.ndim)
#
print('__元素__values',df.values)
#
print('＿索引＿index',df.index)
# 　
print('__列＿columns',df.columns)
#单列数据的访问
print('访问Country列：\n',df.Country,'访问Capital列：',df['Capital'])
#多列数据的访问
print('访问Country，Capital两列：\n',df[['Country','Capital']])
#某几行的访问
print('访问前3行：\n',df2[0:3])
print('访问前5行：\n',df2.head(5))
print('访问后２行：\n',df2.tail(2))
print('索引标签访问第３行：\n',df2.loc['2019-01-03'],'访问1~4行前２列：\n',df2.loc['2019-01-03':'2019-01-04',['A','B']])#可读性更高,推荐
print('绝对位置访问第３行：\n',df2.iloc[2],'访问1~4行前２列：\n',df2.iloc[:4,[0,1]])
print('名称＆绝对位置访问第３行：\n',df2.ix[2],'访问1~4行前２列：\n',df2.ix[:4,['A','B']])
print('2019-01-04之前的数据：\n',df2.loc[df2.index<'2019-01-04'])
#print(df)

##########练习
test_data2 = np.random.normal(1,2,5)
test_data2 = pd.Series(data=test_data2,index=list('ABCDE'))
test_data3 = pd.Series(data=[6],index=[0])#pd.Series(6)
#print(test_data4)
data3 = [
    ['Snow','M',22],
    ['Tyrion','M',32],
    ['Sansa','F',18],
    ['Arya','F',14],
]
test_data4 = pd.DataFrame(data=data3,columns=['name','gender','age'])
print('选取genger age列\n',
      test_data4[['gender','age']],"\n",
      test_data4.loc[:,['gender','age']],"\n",
      test_data4.iloc[:,[1,2]],"\n",
      test_data4.iloc[:,1:3]
      )
print('选取1～2行（标签的1和2）\n',
      test_data4[1:3],"\n",
      test_data4.loc[[1,2]],"\n",
      test_data4.iloc[[1,2]],"\n",
      test_data4.iloc[1:3]
      )
print('选取1,3行（标签的1和3）,0,1列\n',
      test_data4.iloc[[1,3],0:2],"\n",
      )
print('倒数第3行～倒数第1行，不包括最后1行：',
      test_data4[-3:-1]
      )