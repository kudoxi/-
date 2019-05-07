import sklearn.datasets as sd #数据集
import sklearn.utils as su #辅助工具
import sklearn.tree as st
import sklearn.ensemble as se
import sklearn.metrics as sm
import numpy as np

boston = sd.load_boston()
print(boston.data.shape)
print('样本特征：',boston.feature_names)
#['CRIM' 　'ZN' 　　　'INDUS' 　　'CHAS' 　'NOX' 　　　'RM' 　　　　'AGE' 　　　'DIS' 　　　　'RAD' 　　　'TAX' 　'PTRATIO'　'B' 　　　'LSTAT']
#犯罪率　　住宅用地比例　商用地比例　　是否靠河　NO浓度　　住宅平均房间数　房间建立时间　到市中心距离　　路网密度　　房产税　　师生比例　　黑人比例　　人口地位处于中低下的比例
print('目标值：',boston.target.shape)
x = boston.data
y = boston.target
#打乱分布顺序，抽样更加有效
x = su.shuffle(x,random_state=7)
y = su.shuffle(y,random_state=7)

#划分，有些做训练，有些做测试
train_size = int(len(x) * 0.8)
train_x ,test_x,train_y,test_y = x[:train_size],x[train_size:],y[:train_size],y[train_size:]
model = st.DecisionTreeClassifier(max_depth=4)
model.fit(train_x,train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y,pred_test_y))

#模型的重要性
#决策树模型再根据信息熵减少量选择特征时，会为每个特征计算它的重要程度，即对输出构成影响的程度
model.feature_importances_
#超参数：人为事先给定，比如多项式次数，正则强度
#模型参数：通过训练找到的组成模型的最优参数
342522199405245425