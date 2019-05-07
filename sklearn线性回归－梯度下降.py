import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as plt
x,y = [],[]
with open('single.txt','r') as f:
    for line in f.readlines():
        data = [float(substr) for substr in line.split(",")]
        x.append(data[:-1])
        y.append(data[-1])

x = np.array(x)
y = np.array(y)
model = lm.LinearRegression()
model.fit(x,y)
pred_y = model.predict(x)
for real,pred in zip(y,pred_y):
    print(real,'->',pred)
#平均绝对值误差
print(sm.mean_absolute_error(y,pred_y))
#平均平方误差
print(sm.mean_squared_error(y,pred_y))
#中位数绝对值误差　去除极端误差
print(sm.median_absolute_error(y,pred_y))
#R2得分　平均值不能代表全部数据，综合分散性和平均值
print(sm.r2_score(y,pred_y))
#这里面做过了换算　
#越接近于１，误差越小，越接近于０，误差越大


plt.scatter(x,y,label='real',c="dodgerblue",marker='o',alpha=0.5)
sorted_indes = x.T[0].argsort()
plt.plot(x[sorted_indes],pred_y[sorted_indes],label='pred',color="orangered")

plt.legend()
plt.show()