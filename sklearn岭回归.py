import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as plt
x,y = [],[]
with open('abnormal.txt','r') as f:
    for line in f.readlines():
        data = [float(substr) for substr in line.split(",")]
        x.append(data[:-1])
        y.append(data[-1])

x = np.array(x)
y = np.array(y)

plt.figure('Regression')
model = lm.LinearRegression()
model.fit(x,y)
pred_y = model.predict(x)
plt.scatter(x,y,label='real',c="dodgerblue",marker='o',alpha=0.5)
sorted_indes = x.T[0].argsort()
plt.plot(x[sorted_indes],pred_y[sorted_indes],label='pred',color="orangered")

model2 = lm.Ridge(150,fit_intercept=True,max_iter=1000)
#150　人为根据经验设置惩罚项系数，越大对权重影响越大
# fit_intercept斜率和截距都受惩罚项影响，Ｆalse为只有斜率受影响
# max_iter 最大迭代次数
model2.fit(x,y)
pred_y2 = model2.predict(x)
sorted_indes = x.T[0].argsort()
plt.plot(x[sorted_indes],pred_y2[sorted_indes],label='pred',color="limegreen")


plt.legend()
plt.show()