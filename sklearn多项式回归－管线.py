import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
x,y = [],[]
with open('single.txt','r') as f:
    for line in f.readlines():
        data = [float(substr) for substr in line.split(",")]
        x.append(data[:-1])
        y.append(data[-1])

train_x = np.array(x)
train_y = np.array(y)

model = pl.make_pipeline(
    sp.PolynomialFeatures(10),
    lm.LinearRegression()
)
model.fit(train_x,train_y)
pred_y = model.predict(train_x)
print(sm.r2_score(y,pred_y))
#这里面做过了换算　
#越接近于１，误差越小，越接近于０，误差越大


plt.scatter(train_x,train_y,label='real',c="dodgerblue",marker='o',alpha=0.5)
sorted_indes = train_x.T[0].argsort()
plt.plot(train_x[sorted_indes],pred_y[sorted_indes],label='pred',color="orangered")
#验证其他数据
test_x = np.linspace(
    train_x.min(),train_x.max(),1000
)[:,np.newaxis]
pred_test_y = model.predict(test_x)
plt.plot(test_x,pred_test_y,label='pred',color="limegreen")


plt.legend()
plt.show()