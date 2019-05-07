import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sklearn.linear_model as lm

train_x = np.array([0.5,0.6,0.8,1.1,1.4])
train_y = np.array([5.0,5.5,6.0,6.8,7.0])

n_epoches = 1000#迭代次数
lrate = 0.01 #学习率
epoches,losses = [],[]
w0,w1 = [1],[1]
for epoch in range(1,n_epoches+1):
    epoches.append(epoch)
    #用最新加入的w0,w1计算损失值
    single_loss = (train_y - (w0[-1]+w1[-1]*train_x))**2
    total_loss = single_loss.sum() / 2
    #求w0 w1偏导
    dloss_w0 = -(train_y - (w0[-1]+w1[-1]*train_x)).sum()
    dloss_w1 = -((train_y - (w0[-1]+w1[-1]*train_x))*train_x).sum()
    #print('{:4}>w0={:.8f},w1={:.8f},loss={:.8f}'.format(epoch,w0[-1],w1[-1],total_loss))
    #下一个 w0,w1
    w0.append(w0[-1] - lrate*dloss_w0)
    w1.append(w1[-1] - lrate*dloss_w1)
    losses.append(total_loss)

w0 = np.array(w0[:-1])
w1 = np.array(w1[:-1])
sorted_indies = train_x.argsort()#将x中的元素从小到大排列，提取其对应的index  
test_x = train_x[sorted_indies]
test_y = train_y[sorted_indies]
predict_test_y = w0[-1] + w1[-1] * test_x

plt.grid(linestyle=":")
plt.scatter(train_x,train_y,c='dodgerblue',alpha=0.5,s=80,marker='s',label="training data")
plt.scatter(test_x,test_y,c='orangered',alpha=0.5,s=60,marker='D',label="test data")
plt.scatter(test_x,predict_test_y,c="red",alpha=0.5,marker='D',label="predict line")
plt.plot(test_x,predict_test_y,c="red",alpha=0.5,label="predict line")

#连接原数据点到预测点
for x,y,pred_y in zip(test_x,test_y,predict_test_y):
    plt.plot([x,x],[y,pred_y],c="limegreen",alpha=0.5)

plt.legend(loc="upper left")

#训练过程图
plt.figure("Training Process",facecolor='lightgray')
plt.subplot(311)
plt.title('Training Process',fontsize=20)
plt.ylabel('w0',fontsize=14)
plt.xlabel('training times',fontsize=14)
plt.tick_params(labelsize=10)
plt.grid(linestyle=":")
plt.plot(epoches,w0,c='dodgerblue',label='w0')

plt.subplot(312)
plt.ylabel('w1',fontsize=14)
plt.xlabel('training times',fontsize=14)
plt.tick_params(labelsize=10)
plt.grid(linestyle=":")
plt.plot(epoches,w1,c='limegreen',label='w1')

plt.subplot(313)
plt.ylabel('loss',fontsize=14)
plt.xlabel('training times',fontsize=14)
plt.tick_params(labelsize=10)
plt.grid(linestyle=":")
plt.plot(epoches,losses,c='orangered',label='loss')

#w0 w1 loss三轴画面
#画出w0在0~9范围内，w1在0~3.5范围内的数据点
grid_w0,grid_w1 = np.meshgrid(
    np.linspace(0,9,500),
    np.linspace(0,3.5,500))
#扁平化处理，方便带入方程
flat_w0,flat_w1 = grid_w0.ravel(),grid_w1.ravel()
#   w1 w1 w1 w1 w1 ...w1 500个　
#   w0 w0 w0 w0 w0 ...w0　500个 loss
#x1  y' y' y' y' y' ...y'  y    y-y'
#x2  y' y' y' y' y' ...y'  y    y-y'
#x3  y' y' y' y' y' ...y'  y    y-y'
#... ...　...　...　...
pred_y = flat_w0 + np.outer(train_x,flat_w1)#外积
oneline_loss = (pred_y - train_y.reshape(-1,1))**2
flat_loss = oneline_loss.sum(axis=0) / 2
grid_loss = flat_loss.reshape(grid_w1.shape)#转点阵

plt.figure('Loss Function')
ax = plt.gca(projection='3d')
plt.title('Loss Function',fontsize=20)
ax.set_xlabel('w0',fontsize=14)
ax.set_ylabel('w1',fontsize=14)
ax.set_zlabel('loss',fontsize=14)
plt.tick_params(labelsize=10)
ax.plot_surface(grid_w0,grid_w1,grid_loss,
                rstride=10,cstride=10,cmap='jet')
ax.plot(w0,w1,losses,'o-',c='orangered',label='BGD')

plt.legend(loc='lower left')

plt.tight_layout()
plt.show()
