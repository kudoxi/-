
from __future__ import unicode_literals
import numpy as np
import sklearn.preprocessing as sp
raw_samples = np.array([
     [3, -1.5, 2,   -5.4],
     [0,  4,   -0.3, 2.1],
     [1,  3.3, -1.9, -4.3]])
print(raw_samples)
print(raw_samples.mean(axis=0))
print(raw_samples.std(axis=0))

std_samples = raw_samples.copy()
mms_samples = raw_samples.copy()
# print(std_samples.T)
# for col in std_samples.T:#得到每一列
#     col_mean = col.mean()
#     col_std = col.std()
#     col -= col_mean
#     col /= col_std
# print(std_samples)
# print(std_samples.mean(axis=0))
# print(std_samples.std(axis=0))
#均值移除
std_samples = sp.scale(raw_samples)
print(std_samples)
print(std_samples.mean(axis=0))
print(std_samples.std(axis=0))

for col in mms_samples.T:#得到每一列
    col_min = col.min()
    col_max = col.max()
    A = np.array([
        [col_min,1],
        [col_max,1],
    ])
    B = np.array([0,1])
    x = np.linalg.solve(A,B) #k = x[0] b = x[1]
    # y=kx +b
    col *= x[0]
    col += x[1]
print(mms_samples)
#范围缩放
mms = sp.MinMaxScaler(feature_range=(0,1))
mms_samples = mms.fit_transform(raw_samples)
print(mms_samples)