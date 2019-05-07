# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.ensemble as se
import matplotlib.pyplot as mp
boston = sd.load_boston()
fn = boston.feature_names
x, y = su.shuffle(
    boston.data, boston.target, random_state=7)
train_size = int(len(x) * 0.8)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:]
model = st.DecisionTreeRegressor(max_depth=4)
model.fit(train_x, train_y)
fi1 = model.feature_importances_
print(fi1)
model = se.AdaBoostRegressor(
    st.DecisionTreeRegressor(max_depth=4),
    n_estimators=400, random_state=7)
model.fit(train_x, train_y)
fi2 = model.feature_importances_
mp.figure('Feature Importance', facecolor='lightgray')
mp.subplot(211)
mp.title('Decision Tree', fontsize=16)
mp.ylabel('Importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
sorted_indices = fi1.argsort()[::-1]
pos = np.arange(len(sorted_indices))
mp.bar(pos, fi1[sorted_indices],
       edgecolor='steelblue', facecolor='deepskyblue')
mp.xticks(pos, fn[sorted_indices], rotation=30)
mp.subplot(212)
mp.title('AdaBoost Decision Tree', fontsize=16)
mp.xlabel('Feature', fontsize=12)
mp.ylabel('Importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
sorted_indices = fi2.argsort()[::-1]
pos = np.arange(len(sorted_indices))
mp.bar(pos, fi2[sorted_indices],
       edgecolor='indianred', facecolor='lightcoral')
mp.xticks(pos, fn[sorted_indices], rotation=30)
mp.tight_layout()
mp.show()
