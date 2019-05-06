import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    'audi','ford','audi','toyota',
    'ford','bmw','toyota','ford','audi'
])
print(raw_samples)
lbe = sp.LabelEncoder()
lbe_samples = lbe.fit_transform(raw_samples)
print(lbe_samples)
lbe_samples = lbe.inverse_transform(lbe_samples)
print(lbe_samples)

