import numpy as np
import sklearn.preprocessing as sp
raw_samples = np.array([
    [3, -1.5,  2,  -5.4],
    [0,  4,   -0.3, 2.1],
    [1,  3.3, -1.9, -4.3]])
print(raw_samples)
print(abs(raw_samples).sum())#L1范数
bin_samples = raw_samples.copy()
threshold = 1.4#阈值
bin_samples[bin_samples <= threshold] = 0
bin_samples[bin_samples > threshold] = 1
print(bin_samples)

bin = sp.Binarizer(threshold=threshold)
bin_samples = bin.transform(raw_samples)
print(bin_samples)