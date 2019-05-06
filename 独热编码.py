import numpy as np
import sklearn.preprocessing as sp
raw_samples = np.array([
    [1,3,2],
    [7,5,4],
    [1,8,6],
    [7,3,9]
])
print(raw_samples)
#构建编码字典
code_tables = []
for col in raw_samples.T:
    code_table = {}
    for val in col:
        code_table[val] = None#空值占位
    code_tables.append(code_table)

for code_table in code_tables:
    size = len(code_tables)#每一列有多少个不同的值
    sortcode_table = sorted(code_table.keys())#每一列不同数字，从小到大排序
    for key,val in enumerate(sortcode_table):
        print(key,val)
    #     code_table[val] = np.zeros(shape=size)#创建多少个０
    #     code_table[val][key] = 1

#按字典编码
ohe_samples = []
for row in raw_samples:
    ohe_sample = np.array([],dtype=int)
    for key,val in enumerate(row):
        ohe_sample = np.hstack(
            ohe_sample,code_tables[key][val]
        )#水平拼接
        ohe_samples.append(ohe_sample)

#独热编码
one = sp.OneHotEncoder(sparse=True,dtype='int')
ohe_samples = one.fit_transform(raw_samples)

print(ohe_samples)
new_samples = np.array([#用已有字典去模拟，如果列内出现字典里没有的编码，结果将会出错
    [7,8,9],
    [2,5,2],
])
ohe_samples2 = one.transform(new_samples)
print(ohe_samples)

