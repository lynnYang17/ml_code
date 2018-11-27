import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

print('Load data...')
df_train = pd.read_csv('../data/regression.train', header=None, sep='\t')
df_test = pd.read_csv('../data/regression.test', header=None, sep='\t')

X_train = df_train.drop(0, axis=1).values  # 训练数据有7000个标签 ——> 一维数组
X_test = df_test.drop(0, axis=1).values  # 测试数据有500个标签
y_train = df_train[0].values  # 训练数据7000条，每条28个维度 ——> 二维数组
y_test = df_test[0].values  # 测试数据500条，每条28个维度
print('-------------------- The shape of the data -------------------- ')
print('The shape of X_train is: ', end=' ')
print(X_train.shape)
print('The shape of X_test is: ', end=' ')
print(X_test.shape)
print('The shape of y_train is: ', end=' ')
print(y_train.shape)
print('The shape of y_test is: ', end=' ')
print(y_test.shape)

# 使用逻辑回归模型
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
error_num = np.sum(abs(predictions-y_test))
correct_rate = (len(y_test) - error_num)/len(y_test)

print('The test data set is: ')
print(y_test)

print('The number of the wrong data is: ', end=' ')
print(error_num)

print('The correct rate of the data is: ', end=' ')
print(correct_rate)

# print(df_train[1].values)
# print(df_train.drop(0, axis=1).values.shape)
