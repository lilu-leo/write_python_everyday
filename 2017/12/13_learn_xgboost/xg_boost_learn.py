#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:09:14 2017

@author: leo

url http://www.voidcn.com/article/p-zkdwsdyz-bpo.html
"""
import xgboost as xgb
import scipy

from sklearn import svm 
X = [[0, 0], [1, 1]]
y = [0, 1]

clf = svm.SVC()
clf.fit(X, y)

clf.predict([[2., 2.]])

# 加载numpy的数组
data = np.random.rand(5, 10)
data

label = np.random.randint(2, size=5)
label

dtrain = xgb.DMatrix(data, label=label)

csr = scipy.sparse.csr_matrix((dat, (row, col)))
dtrain = xgb.DMatrix(csr)

# 可以用如下方式处理 DMatrix中的缺失值
