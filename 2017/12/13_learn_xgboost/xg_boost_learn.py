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

dtrain = xgb.DMatrix(data, label = label, missing = -999.0)

# 当需要给样本设置权重时，可以用如下方式

w = np.random.rand(5, 1)
dtrain = xgb.DMatrix(data, label=label, missing = -999.0, weight=w)

# 3.2 参数设置 XGBoost使用key-value字典的方式存储参数：
params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax', # 多分类的问题
        'num_class': 10, # 类别数，与 multisoftmax 并用
        'gamma': 0.1, # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 12, # 构建树的深度，越大越容易过拟合
        'lambda': 2, # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7, # 随机采样训练样本
        'colsample_bytree': 0.7, # 生成树时进行的列采样
        'min_child_weight':3,
        'silent': 1, # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.007,  # 如同学习率
        'seed': 1000,
        'nthread':4 # cpu 线程数
        }

# 3.3 训练模型

num_round = 10
bst = xgb.train(plst, dtrain, num_round, evallist)

# 3.4 模型预测
# X_test类型可以是二维List，也可以是numpy的数组
dtest = DMatrix(X_test)
ans = model.predict(dtest)

# 3.5 保存模型 在训练完成之后可以将模型保存下来，也可以查看模型内部的结构

bst.save_model('test.model')

## 导出模型和特征映射（Map）你可以导出模型到txt文件并浏览模型的含义：

# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
bst.dump_model('dump.raw.txt','featmap.txt')

# 加载模型 通过如下方式可以加载模型：

bst = xgb.Booster({'nthread':4}) # init model
bst.load_model("model.bin")      # load data

'''

4. XGBoost参数详解
　　在运行XGboost之前，必须设置三种类型参数：general parameters，booster parameters和task parameters：

General parameters 
该参数参数控制在提升（boosting）过程中使用哪种booster，常用的booster有树模型（tree）和线性模型（linear model）。

Booster parameters 
这取决于使用哪种booster。

Task parameters 
控制学习的场景，例如在回归问题中会使用不同的参数控制排序。

'''





