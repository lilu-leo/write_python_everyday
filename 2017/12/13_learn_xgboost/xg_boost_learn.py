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

'''
4.1 General Parameters

    booster [default=gbtree]

    有两中模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。缺省值为gbtree

    silent [default=0]

        取0时表示打印出运行时信息，取1时表示以缄默方式运行，不打印运行时信息。缺省值为0

    nthread

        XGBoost运行时的线程数。缺省值是当前系统可以获得的最大线程数

    num_pbuffer

        预测缓冲区大小，通常设置为训练实例的数目。缓冲用于保存最后一步提升的预测结果，无需人为设置。

    num_feature

        Boosting过程中用到的特征维数，设置为特征个数。XGBoost会自动设置，无需人为设置。
'''

'''
Parameters for Tree Booster

    eta [default=0.3] 
        为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。 eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3 
        取值范围为：[0,1]

    gamma [default=0] 
        minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, the more conservative the algorithm will be. 
        取值范围为：[0,∞]

    max_depth [default=6] 
        数的最大深度。缺省值为6 
        取值范围为：[1,∞]

    min_child_weight [default=1] 
        孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative 
        取值范围为：[0,∞]

    max_delta_step [default=0] 
        我们允许每个树的权重被估计的值。如果它的值被设置为0，意味着没有约束；如果它被设置为一个正值，它能够使得更新的步骤更加保守。通常这个参数是没有必要的，但是如果在逻辑回归中类极其不平衡这时候他有可能会起到帮助作用。把它范围设置为1-10之间也许能控制更新。 
        取值范围为：[0,∞]

    subsample [default=1] 
        用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中随机的抽取出50%的子样本建立树模型，这能够防止过拟合。 
        取值范围为：(0,1]

    colsample_bytree [default=1] 
        在建立树时对特征采样的比例。缺省值为1 
        取值范围为：(0,1]

'''





