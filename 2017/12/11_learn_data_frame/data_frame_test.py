#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:41:21 2017

@author: leo

url: http://www.cnblogs.com/chaosimple/p/4153083.html
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# 1 创建对象
## 1.1 可以通过传递一个list对象来创建一个Series，pandas会默认创建整型索引
s = pd.Series([1, 3, 5, np.nan, 6, 8])

s

## 1.2 通过传递一个numpy array，时间索引以及列标签来创建一个DataFrame
dates = pd.date_range('20130101', periods=6)

dates


df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

df

## 1.3 通过传递一个能够被转换成类似序列结构的字典对象来创建一个DataFrame
df2 = pd.DataFrame(
        {'A' : 1.,
         'B' : pd.Timestamp('20130102'),
         'C' : pd.Series(1, index=list(range(4)), dtype='float32'),
         'D' : np.array([3] * 4, dtype='int32'),
         'E' : pd.Categorical(["test", "train", "test", "train"]),
         'F' : 'foo'})

df2

## 1.4 查看不同列的数据类型
df2.dtypes


# 2 查看数据

## 2.1 查看frame中头部和尾部的行
df.head()

df.tail(3)

## 2.2 显示索引、列和底层的numpy数据
df.index

df.columns

df.values

## 2.3  describe()函数对于数据的快速统计汇总
df.describe()

## 2.4 对数据的转置
df.T

## 2.5 按轴进行排序
df.sort_index(axis=1, ascending=False)

## 2.6 按值进行排序

# python2.7 df.sort(columns='B')

df.sort_index()

# 3 选择


## 3.1 获取
### 3.1.1 选择一个单独的列，这将会返回一个Series，等同于df.A
df.A
#df[A]

### 3.1.2 通过[]进行选择，这将会对行进行切片 
df[0:2]

## 3.2 通过标签选择
### 3.2.1  使用标签来获取一个交叉的区域

df.loc[dates[0]]

### 3.2.2 通过标签来在多个轴上进行选择
df.loc[:, ['A', 'B']]

### 3.2.3 标签切片
df.loc['20130102':'20130104', ['A', 'B']]

### 3.2.4  对于返回的对象进行维度缩减
df.loc['20130102', ['A', 'B']]

dates[0]

### 3.2.5 获取一个标量
df.loc[dates[0], 'A']

### 3.2.6 快速访问一个标量（与上一个方法等价）
df.at[dates[0], 'A']

## 3.3 通过位置选择
### 3.3.1 通过传递数值进行位置选择（选择的是行）
df.iloc[3]

df.shape

df.iloc[0]


### 3.3.2 通过数值进行切片，与numpy/python中的情况类似
df.iloc[3:5, 0:2]

### 3.3.3 通过指定一个位置的列表，与numpy/python中的情况类似
df.iloc[[1,2,4],[0,2]]

### 3.3.4  对行进行切片
df.iloc[1:3, :]

### 3.3.5 对列进行切片
df.iloc[:, 1:3]

### 3.3.6  获取特定的值
df.iloc[1, 1]

## 3.4 布尔索引

### 3.4.1 使用一个单独列的值来选择数据
df[df.A > 0]

### 3.4.2 使用where操作来选择数据
df[df > 0]

print ("hello ", '\n nihao')

### 3.4.3  使用isin()方法来过滤
df2 = df.copy()
df2['E']=['one', 'one', 'two', 'three', 'four', 'three']
df2

df2[df2['E'].isin(['two', 'four'])]

## 3.5 设置 
### 3.5.1  设置一个新的列
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))

s1

df['F'] = s1

### 3.5.2  通过标签设置新的值
df.at[dates[0], 'F'] = 0

### 3.5.3 通过位置设置新的值
df.iat[0,4] = -1

df

### 3.5.4 通过一个numpy数组设置一组新值
df.loc[3:, 'D'] = np.array([5] * 3)

df

### 3.5.5 通过where操作来设置新的值
df2 = df.copy()

df2[df2 > 0] = -df2

df2

# 4. 缺失值处理
## 4.1  reindex()方法可以对指定轴上的索引进行改变/增加/删除操作，这将返回原始数据的一个拷贝
df1 = df.reindex(index=dates[0:5], columns=list(df.columns) + ['E'])
df1.loc[dates[0:2],'E'] = 1

df1

## 4.2 去掉包含缺失值的行
df.dropna(how='any')

## 4.3 对缺失值进行填充
df1.fillna(value=5)

## 4.4 对数据进行布尔填充
pd.isnull(df1)

# 5.相关操作 
## 5.1 统计

### 5.1.1 执行描述性统计
df.mean()

### 5.1.2 在其他轴上进行相同的操作
df.mean(1)
#df.mean(2)

### 5.1.3 对于拥有不同维度，需要对齐的对象进行操作。
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)

s

## 5.2 Apply 
### 5.2.1 对数据应用函数
df.apply(np.cumsum)

df.apply(lambda x: x.max() - x.min())

## 5.3 直方图
s = pd.Series(np.random.randint(0,7, size=10))
s

s.value_counts()

## 字符串方法
s = pd.Series(['A', 'B', 'C', 'D', np.nan, 'dog'])

s.str.lower()

# 6 合并

## 6.1 Concat
df = pd.DataFrame(np.random.randn(10, 4))

df

pieces = [df[:3], df[3:7], df[7:]]
pieces

pd.concat(pieces)

df1=pd.DataFrame({'key':['a','b','b'],'data1':range(3)}) 
df1


df2=pd.DataFrame({'key':['a','b','c'],'data2':range(3)}) 
df2

pd.merge(df1, df2)


## 6.2 Join 类似于SQL类型的合并
left=pd.DataFrame({'key1':['foo','foo','bar'],  
         'key2':['one','two','one'],  
         'lval':[1,2,3]}) 

right=pd.DataFrame({'key1':['foo','foo','bar','bar'],  
         'key2':['one','one','one','two'],  
         'lval':[4,5,6,7]}) 

pd.merge(left,right,on=['key1','key2'],how='outer')

pd.merge(left,right,on=['key1','key2'],how='inner')

left = pd.DataFrame({'key':['foo', 'foo'], 'lval':[1, 2]})

right = pd.DataFrame({'key':['foo', 'foo'], 'lval':[4, 5]})

pd.merge(left, right, on='key', how='outer')


## 6.3 Append 将一行连接到一个DataFrame上
# 8行 4列
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
df

s = df.iloc[3]
s

df.append(s, ignore_index=True)

# 7 分组
'''
  （Splitting）按照一些规则将数据分为不同的组；
  （Applying）对于每组数据分别执行一个函数；
  （Combining）将结果组合到一个数据结构中；
'''

df=pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                 'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                 'C' : np.random.randn(8), 
                 'D' : np.random.randn(8)})

df         
    
## 7.1  分组并对每个分组执行sum函数    
df.groupby('A').sum()

## 7.2 通过多个列进行分组形成一个层次索引，然后执行函数
df.groupby(['A', 'B']).sum()

# 8 Reshaping

## 8.1 stack
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two', 
                     'one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])

df2 = df[:4]

df2

stacked = df2.stack()

stacked

stacked.unstack()

stacked.unstack(1)

stacked.unstack(0)

## 8.2 数据透视表
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3, 
                   'B' : ['A', 'B', 'C'] * 4, 
                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2, 
                   'D' : np.random.randn(12), 
                   'E' : np.random.randn(12)})

df

pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])

# 9 时间序列

rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min').sum()

## 9.1 时区表示
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)

ts

ts_utc = ts.tz_localize('UTC')

ts_utc

## 9.2时区转换
ts_utc.tz_convert('US/Eastern')

# 9.3 时间跨度转换
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts

ps = ts.to_period()
ps

ps.to_timestamp()

# 9.4 时期和时间戳之间的转换使得可以使用一些方便的算术函数。

prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')

ts = pd.Series(np.random.randn(len(prng)), prng)

ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9

ts.head()

# 10 Categorical

## 10.1  将原始的grade转换为Categorical数据类型
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade" :['a', 'b', 'b', 'a', 'a', 'e']})

df["grade"] = df["raw_grade"].astype("category")

df["grade"]

## 10.2 将Categorical类型数据重命名为更有意义的名称
df["grade"].cat.categories = ["very good", "good", "very bad"]
df["grade"]

## 10.3  对类别进行重新排序，增加缺失的类别
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df["grade"] 

##  10.4 排序是按照Categorical的顺序进行的而不是按照字典顺序进行
df.sort("grade")

df.groupby("grade").size()

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

ts = ts.cumsum()

ts.plot()

# 11 画图 
df = pd.DataFrame(200*np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])

df.cumsum()
plt.figure()
df.plot()
plt.legend(loc='best')

# 12 导入和保存数据

## 12.1保存
df.to_csv('foo.csv')

## 12.2 从csv文件中读取

pd.read_csv('foo.csv')

## 12.3 写入HDF5存储
df.to_hdf('foo.h5', 'df')

## 12.4  从HDF5存储中读取

pd.read_hdf('foo.h5', 'df')

## 12.5 写入Excel文件

df.to_excel('foo.xlsx', sheet_name='Sheet1')

## 12.6 从excel文件中读取
pd.read_excel('foo.xlsx', index_col=None, na_values=['NA'])

























