#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:41:56 2017

@author: leo

url http://www.jianshu.com/p/fde8a56fcca5

data set url https://grouplens.org/datasets/movielens/
"""



import pandas as pd
import os
path = '/Users/leo/Develop/write_python_everyday/2017/12/12_learn_data_frame'
os.chdir(path)

# pass in column names for each CSV

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('ml-100k/u.user', sep = '|', names = u_cols, encoding = 'latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep = '\t', names = r_cols, encoding = 'latin-1')

# the movies file contains columns indicating the movie`s genres 
# let`s only load the first five columns of the file with usecols

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep = '|', names = m_cols, usecols = range(5), encoding = 'latin-1')

'''
1.该数据集是一个 DataFrame 实例。
2.数据的行索引是从 0 到 N-1 的一组数字，其中 N 为 DataFrame 的行数。
3.数据集中总共有 1682 行观测值。
4.数据集中有五列变量，其中变量 video_release_date 中没有数据，变量 release_date 和 imdb_url 中存在个别缺失值。
5.最后一行给出了变量数据类型汇总情况，你可以利用 dtypes 方法获取每个变量的数据类型。
6.保存该数据集所耗费的内存，你可以利用 .memory_usage 获取更多信息。

'''
movies.info()

movies.memory_usage()

movies.dtypes

'''
它用于获取数据集的常用统计量信息。需要注意的是，该方法仅会返回数值型变量的信息，所以我们会得到 user_id 和 age 两个变量的统计量信息
'''
users.describe()

'''
默认情况下，head方法会返回数据集的前五条记录，tail方法会返回最后五条记录
'''

movies.head()
movies.tail(4)

movies[20:22]

# Selecting

users['occupation'].head(8)

users[['age', 'zip_code']].head()

columns_need = ['occupation', 'sex']

users[columns_need].head()

# 从 DataFrame 中提取行数据有多种方法，常用的方法有行索引法和布尔索引法

users[users.age > 25].head(4)

users[(users.age == 36) & (users.sex == 'M')].head(4)

users[(users.age < 30) | (users.sex == 'F')].head(5)

'''
由于通常情况下行索引值都是一组无实际意义的数值，我们可以利用set_index方法将user_id设定为索引变量。
默认情况下，set_index方法将返回一个新的 DataFrame。
'''

users.set_index('user_id').head()

users.head()

with_new_index = users.set_index('user_id')
with_new_index.head()

# 你还可以设定参数inplace的数值来修改现有的 DataFrame 数据集。

users.set_index('user_id', inplace = True)
users.head()

# 我们可以利用iloc方法根据位置来选择相应的行数据。

users.iloc[99]

users.iloc[[1, 50, 300], [0, 2]]

# 也可以利用loc方法根据label来选取行数据
'''
loc 方法主要是使用标签
'''

users.loc[100]
users.loc[[2, 51, 301]]

users.reset_index(inplace = True)
users.head()

# Joining

'''
how : {'left', 'right', 'outer', 'inner'}, default 'inner'
left: 只保留左表键值的数据 (SQL:left outer join)
right: 只保留右表键值的数据 (SQL:right outer join)
outer: 保留两个表格键值并集的数据 (SQL: full outer join)
inner: 保留两个表格键值交集的数据 (SQL: inner join)
'''

left_frame = pd.DataFrame({'key':range(5), 'left_value' : ['a', 'b', 'c', 'd', 'e']})

right_frame = pd.DataFrame({'key':range(2, 7), 'right' : ['f', 'g', 'h', 'i', 'j']})


left_frame
right_frame

pd.merge(left_frame, right_frame, on = 'key', how = 'inner')

'''
如果两个 DataFrame 的键值命名不一致，我们可以left_on和right_on参数来指定合并的键值——
pd.merge(left_frame, right_frame, left_on = 'left_key', right_on = 'right_key')。
此外，如果键值是 DataFrame 的索引值，我们还可以利用left_index和right_index参数来指定合并的键值——
pd.merge(left_frame, right_frame, left_on = 'key', right_index = True)
'''

pd.merge(left_frame, right_frame, on = 'key', how = 'left')

pd.merge(left_frame, right_frame, on = 'key', how = 'right')

pd.merge(left_frame, right_frame, on = 'key', how = 'outer')

'''
Combining
Pandas 还提供了另一种沿着轴向合并 DataFrames 的方法——pandas.concat，
该函数等价于 SQL 中的 UNION 语法。
'''
pd.concat([left_frame, right_frame])

# 还可以利用axis参数来控制合并的方向 0 是纵向， 1是横向
pd.concat([left_frame, right_frame], axis = 0)
pd.concat([left_frame, right_frame], axis = 1)

headers = ['name', 'title', 'department', 'salary']

# city-of-chicago-salaries.csv 数据源没有找
chicago = pd.read_csv('city-of-chicago-salaries.csv', header = 0, names = headers, 
                      converters = {'salary' : lambda x: float(x.replace('$', ''))})
chicago.head()

by_dept = chicago.groupby('department')
by_dept.count().head()
by_dept.size().tail()

by_dept.sum()[20:25]
by_dept.mean()[20:25]
by_dept.median()[20:25]

by_dept.title.nunique().sort_values(ascending=False)[:5]

def ranker(df):
    """
    Assigns a rank to each employee based on salary, with 1 being the highest paid.
    """
    df['pept_rank'] = np.arange(len(df)) + 1
    return df

chicago.sort_values('salary', ascending=False, inplace=True)

chicago = chicago.groupby('department').apply(ranker)

chicago[chicago.dept_rank == 1].head(7)
    



