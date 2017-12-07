#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:33:05 2017

@author: leo
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

path = '/Users/leo/Desktop/零基础Python入门课件和代码/lect09_代码'
os.chdir(path)

def main():
    '''
    主函数
    '''
    
    aqi_data = pd.read_csv('china_city_aqi.csv')
    print('基本信息： ')
    print(aqi_data.info())
    
    print('数据预览： ')
    print(aqi_data.head())
    
    # 数据清洗
    # 只保留AQI>0的数据
    # filter_condition = aqi_data['AQI'] > 0
    # clean_aqi_data = aqi_data[filter_condition]
    
    clean_aqi_data = aqi_data[aqi_data['AQI'] > 0]
    
    # 基本信息
    
    print('AQI最大值:', clean_aqi_data['AQI'].max())
    print('AQI最小值：', clean_aqi_data['AQI'].min())
    print('AQI均值：', clean_aqi_data['AQI'].mean())
    
    top50_cities = clean_aqi_data.sort_values(by=['AQI']).head(50)
    top50_cities.plot(kind='bar', x='City', y='AQI', title='空气质量最好的50个城市',
                      figsize=(20, 10))
    
    plt.savefig('top50_aqi_bar_demo.png')
    plt.show()
    
if __name__ =='__main__':
    main()
    
    
    
    
    