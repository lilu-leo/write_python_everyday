#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:12:40 2017

@author: leo
"""

import pandas as pd
import os


def main():
    '''
    主函数
    '''
    
    aqi_data = pd.read_csv('china_city_aqi_demo.csv')
    print('基本信息： ')
    print(aqi_data.info())
    
    print('数据预览： ')
    print(aqi_data.head())
    
    # 基本统计
    print('AQI最大值:', aqi_data['AQI'].max())
    print('AQI最小值：', aqi_data['AQI'].min())
    print('AQI均值：', aqi_data['AQI'].mean())
    
    # top10
    top10_cities = aqi_data.sort_values(by=['AQI']).head(10)
    print('空气质量最好的10个城市：')
    print(top10_cities)
    
    # bottom10
    # bottom10_cities = aqi_data.sort_values(by=['AQI']).tail(10)
    bottom10_cities = aqi_data.sort_values(by=['AQI'], ascending=False).head(10)
    print('空气质量最差的10个城市：')
    print(bottom10_cities)
    
    # 保存csv文件
    top10_cities.to_csv('top10_aqi_demo.csv', index=False)
    bottom10_cities.to_csv('bottom10_aqi_demo.csv', index=False)
    
    
if __name__ == '__main__':
    main()
    
