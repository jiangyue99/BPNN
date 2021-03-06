#!/usr/bin/env python
# -*- coding: gb2312 -*-

import numpy as np
import pandas as pd
import network
import matplotlib.pyplot as plt
import openpyxl

print('开始训练，较耗时，请稍等。。。')


data = pd.read_excel('D:\LRuanJian\Python\python_work\EVPmonth.xlsx', \
                      'wdl52908month', encoding='gb2312')
                      #encoding="unicode_escape

month_data = np.array(data[['EVP']])/2500.
month_data = month_data.tolist()

def get_data(input_data, n):
    num_train = len(input_data)//12 - n+1
    final_data = []
    train_data = []
    test_data = []
    for i in range(num_train):
        locals()['train' + str(i)] = []
        for j in range(12 * n):
            locals()['train' + str(i)] += [input_data[-1 - j - 12*i]]
    locals()['result' + str(0)] = [1]
    for i in range(num_train - 1):
        locals()['result' + str(i+1)] = input_data[-12 - 12*i]
    for i in range(num_train):
        final_data += [(np.array(locals()['train' + str(i)]), \
                        np.array(locals()['result' + str(i)]))]
    final_data = np.array(final_data)
    train_data = final_data[num_train//3:]
    test_data = final_data[:num_train//3]
    return train_data, test_data

for i in range(30*12):
    training_data, test_data = get_data(month_data, 25)
    # 4*12 个输入神经元，一层隐藏层，包含 30 个神经元，输出层包含 10 个神经元
    net = network.Network([25*12, 15, 1])
    month_data += [net.SGD(training_data, 100, len(training_data), 3, test_data = test_data)]
df = pd.DataFrame(month_data)
book = openpyxl.load_workbook('predict.xlsx')
writer = pd.ExcelWriter('predict.xlsx', engine='openpyxl')
writer.book = book
x = pd.Series(pd.period_range('1/1/1960', freq='M', periods=1021))
df.to_excel(writer, 'EVP')
x.to_excel(writer, 'sheet1')
writer.close()
plt.plot(month_data)
plt.show()
