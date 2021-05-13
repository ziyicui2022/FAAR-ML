import os
import numpy as np
import pandas as pd


def read_data(log_path_list):
    temp = pd.read_table(log_path_list[0], sep='\n', header=None)
    for i in range(1, len(log_path_list)):
        temp = pd.concat([temp, pd.read_table(log_path_list[i], sep='\n', header=None)], axis=1)
    return temp

log_path_list = []
c = read_data(log_path_list)

def generate_mean(time_step):
    lt = []
    for i in range(c.shape[1]):
        lt.append(np.array(list(map(float, c.iloc[i, time_step].split()))))
    return np.mean(np.array(lt), axis=0)


for i in range(20):
    res = generate_mean(i)
    with open('result.txt', 'a') as f:
        f.write(" ".join([str(x) for x in res]))
