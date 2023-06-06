import pandas as pd
import numpy as np
import h5py
from utils import read_data, separator_select, at_list, hash_list
from utils import info_all_data, data_preprocessing, generate_dataset

ppty    = 'exp'
repre   = 'S2' 
extens  = False
Flip    = False
dataset_N = [1,]
root = '../data'
top = True
if not top:
    dataset_name  = 'Tg_dataset.csv'
    trainset_name = 'Tg'
if top:
    dataset_name = 'Tg_top_dataset.csv'
    trainset_name = 'Tg_top'

for dataset_n in dataset_N:
    OH_All, N_CHAR, MAX_LEN = info_all_data(f'{root}/{dataset_name}',ppty,repre,top)
    train_data = data_preprocessing(f'{root}/{trainset_name}_train_{dataset_n}.csv',ppty,repre,extens,Flip)
    test_data  = data_preprocessing(f'{root}/{trainset_name}_test_{dataset_n}.csv' ,ppty,repre,False,False)
    test_index = list(test_data.keys())
    index = ['train','test'] + test_index
    D = pd.DataFrame(data=0,index=index,columns = [list(OH_All.columns)])
    
    for k in train_data.keys():
        for letter in train_data[k][0]:
            D.loc['train'][letter] += 1
    for k in test_data.keys():
        for letter in test_data[k][0]:
            D.loc['test'][letter] += 1
            D.loc[k][letter] += 1
    D.to_csv(f'../result/{trainset_name}_inspection_{dataset_n}.csv')
print(D)
