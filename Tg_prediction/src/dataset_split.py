import pandas as pd
import numpy as np
import h5py
from utils import read_data, separator_select, at_list, hash_list
from utils import info_all_data, data_preprocessing, generate_dataset

ppty    = 'exp'
repre   = 'S0'
extens  = False
Flip    = False
dataset_N = [1,]
root = '../data'
dataset_name  = 'data4DL.csv'
trainset_name = 'k-fold'

if repre == 'S0':
    if any([extens,Flip]):
        print("Your representation S0")
        print("Can not operate Flip or Extenstion")
        print("Aborted")
        import sys
        sys.exit()

#  _,N,M = info_all_data(f'{root}/{dataset_name}','Tg','S2')
#  print(N,M)
#  import sys
#  sys.exit()

idx = [20*(i+1) for i in range(20)]
for i in idx:
    for dataset_n in dataset_N:
        OH_All, N_CHAR, MAX_LEN = info_all_data(f'{root}/{dataset_name}',ppty,repre)
        print(f"# of voca : {len(OH_All.columns)}")
        print(f"Max length : {MAX_LEN}")
        train_data = data_preprocessing(f'{root}/{trainset_name}_train_split_all_{i}.csv',ppty,repre,extens,Flip)
        #  train_data = data_preprocessing(f'{root}/{trainset_name}_train_{dataset_n}.csv',ppty,repre,extens,Flip)
        test_data  = data_preprocessing(f'{root}/{trainset_name}_test_{dataset_n}.csv' ,ppty,repre,False,False)

        OH,LP = generate_dataset(train_data, OH_All, MAX_LEN, N_CHAR,Flip=Flip)
        OH_test, LP_test = generate_dataset(test_data, OH_All, MAX_LEN, N_CHAR)

        print(f"Train : {len(OH)}, Test : {len(OH_test)}")
        if extens: exten = 'ext'
        else: exten = 'nor'
       
        filename = f'{trainset_name}_{dataset_n}_{exten}_{ppty}_{repre}'
        print(f"data generated : ../data/{filename}_all_split{i}.h5\n")
        h5f = h5py.File(f'../data/{filename}_all_split{i}.h5','w')
        print(f"data generated : ../data/{filename}.h5\n")
        #  h5f = h5py.File(f'../data/{filename}.h5','w')
        h5f.create_dataset('train_data', data=OH)
        h5f.create_dataset('train_ans', data=LP)
        h5f.create_dataset('test_data', data=OH_test)
        h5f.create_dataset('test_ans', data=LP_test)

        h5f.close()



