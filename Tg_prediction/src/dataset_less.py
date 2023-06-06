import pandas as pd
import numpy as np
import h5py
from utils import read_data, separator_select, at_list, hash_list
from utils import info_all_data, data_preprocessing, generate_dataset

<<<<<<< HEAD:src/dataset_less.py
ppty    = 'Tg'
repre   = 'S0'
extens  = False
=======
ppty    = 'exp'
repre   = 'S2'
extens  = True
Flip    = True
>>>>>>> cb28c4f:src/dataset.py
dataset_N = [1,2,3,4,5]
root = '../data'
dataset_name  = 'data4DL.csv'
trainset_name = 'k-fold'

#  _,N,M = info_all_data(f'{root}/{dataset_name}','Tg','S2')
#  print(N,M)
#  import sys
#  sys.exit()

for dataset_n in dataset_N:
    OH_All, N_CHAR, MAX_LEN = info_all_data(f'{root}/{dataset_name}',ppty,repre)
    print(MAX_LEN)
    print(f"# of voca : {len(OH_All.columns)}")
<<<<<<< HEAD:src/dataset_less.py
    train_data = data_preprocessing(f'{root}/{trainset_name}_train_{dataset_n}.csv',ppty,repre,extens)
    test_data  = data_preprocessing(f'{root}/{trainset_name}_test_{dataset_n}.csv' ,ppty,repre,False)

    OH,LP = generate_dataset(train_data, OH_All, MAX_LEN, N_CHAR,Flip=False)
    OH_test, LP_test = generate_dataset(test_data, OH_All, MAX_LEN, N_CHAR,Flip=False)
=======
    print(f"Max length : {MAX_LEN}")
    train_data = data_preprocessing(f'{root}/{trainset_name}_train_{dataset_n}.csv',ppty,repre,extens,Flip)
    test_data  = data_preprocessing(f'{root}/{trainset_name}_test_{dataset_n}.csv' ,ppty,repre,False,False)

    OH,LP = generate_dataset(train_data, OH_All, MAX_LEN, N_CHAR,Flip=Flip)
    OH_test, LP_test = generate_dataset(test_data, OH_All, MAX_LEN, N_CHAR)
>>>>>>> f569125:src/dataset.py

    print(f"Train : {len(OH)}")
    print(f"Test : {len(OH_test)}")
    if extens: exten = 'ext'
    else: exten = 'nor'
    
    #  print(len(OH))
    #  import sys
    #  sys.exit()
    filename = f'{trainset_name}_{dataset_n}_{exten}_{ppty}_{repre}_less'
    print(f"data generated : ../data/{filename}.h5")
    h5f = h5py.File(f'../data/{filename}.h5','w')
    h5f.create_dataset('train_data', data=OH[:20])
    h5f.create_dataset('train_ans', data=LP[:20])
    h5f.create_dataset('test_data', data=OH_test)
    h5f.create_dataset('test_ans', data=LP_test)

    h5f.close()



