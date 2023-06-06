import random
import pandas as pd
import numpy as np
import h5py
import random
from utils import separator_select
from utils import read_data

random.seed(47945)
data_pan = pd.read_csv('../data/NEW_Tg_top_dataset.csv',sep=',',names=['S0','S1','S2','S3','exp','syn','Tg','Density','Solu','Tc'],skiprows=1)
#  print(data_pan)
ind = list(np.arange(len(data_pan)))
random.shuffle(ind)
testidx = [40]

test_tmp = 0
data_ind = []
for i in range(len(testidx)):
    t = testidx[i]
    dataset1 = ind[:test_tmp]
    testset  = ind[test_tmp:test_tmp+t]
    dataset2 = ind[test_tmp+t:]
    dataset = dataset1+dataset2
    data_ind.append([testset,dataset])
    test_tmp += t

def generate_dict(location,lang):
    #  if ppty == 'S0' separator=separator_smiles
    separator = separator_select(lang)
    raw_data, _ = read_data(location,'exp',lang)
    #  print(list(raw_data),lang)
    MAX_LEN =[]
    flatten_list = []
    for num in range(len(raw_data)):
        #  print(list(raw_data))
        s = list(raw_data)[num]
        #  print(list(raw_data),s)
        if pd.isna(s): continue
        else:
            s_list = separator(s)
            MAX_LEN.append(len(s_list))
            flatten_list.extend(s_list)
    flatten_list.append('empty')
    flatten_list = set(flatten_list)
    OH_ALL = pd.get_dummies(list(flatten_list))
    N_CHAR = len(flatten_list)
    OH_ALL.to_csv(f'../data/{lang}_dictionary_top')
    return OH_ALL, N_CHAR,max(MAX_LEN)


n = 1
print(len(data_ind))
for d in data_ind:
  test,train = d
  data_pan.iloc[train].to_csv(f'../data/NEW_Tg_top_train_{n}.csv')
  data_pan.iloc[test].to_csv(f'../data/NEW_Tg_top_test_{n}.csv')
  n +=1
for lang in ['S2','S0']:
    print(lang)
    generate_dict('../data/NEW_Tg_top_dataset.csv',lang)

#  print(pd.read_csv('../data/S2_dictionary',index_col=0))
