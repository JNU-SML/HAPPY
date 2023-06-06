import random
import pandas as pd
import numpy as np
import h5py

#data_pan = pd.read_csv('../data/data_smiles.csv',sep=',',names=['S0','S1','S2','exp','syn'])
data_pan = pd.read_csv('../data/data_smiles.csv',sep=',',names=['S0','S1','S2','S3','exp','syn','Tg','Density','Solu','Tc'],skiprows=1)
data_pan = data_pan.droplevel(0,axis=0)
#  print(data_pan)
S = data_pan
count = 0
col = list(S.columns)

idx =[]
for n in range(len(S)):
    if S.iloc[n].isna().any():
        idx.append(n)

print(idx)
print(idx)
print(S.drop(index=S.iloc[idx].index,axis=1,inplace=True))

S.to_csv('../data/data4DL.csv')
print("Done")
