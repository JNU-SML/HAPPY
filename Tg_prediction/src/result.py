import pandas as pd
import glob
import numpy as np

LR = ['0.0001','1e-05','1e-06']
ext = ['ext','nor']
repre = ['S2']
node_n = [16,32]
idx1 = []
idx2 = []
idx3 = []
idx4 = []
data = []

index = pd.MultiIndex.from_product([LR,node_n,['d1','d2','d3','d4','d5'],list(np.arange(0,12))])
column = ['name','S0_nor','S2_nor','S2_ext','S3_nor','S3_ext','ans']

df = pd.DataFrame(index=index,columns=column)
print(df)

for lr in LR:
    for nn in node_n:
        for i in [1,2,3,4,5]:
            result = pd.DataFrame()
            for rep in repre:
                for ex in ext:
                    if ex=='ext' and rep=='S0': continue
                    loc = f'../result/k-fold_{i}_{ex}_Solu_{rep}_{lr}_16.txt'
                    D = pd.read_csv(loc,names=['pre','ans'],sep='  ', skipfooter=2) 
                    dataset = f'../data/k-fold_test_{i}.csv'
                    #  dataset = pd.read_csv(dataset,
                    dataset = pd.read_csv(dataset,sep=',',names=['S0','S1','S2','S3','exp','syn','Tg','Density','Solu','Tc'],skiprows=1)
                    name = rep+'_'+ex
                    #  df.loc["0.0001",32,"d1",1]["S0_nor"]=10
                    for di in range(len(dataset)):
                        polyname = dataset.iloc[di].name
                        ans   = dataset.iloc[di].exp
                        pred  = D.iloc[di].pre
                        df.loc[f"{lr}",nn,f"d{i}",di][name] = pred
                        df.loc[f"{lr}",nn,f"d{i}",di]["ans"] = ans
                        df.loc[f"{lr}",nn,f"d{i}",di]["name"] = polyname




D = pd.read_csv('../result/k-fold_5_nor_exp_S2_0.0001_16.txt',names = ['pre','ans'],sep='  ', skipfooter=2)
#  print(D)
print(df)
df.to_csv('../result/combined.csv')
