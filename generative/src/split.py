import pandas as pd 
import random

random.seed(123)
location = '../data/final.csv'
data_pan = pd.read_csv(f'{location}',sep=',',names=['S0','S1','S2','S3','exp','syn','Tg','Density','Solu','Tc'],skiprows=1)
 

origin_idx = [i for i in range(len(data_pan))]
idx = [20*(i+1) for i in range(20)]

#  origin_idx.shuffle()
random.shuffle(origin_idx)


for i in idx:
    print(i)
    random.shuffle(origin_idx)
    new = origin_idx[:i]
    print(new)
    new_pan = data_pan.iloc[new,:]
    
    new_pan.to_csv(f'../data/final_train_split_{i}.csv')

