import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def preprocessing_smiles(s):
    smiles_dict = {'Cl' : "L", 'Si' : "U", "Na" : "A",  }
    old_s = s
    for name in smiles_dict.keys():
        s = s.replace(name,smiles_dict[name])
    return s


def read_data(location,ppty,lang): 
    data_pan = pd.read_csv(f'{location}',sep=',',names=['S0','S1','S2','S3','exp','syn','Tg','Density','Solu','Tc'],skiprows=1)
    ans = data_pan.exp
    for i,a in enumerate(ans):
        if pd.isna(a):
            n = data_pan.iloc[i].name
            data_pan.loc[n,"exp"] = data_pan.iloc[i].syn
     
    S2 = data_pan[f'{lang}']
    ans = data_pan[f'{ppty}']

    idx = ans[ans.isnull()].index
    S2 = data_pan[f'{lang}'].drop(idx)
    ans = data_pan[f'{ppty}'].drop(idx)
    return S2,ans

def separator_S2(s):
    s_list = []
    skip = -1
    for i in range(len(s)):
        if skip > 1:
            skip -= 1
            continue
        if s[i] == 'H':
            s_list.append(s[i])
        elif s[i] == 'T':
            s_list.append(s[i])
            break
        elif s[i] == 'C':
            if s[i+1].isalpha() and s[i+1].islower():
                s_list.append(s[i]+s[i+1])
                skip += 1
            else:
                s_list.append(s[i])
        elif s[i] == '@':
            s_list.append(s[i])
        elif s[i] == '#':
            s_list.append(s[i])
        elif s[i] == '*':
            tmp = s[i]+s[i+1]
            skip = 2
            N_s = s[i+skip]
            while N_s.isalpha() and N_s.islower():
                tmp += N_s
                skip += 1
                N_s = s[i+skip]
            s_list.append(tmp)
            tmp = ''
        elif s[i].isupper():
            tmp = s[i]
            skip = 1
            N_s = s[i+skip]
            while N_s.isalpha() and N_s.islower():
                tmp += N_s
                skip += 1
                N_s = s[i+skip]
            s_list.append(tmp)
    #  print(s_list)
    return s_list

def separator_S0(s):
    s = preprocessing_smiles(s)
    s_list = []
    for i in range(len(s)):
        s_list.append(s[i])
    return s_list


def at_list(s_list):
    at_s_list = []
    c=0
    for i in range(len(s_list)-1):
        if c > 1:
            c -= 1
            continue
        if s_list[i+1] == '@':
            c = 1
            while s_list[i+c] == '@':
                c += 2
            tmp = ''.join(s_list[i:i+c])
            at_s_list.append(tmp)
        else:
            at_s_list.append(s_list[i])
    at_s_list.append('T')
    return at_s_list
        #  print("##")
def hash_list(at_s_list):
    hash_s_list = []
    c = 0
    for i in range(len(at_s_list)-1):
        if c > 1:
            c -= 1
            continue
        if at_s_list[i+1] == '#':
            c = 1
            while '#' in at_s_list[i+1+c:i+1+c+3]:
                c += 3
            tmp = ''.join(at_s_list[i:i+c+3])
            c += 3
            hash_s_list.append(tmp)
        else:
            hash_s_list.append(at_s_list[i])
    hash_s_list.append('T')
    return hash_s_list

def separator_select(lang):
    if lang == 'S0' : separator = separator_S0
    if lang == 'S2' : separator = separator_S2
    if lang == 'S3' : separator = separator_S2
    return separator

def info_all_data(location,ppty,lang,TOP):
    #  if ppty == 'S0' separator=separator_smiles
    separator = separator_select(lang)
    raw_data, _ = read_data(location,ppty,lang)
    MAX_LEN =[]
    flatten_list = []
    loc = '/'.join(location.split('/')[:-1])
    for num in range(len(raw_data)):
        s = list(raw_data)[num]
        if pd.isna(s): continue
        else:
            s_list = separator(s)
            MAX_LEN.append(len(s_list))
            flatten_list.extend(s_list)
    flatten_list.append('empty')
    flatten_list = set(flatten_list)
    if TOP:
        dictionary_loc = loc + f'/{lang}_dictionary_top'
    else:
        dictionary_loc = loc + f'/{lang}_dictionary'

    print(f"Voca dictionary : {dictionary_loc}")
    OH_ALL = pd.read_csv(dictionary_loc,index_col=0)
    N_CHAR = len(flatten_list)
    return OH_ALL, N_CHAR,max(MAX_LEN)

def data_preprocessing(location, ppty, lang, extenstion=True,flip=True):
    #  if ppty == 'S0': separator=separator_smiles
    separator = separator_select(lang)
    raw ,ans = read_data(location, ppty, lang)
    data = {}
    for num in range(len(raw)):
        A = list(ans)[num]
        S = list(raw)[num]
        
        s_list = separator(S)
        data[S] = [s_list,A]
    processed_data = {}
    for d in data:
        processed_data[d]=data[d]
        s_list = data[d][0]
        constant = data[d][1]
        at_s_list = at_list(s_list)
        hash_s_list = hash_list(at_s_list)
        wot_HT = hash_s_list[1:-1]
        #  if extenstion:
        for j in range(len(wot_HT)-1):
            if not extenstion: j = -1
            new_hash_list = ['H'] + wot_HT[j+1:]+wot_HT[:j+1] + ['T']
            #  print(new_hash_list)
            new_d = ''.join(new_hash_list)
            new_s_list = separator(new_d)
            if extenstion: processed_data[new_d] = [new_s_list, constant]
            if flip:
                Flip_hash_list = wot_HT[j+1:]+wot_HT[:j+1]
                Flip_hash_list.reverse()
                new_hash_list  = ['H'] + Flip_hash_list + ['T']
                new_d = ''.join(new_hash_list)
                new_s_list = separator(new_d)
                processed_data[new_d] = [new_s_list, constant]                   
            if not extenstion: break
    #  print(processed_data)
    return processed_data

def generate_dataset(raw_data, OH_All ,MAX_LEN, N_CHAR, Flip=False,pre=False):
    OH = np.zeros((len(raw_data),MAX_LEN,N_CHAR),dtype='uint8')
    LP = np.zeros((len(raw_data),1),dtype='float')

    for c,d in enumerate(raw_data):
        #  print(d)
        L = raw_data[d][0]
        OH_tmp = []
        for l in L:
            OH_tmp.append(list(OH_All[l]))
            #  print(list(OH_All[l]))
        #  if i in test_list:
        if pre:
            OH[c, -len(OH_tmp):,:N_CHAR]   = np.array(OH_tmp,dtype='uint8')
            OH[c, :-len(OH_tmp), :N_CHAR] = np.array(list(OH_All['empty']),dtype='uint8')
            LP[c,:] = raw_data[d][1]
        else:
            print(len(OH_tmp))
            print(c,d)
            OH[c,:len(OH_tmp),:N_CHAR]   = np.array(OH_tmp,dtype='uint8')
            OH[c, len(OH_tmp):, :N_CHAR] = np.array(list(OH_All['empty']),dtype='uint8')
            LP[c,:] = raw_data[d][1]
    return OH,LP




if __name__ == "__main__":
    #  ,c print(read_data('../data/data_smiles.csv','Tg','S0'))
    print(data_preprocessing('../data/data_smiles.csv','exp','S2'))
