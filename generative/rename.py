import glob
import os

for f in glob.glob('./*final*S0*400*.h5'):
    name = "SMILES_"+f.split('_')[3]+".h5"
    print(name)
    os.rename(f, name)
    print(f)

