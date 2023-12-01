import glob
import os

ns = ['1', '2', '3', '4', '5']
for n in ns:
    for file in glob.glob(f'/home/Gabriella/polymagic_Tg/data/review_top_{n}*Tg*'):
        if len(file) == 68:
            continue
            name = file.split('/')[-1]
            new = name.replace('review', 'Extended')
            new = new.replace('all_', '')
            new = new.replace('nor_', '')
            new = new.replace('S0', 'SMILES')
            new = new.replace('S2', 'HAPPY')
        elif len(file) == 72:
            continue
            print(file,len(file))
            name = file.split('/')[-1]
            new = name.replace('review', 'Extended')
            new = new.replace('_top', '')
            new = new.replace('all_', '')
            new = new.replace('nor_', '')
            new = new.replace('S0', 'SMILES')
            new = new.replace('S2', 'topo-HAPPY')
            print(new)
            #  os.system(f'cp {file} ./{new}')
        elif len(file) == 73:
            name = file.split('/')[-1]
            new = name.replace('review', 'Extended')
            new = new.replace('_top', '')
            new = new.replace('all_', '')
            new = new.replace('nor_', '')
            new = new.replace('S0', 'SMILES')
            new = new.replace('S2R', 'refined_topo-HAPPY')

            print(new)
            os.system(f'cp {file} ./{new}')
