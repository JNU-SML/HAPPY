import argparse

parser = argparse.ArgumentParser(description='Hyperparameter parser')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--repre',      type=str)
parser.add_argument('--ppty',       type=str)
parser.add_argument('--dataset',    type=str)
parser.add_argument('--trainset',   type=str)
parser.add_argument('--trainset_n', type=int)
parser.add_argument('--exten',      type=str2bool)
parser.add_argument('--train',      type=str2bool)
parser.add_argument('--LR',         type=float)
parser.add_argument('--GPU',        type=int)
parser.add_argument('--split',     type=int)
#  parser.add_argument('--name',      type=str)
parser.add_argument('--top',       type=str2bool)


#  parser.add_argument('--node_n',     type=int)

#  parser.add_argument('--train', type=bool)








