import time
import yaml
import torch
import argparse

def configurate(explicit_args):
    parser = argparse.ArgumentParser()
    
    ### Basic
    parser.add_argument('--defaults',     type=str, default='./config/defaults.yml')
    parser.add_argument('--rand_seed',    type=int)
    parser.add_argument('--dataset',      type=str)
    parser.add_argument('--epoch',        type=int)
    parser.add_argument('--trn_batch',    type=int)
    parser.add_argument('--tensorboard',  type=int)

    ### Early-stop
    parser.add_argument('--tst_step',     type=int)
    parser.add_argument('--patience',     type=int)
    parser.add_argument('--criterion',    type=str)
    
    ### Test
    parser.add_argument('--metrics',      type=str)
    parser.add_argument('--ks',           type=str)
    parser.add_argument('--tst_batch',    type=int)
    
    ### Tunable
    parser.add_argument('--embed_reg',    type=float)
    parser.add_argument('--lr',           type=float)
    
    ### Develop (General)
    parser.add_argument('--comment',      type=str)
    parser.add_argument('--summary',      type=str)
    parser.add_argument('--model',        type=str)
    parser.add_argument('--loss',         type=str)
    parser.add_argument('--trainer',      type=str)
    
    ### Develop (Specific)
    parser.add_argument('--embed_dim',    type=int)
    parser.add_argument('--layer_num',    type=int)
    parser.add_argument('--keep_rate',    type=float)
    # parser.add_argument('--gamma',        type=float, help='[DirectAU] weight of uniformity loss')

    ### UltraGCN
    parser.add_argument('--w1',    type=float)
    parser.add_argument('--w2',    type=float)
    parser.add_argument('--w3',    type=float)
    parser.add_argument('--w4',    type=float)
    
    parser.add_argument('--negative_num',    type=int)
    parser.add_argument('--negative_weight',    type=int)
    parser.add_argument('--gamma',    type=float, help='[UltraGCN] loss')
    parser.add_argument('--lambda_',    type=float)
    parser.add_argument('--sampling_sift_pos',    type=bool)

    ### SimpleX
    parser.add_argument('--neg_num',      type=int, help='The number of negative sampling in `MultiNegTrnData`.')
    parser.add_argument('--score',        type=str, help='Possible: [cosine, dot]')
    parser.add_argument('--aggregator',        type=str, help='Possible: [mean, user_attention, self_attention]')
    parser.add_argument('--fusing_weight',     type=float)
    parser.add_argument('--attention_dropout', type=float)
    parser.add_argument('--history_num',  type=int)
    parser.add_argument('--dropout',      type=int)
    parser.add_argument('--neg_weight',   type=float)
    parser.add_argument('--margin', type=float)

    args = parser.parse_args(explicit_args)
    
    # Use default values if args are not given
    with open(args.defaults, mode='r', encoding='utf-8') as f:
        defaults = yaml.safe_load(f.read())
    for arg, value in args.__dict__.items():
        if (value is None) and (arg in defaults):
            setattr(args, arg, defaults[arg])
    
    # Convert comma-separated string to list
    def str_to_list(string, elem_type):
        return [elem_type(x) for x in string.split(",")]
    def is_str(x):
        return isinstance(x, str)
    args.metrics = str_to_list(args.metrics, str) if is_str(args.metrics) else args.metrics
    args.ks = str_to_list(args.ks, int) if is_str(args.ks) else args.ks
    args.criterion = str_to_list(args.criterion, int) if is_str(args.criterion) else args.criterion
    
    # Automatically set args
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.path = f'./log/{time.strftime("%m%d-%H:%M:%S")}-{args.comment}'
    
    return args

EXPLICIT = False # True if you want to give arguments without using command line
explicit_args = [
    "--comment", "test",
    "--model", "lightgcn",
    "--loss", "bpr",
    "--embed_dim", "64",
    "--layer_num", "4",
]
explicit_args = explicit_args if EXPLICIT else None
args = configurate(explicit_args)