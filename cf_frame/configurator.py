import time
import yaml
import torch
import argparse

def configurate():
    parser = argparse.ArgumentParser()
    
    ### Essential
    parser.add_argument('--comment', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--trainer', type=str)
    
    ### Extra
    parser.add_argument('--default', type=str, default='default')
    parser.add_argument('--rand_seed', type=int)
    parser.add_argument('--summary', type=str)
    parser.add_argument('--folder', type=str)
    
    ### Train
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--trn_batch', type=int)
    parser.add_argument('--tensorboard', type=int)

    ### Early-stop
    parser.add_argument('--tst_step', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--criterion', type=str)
    
    ### Test
    parser.add_argument('--metrics', type=str)
    parser.add_argument('--ks', type=str)
    parser.add_argument('--tst_batch', type=int)
    
    ### Hyperparameters
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--layer_num', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--keep_rate', type=float)

    ### Loss
    parser.add_argument('--embed_reg', type=float, help='L2 regularization for embeddings')
    parser.add_argument('--uniform', type=float, help='[DirectAU] weight of uniformity loss')
    
    ### GFCF & PGSP
    parser.add_argument('--ideal', type=float, help='weight of ideal low-pass filter')

    ### UltraGCN
    parser.add_argument('--w1', type=float)
    parser.add_argument('--w2', type=float)
    parser.add_argument('--w3', type=float)
    parser.add_argument('--w4', type=float)
    parser.add_argument('--negative_num', type=int)
    parser.add_argument('--negative_weight', type=int)
    parser.add_argument('--gamma', type=float, help='[UltraGCN] loss')
    parser.add_argument('--lambda_', type=float)
    parser.add_argument('--sampling_sift_pos', type=bool)

    ### SimpleX
    parser.add_argument('--neg_num', type=int, help='The number of negative sampling in `MultiNegTrnData`.')
    parser.add_argument('--score', type=str, help='Possible: [cosine, dot]')
    parser.add_argument('--aggregator', type=str, help='Possible: [mean, user_attention, self_attention]')
    parser.add_argument('--fusing_weight', type=float)
    parser.add_argument('--attention_dropout', type=float)
    parser.add_argument('--history_num', type=int)
    parser.add_argument('--dropout', type=int)
    parser.add_argument('--neg_weight', type=float)
    parser.add_argument('--margin', type=float)

    # ### GSP components analysis
    # parser.add_argument('--normalize', type=int)
    # parser.add_argument('--coeffs', type=str) # Poly
    # parser.add_argument('--freq_num', type=int) # Cutoff
    # parser.add_argument('--freq_threshold', type=str) # Cutoff
    # parser.add_argument('--filter', type=str) # Cutoff
    # parser.add_argument('--hyp', type=float) # Cutoff

    ### BSPM 
    parser.add_argument('--solver_idl', type=str, default='euler', help="heat equation solver")
    parser.add_argument('--solver_blr', type=str, default='euler', help="ideal low-pass solver")
    parser.add_argument('--solver_shr', type=str, default='euler', help="sharpening solver")
    parser.add_argument('--K_idl', type=int, default=1, help='T_idl / \tau')
    parser.add_argument('--T_idl', type=float, default=1, help='T_idl')
    parser.add_argument('--K_b', type=int, default=1, help='T_b / \tau')
    parser.add_argument('--T_b', type=float, default=1, help='T_b')
    parser.add_argument('--K_s', type=int, default=1, help='T_s / \tau')
    parser.add_argument('--T_s', type=float, default=1, help='T_s')
    parser.add_argument('--factor_dim', type=int, default=256, help='factor_dim')
    parser.add_argument('--idl_beta', type=float, default=0.3, help='beta')
    parser.add_argument('--final_sharpening', type=lambda x: x.lower() in ('true', '1'), default=True, choices=[True, False])
    parser.add_argument('--sharpening_off', type=lambda x: x.lower() in ('true', '1'), default=False, choices=[True, False])
    parser.add_argument('--t_point_combination', type=lambda x: x.lower() in ('true', '1'), default=False, choices=[True, False])
    
    ### PolyFilter
    parser.add_argument('--order', type=int)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--ideal_num', type=int)
    parser.add_argument('--ideal_weight', type=float)
    parser.add_argument('--nonparam', type=int)                                     
                                                                            
    args = parser.parse_args()
    
    # Use default values if args are not given
    with open(f'./config/{args.default}.yml', mode='r', encoding='utf-8') as f:
        default = yaml.safe_load(f.read())
    for arg, value in args.__dict__.items():
        if (value is None) and (arg in default):
            setattr(args, arg, default[arg])
    
    # Convert comma-separated string to list
    def str_to_list(string, elem_type):
        return [elem_type(x) for x in string.split(",")]
    def is_str(x):
        return isinstance(x, str)
    args.metrics = str_to_list(args.metrics, str) if is_str(args.metrics) else args.metrics
    args.ks = str_to_list(args.ks, int) if is_str(args.ks) else args.ks
    args.criterion = str_to_list(args.criterion, int) if is_str(args.criterion) else args.criterion
    args.coeffs = str_to_list(args.coeffs, float) if is_str(args.coeffs) else args.coeffs
    
    # Automatically set args
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.path = f'./log/' if args.folder is None else f'./log/{args.folder}/'
    args.path += f'{time.strftime("%m%d-%H:%M:%S")}-{args.comment}'
    
    return args

args = configurate()

''' <<< Tip >>> Arguments can be set without using command line (in case of Jupyter Notebook).
import sys
sys.argv = ['configurator.py', '--comment', 'jupyter', '--dataset', 'gowalla']
'''