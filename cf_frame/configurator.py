import time
import yaml
import torch
import argparse


def configurate():
    """ Common Arguments """
    common_parser = argparse.ArgumentParser(add_help=False)
    
    common_parser.add_argument('--comment', type=str)
    common_parser.add_argument('--dataset', type=str)
    common_parser.add_argument('--loss', type=str)
    common_parser.add_argument('--trainer', type=str)
    
    common_parser.add_argument('--default', type=str, default='default')
    common_parser.add_argument('--rand_seed', type=int)
    common_parser.add_argument('--summary', type=str)
    common_parser.add_argument('--folder', type=str)
    
    # train
    common_parser.add_argument('--epoch', type=int)
    common_parser.add_argument('--trn_batch', type=int)
    common_parser.add_argument('--tensorboard', type=int)

    # test
    common_parser.add_argument('--tst_step', type=int)
    common_parser.add_argument('--patience', type=int)
    common_parser.add_argument('--criterion', type=str)
    
    # metric
    common_parser.add_argument('--metrics', type=str)
    common_parser.add_argument('--ks', type=str)
    common_parser.add_argument('--tst_batch', type=int)
    
    # model
    common_parser.add_argument('--embed_dim', type=int)
    common_parser.add_argument('--layer_num', type=int)
    common_parser.add_argument('--lr', type=float)
    common_parser.add_argument('--keep_rate', type=float)

    # loss
    common_parser.add_argument('--embed_reg', type=float, help='L2 regularization for embeddings')
    common_parser.add_argument('--uniform', type=float, help='[DirectAU] weight of uniformity loss')

    """ Model Arguments """
    parser = argparse.ArgumentParser(description='Main parser')
    subparsers = parser.add_subparsers(dest='model', help="Model selection")

    # LightGCN & MF
    lightgcn_parser = subparsers.add_parser('lightgcn', parents=[common_parser], help="lightgcn model operations")
    mf_parser = subparsers.add_parser('mf', parents=[common_parser], help="mf model operations")

    # TurboCF
    turbocf_parser = subparsers.add_parser('turbocf', parents=[common_parser], help="turbocf model operations")
    turbocf_parser.add_argument("--alpha", type=float, default=0.5, help="For normalization of R")
    turbocf_parser.add_argument("--power", type=float, default=1, help="For normalization of P")
    turbocf_parser.add_argument("--filter", type=float, default=1, help="1:linear, 2: 2nd-order, 3: Poly.approx of ideal LPF")
    
    # BSPM
    bspm_parser = subparsers.add_parser('bspm', parents=[common_parser], help="bspm model operations")
    bspm_parser.add_argument('--solver_idl', type=str, default='euler', help="heat equation solver")
    bspm_parser.add_argument('--solver_blr', type=str, default='euler', help="ideal low-pass solver")
    bspm_parser.add_argument('--solver_shr', type=str, default='euler', help="sharpening solver")
    bspm_parser.add_argument('--K_idl', type=int, default=1, help='T_idl / \tau')
    bspm_parser.add_argument('--T_idl', type=float, default=1, help='T_idl')
    bspm_parser.add_argument('--K_b', type=int, default=1, help='T_b / \tau')
    bspm_parser.add_argument('--T_b', type=float, default=1, help='T_b')
    bspm_parser.add_argument('--K_s', type=int, default=1, help='T_s / \tau')
    bspm_parser.add_argument('--T_s', type=float, default=1, help='T_s')
    bspm_parser.add_argument('--factor_dim', type=int, default=256, help='factor_dim')
    bspm_parser.add_argument('--idl_beta', type=float, default=0.3, help='beta')
    bspm_parser.add_argument('--final_sharpening', type=lambda x: x.lower() in ('true', '1'), default=True, choices=[True, False])
    bspm_parser.add_argument('--sharpening_off', type=lambda x: x.lower() in ('true', '1'), default=False, choices=[True, False])
    bspm_parser.add_argument('--t_point_combination', type=lambda x: x.lower() in ('true', '1'), default=False, choices=[True, False])

    # UltraGCN
    ultragcn_parser = subparsers.add_parser('ultragcn', parents=[common_parser], help="ultragcn model operations")
    ultragcn_parser.add_argument('--w1', type=float)
    ultragcn_parser.add_argument('--w2', type=float)
    ultragcn_parser.add_argument('--w3', type=float)
    ultragcn_parser.add_argument('--w4', type=float)
    ultragcn_parser.add_argument('--negative_num', type=int)
    ultragcn_parser.add_argument('--negative_weight', type=int)
    ultragcn_parser.add_argument('--gamma', type=float, help='[UltraGCN] loss')
    ultragcn_parser.add_argument('--lambda', type=float)
    ultragcn_parser.add_argument('--sampling_sift_pos', type=bool)

    # GFCF
    gfcf_parser = subparsers.add_parser('gfcf', parents=[common_parser], help="gfcf model operations")
    gfcf_parser.add_argument('--ideal', type=float, help='weight of ideal low-pass filter')

    # PGSP
    pgsp_parser = subparsers.add_parser('pgsp', parents=[common_parser], help="pgsp model operations")
    pgsp_parser.add_argument('--ideal', type=float, help='weight of ideal low-pass filter')
    
    # SimpleX
    simplex_parser = subparsers.add_parser('simplex', parents=[common_parser], help="simplex model operations")
    simplex_parser.add_argument('--neg_num', type=int, help='The number of negative sampling in `MultiNegTrnData`.')
    simplex_parser.add_argument('--score', type=str, help='Possible: [cosine, dot]')
    simplex_parser.add_argument('--aggregator', type=str, help='Possible: [mean, user_attention, self_attention]')
    simplex_parser.add_argument('--fusing_weight', type=float)
    simplex_parser.add_argument('--attention_dropout', type=float)
    simplex_parser.add_argument('--history_num', type=int)
    simplex_parser.add_argument('--dropout', type=int)
    simplex_parser.add_argument('--neg_weight', type=float)
    simplex_parser.add_argument('--margin', type=float)

    # SGFCF
    sgfcf_parser = subparsers.add_parser('sgfcf', parents=[common_parser], help="sgfcf model operations")
    sgfcf_parser.add_argument('--k', type=int, default=100, help='The number of required features')  
    sgfcf_parser.add_argument('--beta_1', type=float, default=1.0, help='coef for the filter')
    sgfcf_parser.add_argument('--beta_2', type=float, default=1.0, help='coef for the filter')
    sgfcf_parser.add_argument('--alpha', type=float, default=0.0, help='param for G^2N')
    sgfcf_parser.add_argument('--eps', type=float, default=0.5, help='param for G^2N')
    sgfcf_parser.add_argument('--gamma', type=float, default=1.0, help='weight for non-low frequency')

    # PF_Bern
    pf_bern_parser = subparsers.add_parser('pf_bern', parents=[common_parser], help="pf_bern model operations")
    pf_bern_parser.add_argument('--order', type=int)
    pf_bern_parser.add_argument('--weights', type=str)
    pf_bern_parser.add_argument('--ideal_num', type=int)
    pf_bern_parser.add_argument('--ideal_weight', type=float)
    pf_bern_parser.add_argument('--nonparam', type=int)

    # PF_Cheb
    pf_cheb_parser = subparsers.add_parser('pf_cheb', parents=[common_parser], help="pf_cheb model operations")
    pf_cheb_parser.add_argument('--order', type=int)
    pf_cheb_parser.add_argument('--weights', type=str)
    pf_cheb_parser.add_argument('--ideal_num', type=int)
    pf_cheb_parser.add_argument('--ideal_weight', type=float)
    pf_cheb_parser.add_argument('--nonparam', type=int)

    # PF_ChebII_Prefix
    pf_chebii_prefix_parser = subparsers.add_parser('pf_chebii_prefix', parents=[common_parser], help="pf_chebii_prefix model operations")
    pf_chebii_prefix_parser.add_argument('--order', type=int)
    pf_chebii_prefix_parser.add_argument('--weights', type=str)
    pf_chebii_prefix_parser.add_argument('--ideal_num', type=int)
    pf_chebii_prefix_parser.add_argument('--ideal_weight', type=float)
    pf_chebii_prefix_parser.add_argument('--nonparam', type=int)

    # PF_ChebII_Prefix
    pf_chebii_parser = subparsers.add_parser('pf_chebii', parents=[common_parser], help="pf_chebii model operations")
    pf_chebii_parser.add_argument('--order', type=int)
    pf_chebii_parser.add_argument('--weights', type=str)
    pf_chebii_parser.add_argument('--ideal_num', type=int)
    pf_chebii_parser.add_argument('--ideal_weight', type=float)
    pf_chebii_parser.add_argument('--nonparam', type=int)

    # PF_Mono
    pf_mono_parser = subparsers.add_parser('pf_mono', parents=[common_parser], help="pf_mono model operations")
    pf_mono_parser.add_argument('--order', type=int)
    pf_mono_parser.add_argument('--weights', type=str)
    pf_mono_parser.add_argument('--ideal_num', type=int)
    pf_mono_parser.add_argument('--ideal_weight', type=float)
    pf_mono_parser.add_argument('--nonparam', type=int)
    
    # GSP_Poly
    gsp_poly_parser = subparsers.add_parser('gsp_poly', parents=[common_parser], help="gsp_poly model operations")
    gsp_poly_parser.add_argument('--normalize', type=int)
    gsp_poly_parser.add_argument('--coeffs', type=str)

    # GSP_Cutoff
    gsp_cutoff_parser = subparsers.add_parser('gsp_cutoff', parents=[common_parser], help="gsp_cutoff model operations")
    gsp_cutoff_parser.add_argument('--freq_num', type=int) # Cutoff
    gsp_cutoff_parser.add_argument('--freq_threshold', type=str) # Cutoff
    gsp_cutoff_parser.add_argument('--filter', type=str) # Cutoff
    gsp_cutoff_parser.add_argument('--hyp', type=float) # Cutoff
    
    # GSP_Norm
    gsp_norm_parser = subparsers.add_parser('gsp_norm', parents=[common_parser], help="gsp_norm model operations")
    gsp_norm_parser.add_argument('--normalize', type=int)

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
    
    if 'pf' in args.model:
        args.coeffs = str_to_list(args.coeffs, float) if is_str(args.coeffs) else args.coeffs
        args.weights = str_to_list(args.weights, float) if is_str(args.weights) else args.weights
    
    # Automatically set args
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.path = f'./log/' if args.folder is None else f'./log/{args.folder}/'
    args.path += f'{time.strftime("%m%d-%H%M%S")}-{args.comment}'
    
    return args

args = configurate()

''' <<< Tip >>> Arguments can be set without using command line (in case of Jupyter Notebook).
import sys
sys.argv = ['configurator.py', '--comment', 'jupyter', '--dataset', 'gowalla']
'''