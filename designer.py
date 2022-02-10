#!/usr/bin/env python
import torch
from rdkit import rdBase
import utils
import pandas as pd
from torch.utils.data import DataLoader
import os

import argparse
import json


rdBase.DisableLog('rdApp.error')
torch.set_num_threads(1)

def DesignArgParser(txt=None):
    """ Define and read command line arguments """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains folders 'data' and 'output'")
    parser.add_argument('-i', '--input', type=str, default='ligand_mf_brics',
                        help="Prefix of ligand input file")   
    parser.add_argument('-ft_model', '--finetuned_model', type=str, default='ligand_mf_brics_gpt_128',
                        help="Name of finetuned model file without .pkg extension")

    parser.add_argument('-gpu', '--gpu', type=str, default='1,2,3,4',
                        help="List of GPUs") 

    parser.add_argument('-a', '--algorithm', type=str, default='smiles',
                        help="Generator algorith: 'smiles' or 'graph' ")
    parser.add_argument('-m', '--method', type=str, default='gpt',
                        help="In case of SMILES-based generator, method: 'gpt', 'ved' or 'attn'") 
    parser.add_argument('-bs', '--batch_size', type=int, default=1048,
                        help="Batch size")
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-2,
                        help="Exploring rate")
    parser.add_argument('-bet', '--beta', type=float, default=0.0,
                        help="Reward baseline")
    parser.add_argument('-s', '--scheme', type=str, default='WS',
                        help="Reward calculation scheme: 'WS', 'PR', 'CD'")
    
    parser.add_argument('-et', '--env_task', type=str, default='REG',
                        help="Environment-predictor task: 'REG' or 'CLS'")
    parser.add_argument('-ea', '--env_alg', type=str, default='RF',
                        help="Environment-predictor algorith: 'RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN', 'MT_DNN'")
    
    parser.add_argument('-ta', '--active_targets', type=str, nargs='*', default=['P29274', 'P29275', 'P30542','P0DMS8'],
                        help="Target IDs for which activity is desirable")
    parser.add_argument('-ti', '--inactive_targets', type=str, nargs='*', default=[],
                        help="Target IDs for which activity is undesirable")
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
    
    args.targets = args.active_targets + args.inactive_targets

    return args

def Design(args):
    
    args.git_commit = utils.commit_hash() 
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    
    utils.devices = eval(args.gpu) if ',' in args.gpu else [eval(args.gpu)]
    torch.cuda.set_device(utils.devices[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load data
    if args.algorithm == 'smiles':
        if args.method == 'gpt':
            voc = utils.Voc( args.base_dir + '/data/chembl_voc.txt', src_len=100, trg_len=100)
        else:
            voc = utils.VocSmiles( args.base_dir + '/data/chembl_voc.txt', max_len=100)
        data = pd.read_table(args.base_dir + '/data/' + args.input + '_test_smi.txt')
        data = voc.encode([seq.split(' ')[:-1] for seq in data.values])
        loader = DataLoader(data, batch_size=args.batch_size)
    elif args.algorithm == 'graph':
        voc = utils.VocGraph( args.base_dir + '/data/voc_graph.txt')
        data = pd.read_table(args.base_dir + '/data/' + args.input + '_test_code.txt')
        data = torch.from_numpy(data.values).long().view(len(data), voc.max_len, -1)
        loader = DataLoader(data, batch_size=args.batch_size)   
    
    # Load finetuned model
    ft_path = args.base_dir + '/generators/' + args.finetuned_model + '.pkg'
    agent = utils.SetAlgorithm(voc, args.algorithm, args.method)
    agent.load_state_dict(torch.load( ft_path, map_location=utils.dev))
    
    # Set up environment-predictor
    objs, keys = utils.CreateDesirabilityFunction(args)
    mods, ths = utils.SetModes(args.scheme, keys, args.env_task, args.active_targets, args.inactive_targets)
    env =  utils.Env(objs=objs, mods=None, keys=keys, ths=ths)
    
    out = args.base_dir + '/predictions/' + args.finetuned_model + '.tsv'
    
    # Generate molecules and save them
    frags, smiles, scores = agent.evaluate(loader, repeat=10, method=env)
    scores['Frags'], scores['SMILES'] = frags, smiles
    scores.to_csv(out, index=False, sep='\t')


if __name__ == "__main__":
    
    args = DesignArgParser()
    design(args)