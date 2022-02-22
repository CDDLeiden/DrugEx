#!/usr/bin/env python
import os
import json
import torch
import argparse
import pandas as pd

from rdkit import rdBase
from torch.utils.data import DataLoader

import utils
from train import SetAlgorithm, CreateDesirabilityFunction, SetModes


rdBase.DisableLog('rdApp.error')
torch.set_num_threads(1)

def DesignArgParser(txt=None):
    """ Define and read command line arguments """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains folders 'data' and 'output'")
    parser.add_argument('-ft_model', '--finetuned_model', type=str, default='ligand_mf_brics_gpt_128',
                        help="Name of finetuned model file without .pkg extension")
    parser.add_argument('-gpu', '--gpu', type=str, default='1,2,3,4',
                        help="List of GPUs") 
    parser.add_argument('-bs', '--batch_size', type=int, default=1048,
                        help="Batch size")
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
        
    # Load parameters generator/environment from trained model
    with open(args.base_dir + '/generators/' + args.finetuned_model + '.json') as f:
        pt_params = json.load(f)
    
    args.algorithm = pt_params['algorithm']
    args.epsilon = pt_params['epsilon']
    args.beta = pt_params['beta']
    args.scheme = pt_params['scheme']
    args.env_alg = pt_params['env_alg']
    args.env_task = pt_params['env_task']
    args.active_targets = pt_params['active_targets']
    args.inactive_targets = pt_params['inactive_targets']
    args.qed = pt_params['qed']
    args.input = pt_params['input']
    
    args.targets = args.active_targets + args.inactive_targets

    return args

def Design(args):
    
    args.git_commit = utils.commit_hash(os.path.dirname(os.path.realpath(__file__)))
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    
    utils.devices = eval(args.gpu) if ',' in args.gpu else [eval(args.gpu)]
    torch.cuda.set_device(utils.devices[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if not os.path.exists(args.base_dir + '/new_molecules'):
        os.makedirs(args.base_dir + '/new_molecules')
    
    # Load data
    if args.algorithm == 'graph':
        voc = utils.VocGraph( args.base_dir + '/data/voc_graph.txt')
        data = pd.read_table(args.base_dir + '/data/' + args.input + '_test_code.txt')
        data = torch.from_numpy(data.values).long().view(len(data), voc.max_len, -1)
        loader = DataLoader(data, batch_size=args.batch_size)  
    else:
        if args.method == 'gpt':
            voc = utils.Voc( args.base_dir + '/data/chembl_voc.txt', src_len=100, trg_len=100)
        else:
            voc = utils.VocSmiles( args.base_dir + '/data/chembl_voc.txt', max_len=100)
        data = pd.read_table(args.base_dir + '/data/' + args.input + '_test_smi.txt')
        data = voc.encode([seq.split(' ')[:-1] for seq in data.values])
        loader = DataLoader(data, batch_size=args.batch_size)
    
    # Load finetuned model
    ft_path = args.base_dir + '/generators/' + args.finetuned_model + '.pkg'
    agent = SetAlgorithm(voc, args.algorithm)
    agent.load_state_dict(torch.load( ft_path, map_location=utils.dev))
    
    # Set up environment-predictor
    objs, keys = CreateDesirabilityFunction(args)
    mods, ths = SetModes(args.scheme, keys, args.env_task, args.active_targets, args.inactive_targets, args.qed)
    env =  utils.Env(objs=objs, mods=None, keys=keys, ths=ths)
    
    out = args.base_dir + '/new_molecules/' + args.finetuned_model + '.tsv'
    
    # Generate molecules and save them
    frags, smiles, scores = agent.evaluate(loader, repeat=10, method=env)
    scores['Frags'], scores['SMILES'] = frags, smiles
    scores.to_csv(out, index=False, sep='\t')


if __name__ == "__main__":
    
    args = DesignArgParser()
    Design(args)
