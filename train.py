#!/usr/bin/env python
import os
import json
import time
import torch
import utils
import argparse

import pandas as pd
from shutil import copy2

    
def GeneratorArgParser(txt=None):
    """ Define and read command line arguments """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains folders 'data' (and 'envs')")
    parser.add_argument('-i', '--input', type=str, default=None,
                        help="Prefix of ligand input file. If --pretraining on, default is 'chembl_mf_brics' else 'ligand_mf_brics' ")   
    parser.add_argument('-pt_model', '--pretrained_model', type=str, default='chembl_mf_brics_gpt_128',
                        help="Name of pretrained model file without .pkg extension")

    parser.add_argument('-gpu', '--gpu', type=str, default='1,2,3,4',
                        help="List of GPUs") 

    parser.add_argument('-pt', '--pretraining', action='store_true', 
                        help="If on, does pretraining, else fine tunes the model with reinforcement learning") 

    parser.add_argument('-a', '--algorithm', type=str, default='smiles',
                        help="Generator algorith: 'smiles' or 'graph' ")
    parser.add_argument('-m', '--method', type=str, default='gpt',
                        help="Method: 'gpt', 'ved' or 'attn'") 
    parser.add_argument('-bs', '--batch_size', type=int, default=128,
                        help="Batch size")
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-2,
                        help="Exploring rate")
    parser.add_argument('-bet', '--beta', type=float, default=0.0,
                        help="Reward baseline")
    parser.add_argument('-s', '--scheme', type=str, default='WS',
                        help="Reward calculation scheme: 'WS', 'PR', CD")
    
    parser.add_argument('-et', '--env_task', type=str, default='REG',
                        help="Environment-predictor task: 'REG' or 'CLS'")
    parser.add_argument('-ea', '--env_alg', type=str, default='RF',
                        help="Environment-predictor algorith: 'RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN', 'MT_DNN'")
    
    parser.add_argument('-ta', '--active_targets', type=str, nargs='*', default=['P29274', 'P29275', 'P30542','P0DMS8'],
                        help="Targets")
    parser.add_argument('-ti', '--inactive_targets', type=str, nargs='*', default=[],
                        help="Targets")
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
    
    if args.input is None:
        if args.pretraining:
            args.input = 'chembl_mf_brics'
        else:
            args.input = 'ligand_mf_brics'
    
    args.targets = args.active_targets + args.inactive_targets

    return args

def PreTrain(args):
    
    print('Pretraining {}-based model with {} data...'.format(args.algorithm, args.input))
    if args.algorithm == 'smiles':
        voc, train_loader, valid_loader = utils.DataPreparationSmiles(args)
    elif args.algorithm == 'graph':
        voc, train_loader, valid_loader = utils.DataPreparationGraph(args)
        
    agent = utils.SetAlgorithm(voc, args.algorithm, method=args.method)
    out = '%s/generators/%s_%s_%d' % (args.base_dir, args.input, args.algorithm, args.batch_size)
    agent.fit(train_loader, valid_loader, epochs=args.epochs, out=out)
    
    
def RLTrain(args):
    
    pt_path = args.base_dir + '/generators/' + args.pretrained_model + '.pkg'
    
    print('Finetuning {}-based pretrained {} model with {} data...'.format(args.algorithm, args.pretrained_model, args.input))
    # Load input data
    if args.algorithm == 'smiles':
        voc, train_loader, valid_loader = utils.DataPreparationSmiles(args)    
    elif args.algorithm == 'graph':
        voc, train_loader, valid_loader = utils.DataPreparationGraph(args)
    
    # Initialize agent and prior by loading pretrained model
    agent = utils.SetAlgorithm(voc, args.algorithm, method=args.method)        
    agent.load_state_dict(torch.load( pt_path, map_location=utils.dev))
    prior = utils.SetAlgorithm(voc, args.algorithm, method=args.method)        
    prior.load_state_dict(torch.load( pt_path, map_location=utils.dev))  

    # Initialize evolver algorithm
    evolver = utils.InitializeEvolver(agent, prior, args)
    
    # Chose the desirability function
    # 1. Load predictors per task : targets + QED
    # 2. Set clipped scores and threasholds per task depending wanted quality
    objs, keys = utils.CreateDesirabilityFunction(args)
    mods, ths = utils.SetModes(evolver.scheme, keys, args.env_task, args.active_targets, args.inactive_targets)
    
    # Set Evolver's environment-predictor
    evolver.env = utils.Env(objs=objs, mods=mods, keys=keys, ths=ths)

    root = '%s/generators/%s_%s' % (args.base_dir, args.algorithm, time.strftime('%y%m%d_%H%M%S', time.localtime()))

    os.mkdir(root)

    # No idea what the following lines should do as the files that should be copied does not exist !?!?!?
#     copy2(args.algorithm + '_ex.py', root)
#     copy2(args.algorithm + '.py', root)

    # import evolve as agent
    evolver.out = root + '/%s_%s_%s_%.0e' % (args.algorithm, evolver.scheme, args.env_task, evolver.epsilon)
    evolver.fit(train_loader, test_loader=valid_loader, epochs=args.epochs)

def TrainGenerator(args):
    
    args.git_commit = utils.commit_hash() 
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    
    utils.devices = eval(args.gpu) if ',' in args.gpu else [eval(args.gpu)]
    torch.cuda.set_device(utils.devices[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if not os.path.exists(args.base_dir + '/generators'):
        os.makedirs(args.base_dir + '/generators')  
    
    if args.pretraining:
        PreTrain(args)
    else:
        RLTrain(args)   

if __name__ == "__main__":
    
    args = GeneratorArgParser()
    args.epochs = 2 #1000
    TrainGenerator(args)