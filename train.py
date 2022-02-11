#!/usr/bin/env python
import os
import json
import time
import torch
import utils
import argparse
import pandas as pd

from shutil import copy2
from torch.utils.data import DataLoader, TensorDataset

from models import generator, GPT2Model, GraphModel
from models.explorer import SmilesExplorer, GraphExplorer

    
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
                        help="Generator algorithm: 'smiles' or 'graph' ")
    parser.add_argument('-m', '--method', type=str, default='gpt',
                        help="In case of SMILES-based generator, method: 'gpt', 'ved' or 'attn'") 
    parser.add_argument('-bs', '--batch_size', type=int, default=128,
                        help="Batch size")
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-2,
                        help="Exploring rate")
    parser.add_argument('-bet', '--beta', type=float, default=0.0,
                        help="Reward baseline")
    parser.add_argument('-s', '--scheme', type=str, default='WS',
                        help="Reward calculation scheme: 'WS', 'PR', CD")
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help="Number of epochs")
    
    parser.add_argument('-et', '--env_task', type=str, default='REG',
                        help="Environment-predictor task: 'REG' or 'CLS'")
    parser.add_argument('-ea', '--env_alg', type=str, default='RF',
                        help="Environment-predictor algorith: 'RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN', 'MT_DNN'")
    parser.add_argument('-qed', '--qed', action='store_true',
                        help="If on, QED is used in desribality function")
    
    parser.add_argument('-ta', '--active_targets', type=str, nargs='*', default=['P29274', 'P29275', 'P30542','P0DMS8'],
                        help="Target IDs for which activity is desirable")
    parser.add_argument('-ti', '--inactive_targets', type=str, nargs='*', default=[],
                        help="Target IDs for which activity is undesirable")
    
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

def DataPreparationGraph(args):
    
    """
    Reads and preprocesses the vocabulary and input data for a graph-based generator
    
    Arguments:
        args: namespace containing command line arguments
    Returns:
        voc: atom vocabulary
        train_loader: torch DataLoader containing training data
        valid_loader: torch DataLoader containing validation data
    """
    
    data_path = args.base_dir + '/data/'

    voc = utils.VocGraph( data_path + 'voc_graph.txt', max_len=80, n_frags=4)
    
    data = pd.read_table( data_path + '%s_train_code.txt' % args.input)
    data = torch.from_numpy(data.values).long().view(len(data), voc.max_len, -1)
    train_loader = DataLoader(data, batch_size=args.batch_size * 4, drop_last=True, shuffle=True)

    test = pd.read_table( data_path + '%s_test_code.txt' % args.input)
    # test = test.sample(int(1e4))
    test = torch.from_numpy(test.values).long().view(len(test), voc.max_len, -1)
    valid_loader = DataLoader(test, batch_size=args.batch_size * 10, drop_last=True, shuffle=True)
    
    return voc, train_loader, valid_loader

def DataPreparationSmiles(args):
    
    """
    Reads and preprocesses the vocabulary and input data for a SMILES-based generator
    
    Arguments:
        args: namespace containing command line arguments
    Returns:
        voc: SMILES vocabulary
        train_loader: torch DataLoader containing training data
        valid_loader: torch DataLoader containing validation data
    """
    
    data_path = args.base_dir + '/data/'
    
    if args.method in ['gpt']:
        voc = utils.Voc( data_path + 'voc_smiles.txt', src_len=100, trg_len=100)
    else:
        voc = utils.VocSmiles( data_path + 'voc_smiles.txt', max_len=100)

    data = pd.read_table( data_path + '%s_train_smi.txt' % args.input)
    data_in = voc.encode([seq.split(' ') for seq in data.Input.values])
    data_out = voc.encode([seq.split(' ') for seq in data.Output.values])
    data_set = TensorDataset(data_in, data_out)
    train_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
    

    test = pd.read_table( data_path + '%s_test_smi.txt' % args.input)
    print(len(test), 10*args.batch_size)
    test = test.Input.drop_duplicates()
    print(len(test))
    #test = test.sample(args.batch_size * 10).values
    test_set = voc.encode([seq.split(' ') for seq in test])
    test_set = utils.TgtData(test_set, ix=[voc.decode(seq, is_tk=False) for seq in test_set])
    valid_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=test_set.collate_fn)
    
    return voc, train_loader, valid_loader

def InitializeEvolver(agent, prior, args):
    
    """
    Sets up the evolver composed of two generators.
    
    Arguments:
        agent: main generator that is optimized during training
        prior: mutation generator is kept frozen during the process
        args: namespace containing command line arguments
    Returns:
        evolver : evolver composed of two generators
        
    """
    
    if args.algorithm == 'smiles':
        evolver = SmilesExplorer(agent, mutate=prior)
    elif args.algorithm == 'graph':
        evolver = GraphExplorer(agent, mutate=prior)
        
    evolver.batch_size = args.batch_size 
    evolver.epsilon = args.epsilon 
    evolver.sigma = args.beta 
    evolver.scheme = args.scheme 
    evolver.repeat = 1   
    
    return evolver
    

def CreateDesirabilityFunction(args):
    
    """
    Sets up the objectives of the desirability function.
    
    Arguments:
        args: namespace containing command line arguments
    Returns:
        objs (lst): list of selected predictors 
        keys (lst): list of names of selected predictors
        
    """
    
    objs, keys = [], []

    for t in args.targets:
        if args.env_alg.startswith('MT_'):
            sys.exit('TO DO: using multitask model')
        else:
            path = args.base_dir + '/envs/single/' + args.env_alg + '_' + args.env_task + '_' + t + '.pkg'
        objs.append(utils.Predictor(path, type=args.env_task))
        keys.append(t)
    if args.qed :
        objs.append(utils.Property('QED'))
        keys.append('QED')
    
    return objs, keys 

def SetModes(scheme, keys, env_task, active_targets, inactive_targets):
    
    """ 
    Calculate clipped scores and threasholds for each target (and 'QED') of the desirability function 
    depending the evolver scheme ('WS' or other) and environement taks ('REG' or 'CLS').
    
    Arguments:
        scheme (str): scheme of multitarget optimisation
        keys (lst): list of predictor names
        env_task (str): type of predictor task: 'REG' or 'CLS'
        active_targets (lst): list of targets for activity is desirable
        inactive_targets (lst): list of targets for activity is undesirable
    Returns:
        mods (lst): list of clipped scores per task
        ths (lst): list of threasholds per task
    """
    
    if scheme == 'WS':
        if env_task == 'CLS':
            active = utils.ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = utils.ClippedScore(lower_x=0.5, upper_x=0.8)
        else:
            active = utils.ClippedScore(lower_x=3, upper_x=10)
            inactive = utils.ClippedScore(lower_x=10, upper_x=3)
        qed = utils.ClippedScore(lower_x=0, upper_x=1)
        ths = [0.5] * (len(active_targets)+len(inactive_targets)) + [0.0]
    else:
        if env_task == 'CLS':
            active = utils.ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = utils.ClippedScore(lower_x=0.5, upper_x=0.8)
        else:
            active = utils.ClippedScore(lower_x=3, upper_x=6.5)
            inactive = utils.ClippedScore(lower_x=10, upper_x=6.5)
        qed = utils.ClippedScore(lower_x=0, upper_x=0.5)
        ths = [0.99] * (len(active_targets)+len(inactive_targets)) + [0.0]
        
    mods = []
    for k in keys:
        if k in active_targets: 
            mods.append(active)
        elif k in inactive_targets: 
            mods.append(inactive)
        elif k == 'QED' : 
            mods.append(qed)
    
    return mods, ths

def SetAlgorithm(voc, alg, method='gpt'):
    
    """
    Initializes the generator algorithm
    
    Arguments:
        voc (): vocabulary
        alg (str): molecule format type: 'smiles' or 'graph'
        method (str, opt): type of SMILES-based algorith: 'ved', 'attn' or 'gpt' (default)
    Return:
        agent (object): molecule generator 
    """
    
    if alg == 'smiles':
        if method == 'ved':
            agent = generator.EncDec(voc, voc).to(utils.dev)
        elif method == 'attn':
            agent = generator.Seq2Seq(voc, voc).to(utils.dev)
        elif method == 'gpt':
            agent = GPT2Model(voc, n_layer=12).to(utils.dev)
    elif alg == 'graph':
        agent = GraphModel(voc).to(utils.dev)
    
    return agent

def PreTrain(args):
    
    """
    Wrapper to pretain a generator
    
    Arguments:
        args (NameSpace): namespace containing command line arguments
    """
    
    print('Pretraining {}-based model with {} data...'.format(args.algorithm, args.input))
    if args.algorithm == 'smiles':
        voc, train_loader, valid_loader = utils.DataPreparationSmiles(args)
    elif args.algorithm == 'graph':
        voc, train_loader, valid_loader = utils.DataPreparationGraph(args)
        
    agent = SetAlgorithm(voc, args.algorithm, method=args.method)
    out = '%s/generators/%s_%s_%d' % (args.base_dir, args.input, args.algorithm, args.batch_size)
    agent.fit(train_loader, valid_loader, epochs=args.epochs, out=out)
    
    
def RLTrain(args):
    
    """
    Wrapper to fine tune a generator
    
    Arguments:
        args (NameSpace): namespace containing command line arguments
    """
    
    pt_path = args.base_dir + '/generators/' + args.pretrained_model + '.pkg'
    
    print('Finetuning {}-based pretrained {} model with {} data...'.format(args.algorithm, args.pretrained_model, args.input))
    # Load input data
    if args.algorithm == 'smiles':
        voc, train_loader, valid_loader = utils.DataPreparationSmiles(args)    
    elif args.algorithm == 'graph':
        voc, train_loader, valid_loader = utils.DataPreparationGraph(args)
    
    # Initialize agent and prior by loading pretrained model
    agent = SetAlgorithm(voc, args.algorithm, method=args.method)        
    agent.load_state_dict(torch.load( pt_path, map_location=utils.dev))
    prior = SetAlgorithm(voc, args.algorithm, method=args.method)        
    prior.load_state_dict(torch.load( pt_path, map_location=utils.dev))  

    # Initialize evolver algorithm
    evolver = InitializeEvolver(agent, prior, args)
    
    # Chose the desirability function
    # 1. Load predictors per task : targets + QED
    # 2. Set clipped scores and threasholds per task depending wanted quality
    objs, keys = CreateDesirabilityFunction(args)
    mods, ths = SetModes(evolver.scheme, keys, args.env_task, args.active_targets, args.inactive_targets)
    
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
    
    """
    Wrapper to train a setup and train a generator
    
    Arguments:
        args (NameSpace): namespace containing command line arguments
    """
    
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
    args.epochs = 2
    TrainGenerator(args)