#!/usr/bin/env python
import os
import sys
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

import warnings
warnings.filterwarnings("ignore")

    
def GeneratorArgParser(txt=None):
    """ Define and read command line arguments """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains folders 'data' (and 'envs')")
    parser.add_argument('-i', '--input', type=str, default=None,
                        help="Prefix of input files. If --mode is 'PT', default is 'chembl_4:4_brics' else 'ligand_4:4_brics' ")  
    parser.add_argument('-o', '--output', type=str, default=None,
                        help="Prefix of output files. If None, set to be the first world of input. ")     
    parser.add_argument('-m', '--mode', type=str, default='RL',
                        help="Mode, of the training: 'PT' for pretraining, 'FT' for fine-tuning and 'RL' for reinforcement learning") 
    
    # Input models 
    parser.add_argument('-pt', '--pretrained_model', type=str, default=None,
                        help="Name of input model (w/o .pkg extension) used as starting point of FT.")
    parser.add_argument('-ag', '--agent_model', type=str, default=None,
                        help="Name of model (w/o .pkg extension) used for the agent in RL.")
    parser.add_argument('-pr', '--prior_model', type=str, default=None,
                        help="Name of model (w/o .pkg extension) used for the prior in RL.")
    
#     parser.add_argument('-v', '--version', type=int, default=3,
#                         help="DrugEx version")
    
    # General parameters
    parser.add_argument('-a', '--algorithm', type=str, default='graph',
                        help="Generator algorithm: 'graph' for graph-based algorithm (transformer), or 'gpt' (transformer), 'ved' (lstm-based encoder-decoder) or 'attn' (lstm-based encoder-decoder with attension mechanism) for SMILES-based algorithm ")
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help="Number of epochs")
    parser.add_argument('-bs', '--batch_size', type=int, default=256,
                        help="Batch size")
    parser.add_argument('-gpu', '--gpu', type=str, default='1,2,3,4',
                        help="List of GPUs") 
    

    
    # RL parameters
    parser.add_argument('-eps', '--epsilon', type=float, default=0.1,
                        help="Exploring rate")
    parser.add_argument('-bet', '--beta', type=float, default=0.0,
                        help="Reward baseline")
    parser.add_argument('-s', '--scheme', type=str, default='PR',
                        help="Reward calculation scheme: 'WS' for weighted sum, 'PR' for Parento front or 'CD' for 'PR' with crowding distance")

    parser.add_argument('-et', '--env_task', type=str, default='REG',
                        help="Environment-predictor task: 'REG' or 'CLS'")
    parser.add_argument('-ea', '--env_alg', type=str, default='RF',
                        help="Environment-predictor algorith: 'RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN', 'MT_DNN'")
    parser.add_argument('-at', '--activity_threshold', type=float, default=6.5,
                        help="Activity threashold")                    
    parser.add_argument('-qed', '--qed', action='store_true',
                        help="If on, QED is used in desribality function")
    parser.add_argument('-ras', '--ra_score', action='store_true',
                        help="If on, Retrosythesis Accessibility score is used in desribality function")
    parser.add_argument('-ras_model', '--ra_score_model', type=str, default='XBG',
                        help="RAScore model: 'XBG'")
    
    parser.add_argument('-ta', '--active_targets', type=str, nargs='*', default=[], #'P29274', 'P29275', 'P30542','P0DMS8'],
                        help="Target IDs for which activity is desirable")
    parser.add_argument('-ti', '--inactive_targets', type=str, nargs='*', default=[],
                        help="Target IDs for which activity is undesirable")
    
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")

    
    # Load some arguments from string --> usefull functions called eg. in a notebook
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
    
    # Default input file prefix in case of pretraining and finetuning
    if args.input is None:
        if args.mode == 'PT':
            args.input = 'chembl_mf_brics'
        else:
            args.input = 'ligand_mf_brics'
            
    # Setting output file prefix from input file
    if args.output is None:
        args.output = args.input.split('_')[0]
    
#     if args.mode == 'FT':
#         #In case of FT setting some parameters from PT parameters
#         with open(args.base_dir + '/generators/' + args.pretrained_model + '.json') as f:
#             pt_params = json.load(f)
#         args.algorithm = pt_params['algorithm']
#     elif args.mode == 'RL':
#         #In case of RL setting some parameters from FT parameters
#         with open(args.base_dir + '/generators/' + args.finetuned_model + '.json') as f:
#             pt_params = json.load(f)
#         args.algorithm = pt_params['algorithm']
    
        
    
    args.targets = args.active_targets + args.inactive_targets

    return args

def DataPreparationGraph(base_dir, input_prefix, batch_size=128):
    
    """
    Reads and preprocesses the vocabulary and input data for a graph-based generator
    
    Arguments:
        base_dir (str)              : name of the folder containing 'data' folder with input files
        input_prefix (str)          : prefix of input files
        batch_size (int), optional  : batch size
    Returns:
        voc                         : atom vocabulary
        train_loader                : torch DataLoader containing training data
        valid_loader                : torch DataLoader containing validation data
    """
    
    data_path = base_dir + '/data/'

    voc = utils.VocGraph( data_path + 'voc_graph.txt', max_len=80, n_frags=4)
    
    data = pd.read_table( data_path + '%s_train_graph.txt' % input_prefix)
    data = torch.from_numpy(data.values).long().view(len(data), voc.max_len, -1)
    train_loader = DataLoader(data, batch_size=batch_size * 4, drop_last=False, shuffle=True)

    test = pd.read_table( data_path + '%s_test_graph.txt' % input_prefix)
    test = torch.from_numpy(test.values).long().view(len(test), voc.max_len, -1)
    valid_loader = DataLoader(test, batch_size=batch_size * 10, drop_last=False, shuffle=True)
    
    return voc, train_loader, valid_loader

def DataPreparationSmiles(base_dir, input_prefix, batch_size=128):
    
    """
    Reads and preprocesses the vocabulary and input data for a graph-based generator
    
    Arguments:
        base_dir (str)              : name of the folder containing 'data' folder with input files
        input_prefix (str)          : prefix of input files
        batch_size (int), optional  : batch size
    Returns:
        voc                         : atom vocabulary
        train_loader                : torch DataLoader containing training data
        valid_loader                : torch DataLoader containing validation data
    """
    
    data_path = base_dir + '/data/'
    
    if args.algorithm == 'gpt':
        voc = utils.Voc( data_path + 'voc_smiles.txt', src_len=100, trg_len=100)
    else:
        voc = utils.VocSmiles( data_path + 'voc_smiles.txt', max_len=100)

    data = pd.read_table( data_path + '%s_train_smi.txt' % input_prefix)
    data_in = voc.encode([seq.split(' ') for seq in data.Input.values])
    data_out = voc.encode([seq.split(' ') for seq in data.Output.values])
    data_set = TensorDataset(data_in, data_out)
    train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    

    test = pd.read_table( data_path + '%s_test_smi.txt' % input_prefix)
    test = test.Input.drop_duplicates()
    #test = test.sample(args.batch_size * 10).values
    test_set = voc.encode([seq.split(' ') for seq in test])
    test_set = utils.TgtData(test_set, ix=[voc.decode(seq, is_tk=False) for seq in test_set])
    valid_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=test_set.collate_fn)
    
    return voc, train_loader, valid_loader

def InitializeEvolver(agent, prior, algorithm, batch_size, epsilon, beta, scheme):
    
    """
    Sets up the evolver composed of two generators.
    
    Arguments:        
        agent (torch model)         : main generator that is optimized during training
        prior (torch model)         : mutation generator is kept frozen during the process
        algorithm (str)             : name of the generator algorithm
        batch_size (int)            : batch size
        epsilon (float)             : exploration rate
        beta (float)                : reward baseline
        scheme (str)                : name of optimization scheme
    Returns:
        evolver (torch model)       : evolver composed of two generators
    """
    
    if args.algorithm == 'graph':
        evolver = GraphExplorer(agent, mutate=prior)
    else:
        evolver = SmilesExplorer(agent, mutate=prior)
        
    evolver.batch_size = args.batch_size 
    evolver.epsilon = args.epsilon 
    evolver.sigma = args.beta 
    evolver.scheme = args.scheme 
    evolver.repeat = 1   
    
    return evolver
    

def CreateDesirabilityFunction(base_dir, alg, task, scheme, active_targets=[], inactive_targets=[], activity_threshold=6.5, qed=False, ra_score=False, ra_score_model='XBG'):
    
    """
    Sets up the objectives of the desirability function.
    
    Arguments:
        base_dir (str)              : folder containing 'envs' folder with saved environment-predictor models
        alg (str)                   : environment-predictor algoritm
        task (str)                  : environment-predictor task: 'REG' or 'CLS'
        scheme (str)                : optimization scheme: 'WS' for weighted sum, 'PR' for Parento front with Tanimoto-dist. or 'CD' for PR with crowding dist.
        active_targets (lst), opt   : list of active target IDs
        inactive_targets (lst), opt : list of inactive target IDs
        activity_threshold (float), opt : activety threshold in case of 'CLS'
        qed (bool), opt             : if True, 'quantitative estimate of drug-likeness' included in the desirability function
        ra_score (bool), opt        : if True, 'Retrosythesis Accessibility score' included in the desirability function
        ra_score_model (str), opt   : RAscore algorithm: 'NN' or 'XGB'
    Returns:
        objs (lst)                  : list of selected predictors 
        keys (lst)                  : list of names of selected predictors
        mods (lst)                  : list of ClippedScore-functions per predictor
        ths (lst)                   : list desirability thresholds per predictor
        
    """
    
    objs, keys = [], []
    targets = active_targets + inactive_targets
    
    for t in targets:
        if alg.startswith('MT_'):
            sys.exit('TO DO: using multitask model')
        else:
            path = base_dir + '/envs/single/' + alg + '_' + task + '_' + t + '.pkg'
        objs.append(utils.Predictor(path, type=task))
        keys.append(t)
    if qed :
        objs.append(utils.Property('QED'))
        keys.append('QED')
    if ra_score:
        from models.ra_scorer import RetrosyntheticAccessibilityScorer
        objs.append(RetrosyntheticAccessibilityScorer(use_xgb_model=False if ra_score_model == 'NN' else True ))
        keys.append('RAscore')
        
    pad = 3.5
    if scheme == 'WS':
        # Weighted Sum (WS) reward scheme
        if task == 'CLS':
            active = utils.ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = utils.ClippedScore(lower_x=0.8, upper_x=0.5)
        else:
            active = utils.ClippedScore(lower_x=activity_threshold-pad, upper_x=activity_threshold+pad)
            inactive = utils.ClippedScore(lower_x=activity_threshold+pad, upper_x=activity_threshold-pad)
        ths = [0.5] * (len(targets)) 
            
    else:
        # Pareto Front (PR) or Crowding Distance (CD) reward scheme
        if task == 'CLS':
            active = utils.ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = utils.ClippedScore(lower_x=0.8, upper_x=0.5)
        else:
            active = utils.ClippedScore(lower_x=activity_threshold-pad, upper_x=activity_threshold)
            inactive = utils.ClippedScore(lower_x=activity_threshold+pad, upper_x=activity_threshold)
        ths = [0.99] * len((targets))
        
        
    mods = []
    for k in keys:
        if k in active_targets: 
            mods.append(active)
        elif k in inactive_targets: 
            mods.append(inactive)
        elif k == 'QED' : 
            mods.append(utils.ClippedScore(lower_x=0, upper_x=1.0)) 
            ths += [0.0]
        elif k == 'RAscore':
            mods.append(utils.ClippedScore(lower_x=0, upper_x=1.0))
            ths += [0.0]
    
    return objs, keys, mods, ths

def SetGeneratorAlgorithm(voc, alg):
    
    """
    Initializes the generator algorithm
    
    Arguments:
        voc (): vocabulary
        alg (str): molecule format type: 'ved', 'attn', 'gpt' or 'graph'
    Return:
        agent (torch model): molecule generator 
    """
    
    if alg == 'ved':
        agent = generator.EncDec(voc, voc).to(utils.dev)
    elif alg == 'attn':
        agent = generator.Seq2Seq(voc, voc).to(utils.dev)
    elif alg == 'gpt':
        agent = GPT2Model(voc, n_layer=12).to(utils.dev)
    elif alg == 'graph':
        agent = GraphModel(voc).to(utils.dev)
    
    return agent

def PreTrain(args):
    
    """
    Wrapper to pretrain a generator
    
    Arguments:
        args (NameSpace): namespace containing command line arguments
    """
    
    pt_path = args.base_dir + '/generators/' + args.output + '_' + args.algorithm 
        
    print('Loading data from {}/data/{}'.format(args.base_dir, args.input))
    if args.algorithm == 'graph':
        voc, train_loader, valid_loader = DataPreparationGraph(args.base_dir, args.input, args.batch_size)
        print('Pretraining graph-based model ...')
    else:
        voc, train_loader, valid_loader = DataPreparationSmiles(args.base_dir, args.input, args.batch_size)
        print('Pretraining SMILES-based ({}) model ...'.format(args.algorithm))
    
    agent = SetGeneratorAlgorithm(voc, args.algorithm)
    agent.fit(train_loader, valid_loader, epochs=args.epochs, out=pt_path)
        
def FineTune(args):
    
    """
    Wrapper to finetune a generator
    
    Arguments:
        args (NameSpace): namespace containing command line arguments
    """
    
    if args.pretrained_model :
        pt_path = args.base_dir + '/generators/' + args.pretrained_model + '.pkg'
    else:
        raise ValueError('Missing --pretrained_model argument')
    
    args.finetuned_model = args.output + '_' + args.algorithm 
    ft_path = args.base_dir + '/generators/' + args.finetuned_model
        
    print('Loading data from {}/data/{}'.format(args.base_dir, args.input))
    if args.algorithm == 'graph':
        voc, train_loader, valid_loader = DataPreparationGraph(args.base_dir, args.input, args.batch_size)
        print('Fine-tuning graph-based model ...')
    else:
        voc, train_loader, valid_loader = DataPreparationSmiles(args.base_dir, args.input, args.batch_size)
        print('Fine-tuning SMILES-based ({}) model ...'.format(args.algorithm))
    
    agent = SetGeneratorAlgorithm(voc, args.algorithm)
    agent.load_state_dict(torch.load( pt_path, map_location=utils.dev))
    agent.fit(train_loader, valid_loader, epochs=args.epochs, out=ft_path)
      
    
def RLTrain(args):
    
    """
    Wrapper for the Reinforcement Learning
    
    Arguments:
        args (NameSpace): namespace containing command line arguments
    """
    
#     if args.version == 3:
#         # In v3, the agent and prior both use the FT model
#         ft_path = args.base_dir + '/generators/' + args.finetuned_model + '.pkg'
#         pt_path = ft_path
#     else :
#         # In v2, the agent and crover use the FT model and the prior the PT model
#         ft_path = args.base_dir + '/generators/' + args.finetuned_model + '.pkg'
#         pt_path = args.base_dir + '/generators/' + args.pretrained_model + '.pkg'

    ag_path = args.base_dir + '/generators/' + args.agent_model + '.pkg'
    pr_path = args.base_dir + '/generators/' + args.prior_model + '.pkg'

    rl_path = args.base_dir + '/generators/' + '_'.join([args.output, args.algorithm, args.env_alg, args.env_task, 
                                                        args.scheme, str(args.epsilon)]) + 'e'
                                                    
    
    print('Loading data from {}/data/{}'.format(args.base_dir, args.input))
    if args.algorithm == 'graph':
        voc, train_loader, valid_loader = DataPreparationGraph(args.base_dir, args.input, args.batch_size)
    else:
        voc, train_loader, valid_loader = DataPreparationSmiles(args.base_dir, args.input, args.batch_size)
    
    # Initialize agent and prior by loading pretrained model
    agent = SetGeneratorAlgorithm(voc, args.algorithm)        
    agent.load_state_dict(torch.load( ag_path, map_location=utils.dev))
    prior = SetGeneratorAlgorithm(voc, args.algorithm)        
    prior.load_state_dict(torch.load( pr_path, map_location=utils.dev))  

    # Initialize evolver algorithm
    evolver = InitializeEvolver(agent, prior, args.algorithm, args.batch_size, args.epsilon, args.beta, args.scheme)
    
    # Create the desirability function
    objs, keys, mods, ths = CreateDesirabilityFunction(args.base_dir, args.env_alg, args.env_task, args.scheme, 
                                                       active_targets=args.active_targets, inactive_targets=args.inactive_targets,
                                                       activity_threshold=args.activity_threshold, qed=args.qed, 
                                                       ra_score=args.ra_score, ra_score_model=args.ra_score_model)
    
    # Set Evolver's environment-predictor
    evolver.env = utils.Env(objs=objs, mods=mods, keys=keys, ths=ths)
    
    #root = '%s/generators/%s_%s' % (args.base_dir, args.algorithm, time.strftime('%y%m%d_%H%M%S', time.localtime()))
    #os.mkdir(root)

    # No idea what the following lines should do as the files that should be copied does not exist !?!?!?
#     copy2(args.algorithm + '_ex.py', root)
#     copy2(args.algorithm + '.py', root)

    # import evolve as agent
    evolver.out = rl_path #root + '/%s_%s_%s_%.0e' % (args.algorithm, evolver.scheme, args.env_task, evolver.epsilon)
    evolver.fit(train_loader, test_loader=valid_loader, epochs=args.epochs)
    
    with open(rl_path + '.json', 'w') as fp:
        json.dump(vars(args), fp, indent=4)

def TrainGenerator(args):
    
    """
    Wrapper to train a setup and train a generator
    
    Arguments:
        args (NameSpace): namespace containing command line arguments
    """
   
    if args.no_git is False:
        args.git_commit = utils.commit_hash(os.path.dirname(os.path.realpath(__file__)))
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    
    utils.devices = eval(args.gpu) if ',' in args.gpu else [eval(args.gpu)]
    torch.cuda.set_device(utils.devices[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if not os.path.exists(args.base_dir + '/generators'):
        os.makedirs(args.base_dir + '/generators')  
        
    with open(args.base_dir + '/generators/{}_{}_args.json'.format(args.output, args.algorithm), 'w') as f:
        json.dump(vars(args), f)
    
    if args.mode == 'PT':
        PreTrain(args)
    elif args.mode == 'FT':
        FineTune(args)
    elif args.mode == 'RL' :
        RLTrain(args)   
    else:
        raise ValueError("--mode should be either 'PT', 'FT' or 'RL', you gave {}".format(args.mode))

if __name__ == "__main__":
    
    args = GeneratorArgParser()
    TrainGenerator(args)
