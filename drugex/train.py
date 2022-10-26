#!/usr/bin/env python
import os
import sys
import json
import argparse
import warnings

from drugex.data.corpus.vocabulary import VocGraph, VocSmiles
from drugex.data.datasets import SmilesDataSet, SmilesFragDataSet, GraphFragDataSet
from drugex.data.utils import getDataPaths, getVocPaths
from drugex.logs.utils import commit_hash, enable_file_logger, backUpFiles

from drugex.training.environment import DrugExEnvironment
from drugex.training.models import GPT2Model, GraphModel, single_network
from drugex.training.models.explorer import SmilesExplorer, GraphExplorer, SmilesExplorerNoFrag

from drugex.training.monitors import FileMonitor
from drugex.training.rewards import ParetoSimilarity, ParetoCrowdingDistance, WeightedSum
from drugex.training.scorers.modifiers import ClippedScore, SmoothHump
from drugex.training.scorers.predictors import Predictor
from drugex.training.scorers.properties import Property, Uniqueness, LigandEfficiency, LipophilicEfficiency
from drugex.training.scorers.similarity import TverskyFingerprintSimilarity, TverskyGraphSimilarity, FraggleSimilarity

warnings.filterwarnings("ignore")
    
def GeneratorArgParser(txt=None):
    """ Define and read command line arguments """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains folders 'data' (and 'envs')")
    parser.add_argument('-d', '--debug', action='store_true')
    
    
    parser.add_argument('-i', '--input', type=str, default=None,
                        help="Full file name of input file used both as train and validation sets OR common prefix of train and validation set input files.")  
    parser.add_argument('-vfs', '--voc_files', type=str, nargs='*', default=None,
                        help="Prefix of voc files to use as vocabulary ({}_smiles/graph.txt). If None, assumes no prefix.")
    parser.add_argument('-o', '--output', type=str, default=None,
                        help="Prefix of output files. If None, set to be the first word of input. ")     
    parser.add_argument('-m', '--mode', type=str, default='RL',
                        help="Mode, of the training: 'PT' for pretraining, 'FT' for fine-tuning and 'RL' for reinforcement learning") 
    
    # Input models 
    parser.add_argument('-pt', '--pretrained_model', type=str, default=None,
                        help="Name of input model (w/o .pkg extension) used as starting point of FT.")
    parser.add_argument('-ag', '--agent_model', type=str, default=None,
                        help="Name of model (w/o .pkg extension) used for the agent in RL.")
    parser.add_argument('-pr', '--prior_model', type=str, default=None,
                        help="Name of model (w/o .pkg extension) used for the prior in RL.")
    
    # General parameters
    parser.add_argument('-mt', '--mol_type', type=str, default='graph',
                        help="Molecule encoding type: 'smiles' or 'graph'")    
    parser.add_argument('-a', '--algorithm', type=str, default='trans',
                        help="Generator algorithm: 'trans' (graph- or smiles-based transformer model) or "\
                             "'rnn' (smiles, recurrent neural network based on DrugEx v2).")
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help="Number of epochs")
    parser.add_argument('-bs', '--batch_size', type=int, default=256,
                        help="Batch size")
    parser.add_argument('-gpu', '--gpu', type=str, default='1,2,3,4',
                        help="List of GPUs") 
    
    
    # RL parameters
    parser.add_argument('-ns', '--n_samples', type=int, default=640, 
                        help="During RL, n_samples and 0.2*n_samples random input fragments are used for training and validation at each epoch. If -1, all input data is used at once each epoch.") 
                         
    parser.add_argument('-eps', '--epsilon', type=float, default=0.1,
                        help="Exploring rate")
    parser.add_argument('-bet', '--beta', type=float, default=0.0,
                        help="Reward baseline")
    parser.add_argument('-s', '--scheme', type=str, default='PR',
                        help="Reward calculation scheme: 'WS' for weighted sum, 'PR' for Pareto front or 'CD' for 'PR' with crowding distance")
    parser.add_argument('-pa', '--patience', type=int, default=50,
                        help="Number of epochs to wait before early stop if no progress on test set score")

    # Affinity models
    parser.add_argument('-et', '--env_task', type=str, default='CLS',
                        help="Environment-predictor task: 'REG' or 'CLS'")
    parser.add_argument('-ea', '--env_alg', type=str, nargs='*', default=['RF'],
                        help="Environment-predictor algorith: 'RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN', 'MT_DNN', if multiple different environments are required give environment of targets in order active, inactive, window")
    parser.add_argument('-at', '--activity_threshold', type=float, default=6.5,
                        help="Activity threshold")
    parser.add_argument('-ta', '--active_targets', type=str, nargs='*', default=[], #'P29274', 'P29275', 'P30542','P0DMS8'],
                        help="Target IDs for which activity is desirable")
    parser.add_argument('-ti', '--inactive_targets', type=str, nargs='*', default=[],
                        help="Target IDs for which activity is undesirable")
    parser.add_argument('-tw', '--window_targets', type=str, nargs='*', default=[],
                        help="Target IDs for which selectivity window is calculated")
    parser.add_argument('-le', '--ligand_efficiency', action='store_true', 
                        help="If on, use the ligand efficiency instead of the simple affinity as objective for active targets")
    parser.add_argument('-le_ths', '--le_thresholds', type=float, nargs=2, default=[0.0, 0.5],
                        help='Thresholds used calculate ligand efficiency clipped scores in the desirability function.')
    parser.add_argument('-lipe', '--lipophilic_efficiency', action='store_true',
                        help="If on, use the ligand efficiency instead of the simple affinity as objective for active targets")           
    parser.add_argument('-lipe_ths', '--lipe_thresholds', type=float, nargs=2, default=[4.0, 6.0],
                        help='Thresholds used calculate lipophilic efficiency clipped scores in the desirability function.')
    
    # Pre-implemented properties
    parser.add_argument('-qed', '--qed', action='store_true',
                        help="If on, QED is used in desirability function")
    parser.add_argument('-unq', '--uniqueness', action='store_true',
                        help="If on, molecule uniqueness is used in desirability function")
    parser.add_argument('-sas', '--sa_score', action='store_true',
                        help="If on, Synthetic Accessibility score is used in desirability function")       
    parser.add_argument('-ras', '--ra_score', action='store_true',
                        help="If on, Retrosynthesis Accessibility score is used in desirability function")
    parser.add_argument('-ras_model', '--ra_score_model', type=str, default='XBG',
                        help="RAScore model: 'XBG'")
    parser.add_argument('-mw', '--molecular_weight', action='store_true',
                        help='If on, compounds with molecular weights outside a range set by mw_thersholds are penalized in the desirability function')
    parser.add_argument('-mw_ths', '--mw_thresholds', type=int, nargs='*', default=[200, 600],
                        help='Thresholds used calculate molecular weights clipped scores in the desirability function.')
    parser.add_argument('-logP', '--logP', action='store_true',
                        help='If on, compounds with logP values outside a range set by mw_thersholds are penalized in the desirability function')
    parser.add_argument('-logP_ths', '--logP_thresholds', type=float, nargs='*', default=[-5, 5],
                        help='Thresholds used calculate logP clipped scores in the desirability function')
    parser.add_argument('-tpsa', '--tpsa', action='store_true',
                        help='If on, topology polar surface area is used in desirability function')
    parser.add_argument('-tpsa_ths', '--tpsa_thresholds', type=float, nargs=2, default=[0, 140],
                        help='Thresholds used calculate TPSA clipped scores in the desirability function')
    parser.add_argument('-sim_mol', '--similarity_mol', type=str, default=None,
                        help='SMILES string of a reference molecule to which the similarity is used as an objective. Similarity metric and threshold set by --sim_metric and --sim_th.')
    parser.add_argument('-sim_type', '--similarity_type', type=str, default='fraggle',
                        help="'fraggle' for Fraggle similarity, 'graph' for Tversky similarity between graphs or fingerprints name ('AP', 'PHCO', 'BPF', 'BTF', 'PATH', 'ECFP4', 'ECFP6', 'FCFP4', 'FCFP6') for Tversky similarity between fingeprints")
    parser.add_argument('-sim_th', '--similarity_threshold', type=float, default=0.5,
                        help="Threshold for molecular similarity to reference molecule")
    parser.add_argument('-sim_tw', '--similarity_tversky_weights', nargs=2, type=float, default=[0.7, 0.3],
                        help="Weights (alpha and beta) for Tversky similarity. If both equal to 1.0, Tanimoto similarity.")       
    

    
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")

            
    args = parser.parse_args()
    
    # Setting output file prefix from input file
    if args.output is None:
        args.output = args.input.split('_')[0]

    if args.voc_files is None:
        args.voc_files = [args.input.split('_')[0]]
    
    args.targets = args.active_targets + args.inactive_targets + args.window_targets

    args.output_long = '_'.join([args.output, args.mol_type, args.algorithm, args.mode])

    return args

def DataPreparationGraph(voc_files,
                         base_dir,
                         input_prefix,
                         batch_size=128,
                         unique_frags=False,
                         n_samples=-1,
                        ):

    """
    Reads and preprocesses the vocabulary and input data for a graph-based generator

    Arguments:
        base_dir (str)              : name of the folder containing 'data' folder with input files
        input_prefix (str)          : prefix of input files
        batch_size (int), opt       : batch size
        unique_frags (bool), opt    : if True, uses reduced training set containing only unique fragment-combinations
    Returns:
        voc                         : atom vocabulary
        train_loader                : torch DataLoader containing training data
        valid_loader                : torch DataLoader containing validation data
    """

    data_path = base_dir + '/data/'
    mol_type = 'graph'

    voc_paths = getVocPaths(data_path, voc_files, mol_type)
    train_path, test_path = getDataPaths(data_path, input_prefix, mol_type, unique_frags)

    # Load train data
    data_set_train = GraphFragDataSet(train_path)
    if voc_paths:
        data_set_train.readVocs(voc_paths, VocGraph, max_len=80, n_frags=4)

    # Load test data
    data_set_test = GraphFragDataSet(test_path)
    if voc_paths:
        data_set_test.readVocs(voc_paths, VocGraph, max_len=80, n_frags=4)
    
    voc = data_set_train.getVoc() + data_set_test.getVoc()
    train_loader = data_set_train.asDataLoader(batch_size=batch_size * 4, n_samples=n_samples)
    valid_loader = data_set_test.asDataLoader(batch_size=batch_size * 10, n_samples=n_samples, n_samples_ratio=0.2)
    return voc, train_loader, valid_loader

def DataPreparationSmiles(voc_files,
                          base_dir, 
                          input_prefix, 
                          batch_size=128, 
                          unique_frags=False, 
                          n_samples=-1,
                          ):
    
    """
    Reads and preprocesses the vocabulary and input data for a graph-based generator
    
    Arguments:
        base_dir (str)              : name of the folder containing 'data' folder with input files
        input_prefix (str)          : prefix of input files
        batch_size (int), optional  : batch size
        unique_frags (bool), opt    : if True, uses reduced training set containing only unique fragment-combinations
    Returns:
        voc                         : atom vocabulary
        train_loader                : torch DataLoader containing training data
        valid_loader                : torch DataLoader containing validation data
    """
    
    data_path = base_dir + '/data/'
    mol_type = 'smi'

    voc_paths = getVocPaths(data_path, voc_files, 'smiles')
    train_path, test_path = getDataPaths(data_path, input_prefix, mol_type, unique_frags)

    voc = None
    train_loader = None
    valid_loader = None
    if args.algorithm == 'trans':
        data_set_train = SmilesFragDataSet(train_path)
        data_set_train.readVocs(voc_paths, VocSmiles, max_len=100)
        train_loader = data_set_train.asDataLoader(batch_size=batch_size, n_samples=n_samples)

        data_set_test = SmilesFragDataSet(test_path)
        data_set_test.readVocs(voc_paths, VocSmiles, max_len=100)
        valid_loader = data_set_test.asDataLoader(batch_size=batch_size, n_samples=n_samples, n_samples_ratio=0.2)

        voc = data_set_train.getVoc() + data_set_test.getVoc()
    elif args.algorithm == 'rnn':
        data_set_train = SmilesDataSet(train_path)
        data_set_train.readVocs(voc_paths, VocSmiles, max_len=100)
        train_loader = data_set_train.asDataLoader(batch_size=batch_size, n_samples=n_samples)

        data_set_test = SmilesDataSet(test_path)
        data_set_test.readVocs(voc_paths, VocSmiles, max_len=100)
        valid_loader = data_set_test.asDataLoader(batch_size=batch_size, n_samples=n_samples, n_samples_ratio=0.2)

        voc = data_set_train.getVoc()
    else:
        raise ValueError('Unknown algorithm: {}'.format(args.algorithm))

    return voc, train_loader, valid_loader


def InitializeEvolver(agent, env, prior, mol_type, algorithm, batch_size, epsilon, beta, n_samples, gpus):
    
    """
    Sets up the evolver composed of two generators.
    
    Arguments:        
        agent (torch model)         : main generator that is optimized during training
        env (Environment)           : environment used by the reinforcer to do RL
        prior (torch model)         : mutation generator is kept frozen during the process
        mol_type (str)              : molecule type
        algorithm (str)             : name of the generator algorithm
        batch_size (int)            : batch size
        epsilon (float)             : exploration rate
        beta (float)                : reward baseline
        n_samples (int)             : number train and test (0.2*n_samples) of molecules generated at each epoch
        gpus (tuple)                : IDs of GPUs to use for training
    Returns:
        evolver (torch model)       : evolver composed of two generators
    """
    
    if mol_type == 'graph':
        # FIXME:  sigma=beta? strange...
        evolver = GraphExplorer(agent, env, mutate=prior, batch_size=batch_size, epsilon=epsilon, sigma=beta, repeat=1, n_samples=n_samples, use_gpus=gpus)
    else :
        if algorithm == 'rnn':
            evolver = SmilesExplorerNoFrag(agent, env, mutate=prior, crover=agent, batch_size=batch_size, epsilon=epsilon, sigma=beta, repeat=1, n_samples=n_samples, use_gpus=gpus)
        else:
            evolver = SmilesExplorer(agent, env, mutate=prior, batch_size=batch_size, epsilon=epsilon, sigma=beta, repeat=1, n_samples=n_samples, use_gpus=gpus)
        
    return evolver
    

def CreateDesirabilityFunction(base_dir, 
                               alg, 
                               task, 
                               scheme, 
                               active_targets=[], 
                               inactive_targets=[], 
                               window_targets=[],
                               activity_threshold=6.5, 
                               qed=False, 
                               unique=False,
                               sa_score=False,
                               ra_score=False, 
                               ra_score_model='XBG',
                               mw=False,
                               mw_ths=[200,600],
                               logP=False,
                               logP_ths=[0,5],
                               tpsa=False,
                               tpsa_ths=[0,140],
                               sim_smiles=None,
                               sim_type='ECFP6',
                               sim_th=0.6,
                               sim_tw=[1.,1.],
                               le = False,
                               le_ths=[0,1],
                               lipe = False,
                               lipe_ths=[4,6],
                               ):
    
    """
    Sets up the objectives of the desirability function.
    
    Arguments:
        base_dir (str)              : folder containing 'envs' folder with saved environment-predictor models
        alg (list)                   : environment-predictor algoritm
        task (str)                  : environment-predictor task: 'REG' or 'CLS'
        scheme (str)                : optimization scheme: 'WS' for weighted sum, 'PR' for Parento front with Tanimoto-dist. or 'CD' for PR with crowding dist.
        active_targets (lst), opt   : list of active target IDs
        inactive_targets (lst), opt : list of inactive target IDs
        window_targets (lst), opt   : list of target IDs for selectivity window
        activity_threshold (float), opt : activety threshold in case of 'CLS'
        qed (bool), opt             : if True, 'quantitative estimate of drug-likeness' included in the desirability function
        unique (bool), opt          : if Trye, molecule uniqueness in an epoch included in the desirability function
        ra_score (bool), opt        : if True, 'Retrosythesis Accessibility score' included in the desirability function
        ra_score_model (str), opt   : RAscore algorithm: 'NN' or 'XGB'
        mw (bool), opt              : if True, large molecules are penalized in the desirability function
        mw_ths (list), opt          : molecular weight thresholds to penalize large molecules
        logP (bool), opt            : if True, molecules with logP values are penalized in the desirability function
        logP_ths (list), opt        : logP thresholds to penalize large molecules
        tpsa (bool), opt            : if True, tpsa used in the desirability function
        tpsa_ths (list), opt        : tpsa thresholds
        sim_smiles (str), opt       : reference molecules used for similarity calculation
        sim_type (str), opt         : type of fingerprint or 'graph' used for similarity calculation
        sim_th (float), opt         : threshold for similarity desirability  
        sim_tw (list), opt          : tversky similarity weights
        le (bool), opt              : if True, ligand efficiency used instead of activity for active targets in the desirability function
        le_ths (list), opt          : ligand efficiency thresholds
        lipe (bool), opt            : if True, lipophilic efficiency used instead of activity for active targets in the desirability function
        lipe_ths (list), opt        : lipophilic efficiency thresholds
    
    Returns:
        DrugExEnvironment (torch model) : environment-predictor model
        
    """

    schemes = {
        "PR" : ParetoSimilarity(),
        "CD" : ParetoCrowdingDistance(),
        "WS" : WeightedSum()
    }
    objs = []
    ths = []
    targets = active_targets + inactive_targets + window_targets

    pad = 3.5
    pad_window = 1.5
    if scheme == 'WS':
        # Weighted Sum (WS) reward scheme
        if task == 'CLS':
            active = ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = ClippedScore(lower_x=0.8, upper_x=0.5)
            window = ClippedScore(lower_x=0, upper_x=1)
        else:
            active = ClippedScore(lower_x=activity_threshold - pad, upper_x=activity_threshold + pad)
            inactive = ClippedScore(lower_x=activity_threshold + pad, upper_x=activity_threshold - pad)
            window = ClippedScore(lower_x=0 - pad_window, upper_x=0 + pad_window)

    else:
        # Pareto Front (PR) or Crowding Distance (CD) reward scheme
        if task == 'CLS':
            active = ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = ClippedScore(lower_x=0.8, upper_x=0.5)
            window = ClippedScore(lower_x=0, upper_x=1)
        else:
            active = ClippedScore(lower_x=activity_threshold - pad, upper_x=activity_threshold)
            inactive = ClippedScore(lower_x=activity_threshold + pad, upper_x=activity_threshold)
            window = ClippedScore(lower_x=0 - pad_window, upper_x=0 + pad_window)
      
    
    
    for i, t in enumerate(targets):
        
        if len(alg) > 1:  algorithm = alg[i] # Different algorithms for different targets
        else: algorithm = alg[0] # Same algorithm for all targets

        if algorithm.startswith('MT_'): sys.exit('TO DO: using multitask model')

        path = base_dir + '/envs/' + '_'.join([algorithm, task, t]) + '.pkg'
        
        if t in active_targets:
            
            if le or lipe:
                if task == 'CLS': 
                    log.error('Ligand efficiency and lipophilic efficiency are only available for regression tasks')
                if le:
                    objs.append(LigandEfficiency(qsar_scorer=Predictor.fromFile(path, type='REG', name=t), modifier=ClippedScore(lower_x=le_ths[0], upper_x=le_ths[1])))
                    ths.append(0.5)
                if lipe:
                    objs.append(LipophilicEfficiency(qsar_scorer=Predictor.fromFile(path, type='REG', name=t), modifier=ClippedScore(lower_x=lipe_ths[0], upper_x=lipe_ths[1])))  
                    ths.append(0.5)            
            else:
                predictor_modifier = active 
                objs.append(Predictor.fromFile(path, type=task, name=t, modifier=predictor_modifier))
                ths.append(0.5 if scheme == 'WS' else 0.99)
        
        elif t in inactive_targets:
            predictor_modifier = inactive 
            objs.append(Predictor.fromFile(path, type=task, name=t, modifier=predictor_modifier))
            ths.append(0.5 if scheme == 'WS' else 0.99)
        
        elif t in window_targets:
            predictor_modifier = window
            objs.append(Predictor.fromFile(path, type=task, name=t, modifier=predictor_modifier))
            ths.append(0.5)


    if qed:
        objs.append(Property('QED', modifier=ClippedScore(lower_x=0, upper_x=1.0)))
        ths.append(0.5)
    if unique:
        objs.append(Uniqueness(modifier=ClippedScore(lower_x=1.0, upper_x=0.0)))
        ths.append(0.0)
    if sa_score:
        objs.append(Property('SA', modifier=ClippedScore(lower_x=10, upper_x=1.0)))
        ths.append(0.5)
    if ra_score:
        from drugex.training.scorers.ra_scorer import RetrosyntheticAccessibilityScorer
        objs.append(RetrosyntheticAccessibilityScorer(use_xgb_model=False if ra_score_model == 'NN' else True, modifier=ClippedScore(lower_x=0, upper_x=1.0)))
        ths.append(0.0)
    if mw:
        objs.append(Property('MW', modifier=SmoothHump(lower_x=mw_ths[0], upper_x=mw_ths[1], sigma=100)))
        ths.append(0.99)
    if logP:
        objs.append(Property('logP', modifier=SmoothHump(lower_x=logP_ths[0], upper_x=logP_ths[1], sigma=1)))
        ths.append(0.99)
    if tpsa:
        objs.append(Property('TPSA', modifier=SmoothHump(lower_x=tpsa_ths[0], upper_x=tpsa_ths[1], sigma=10)))
        ths.append(0.99)
    if sim_smiles:
        if sim_type == 'fraggle':
            objs.append(FraggleSimilarity(sim_smiles))
        if sim_type == 'graph':
            objs.append(TverskyGraphSimilarity(sim_smiles, sim_tw[0], sim_tw[1]))
        else:
            objs.append(TverskyFingerprintSimilarity(sim_smiles, sim_type, sim_tw[0], sim_tw[1]))
        ths.append(sim_th)

    for o in objs: print(o.getKey())

    return DrugExEnvironment(objs, ths, schemes[scheme])

def SetGeneratorAlgorithm(voc, mol_type, alg, gpus):
    
    """
    Initializes the generator algorithm
    
    Arguments:
        voc (): vocabulary
        mol_type (str) : molecule type
        alg (str): agent algorithm type
        gpus (tuple): a tuple of GPU IDs to use with the initialized model
    Return:
        agent (torch model): molecule generator 
    """
    
    agent = None
    if mol_type == 'graph':
        agent = GraphModel(voc, use_gpus=gpus)
    else:
        if alg == 'trans':
            agent = GPT2Model(voc, use_gpus=gpus)
        elif alg == 'rnn':
            # TODO: add argument for is_lstm
            agent = single_network.RNN(voc, is_lstm=True, use_gpus=gpus)
        else:
            raise ValueError('Unknown algorithm: {}'.format(alg))

    return agent

def PreTrain(args):
    """
    Wrapper to pretrain a generator.
    
    Arguments:
        args (NameSpace): namespace containing command line arguments

    """
        
    print('Loading data from {}/data/{}'.format(args.base_dir, args.input))
    if args.mol_type == 'graph':
        voc, train_loader, valid_loader = DataPreparationGraph(args.voc_files, args.base_dir, args.input, args.batch_size)
        print('Pretraining graph-based ({}) model ...'.format(args.algorithm))
    else:
        voc, train_loader, valid_loader = DataPreparationSmiles(args.voc_files, args.base_dir, args.input, args.batch_size)
        print('Pretraining SMILES-based ({}) model ...'.format(args.algorithm))

    pt_path = os.path.join(args.base_dir, 'generators', args.output_long)
    agent = SetGeneratorAlgorithm(voc, args.mol_type, args.algorithm, args.gpu)
    monitor = FileMonitor(pt_path, verbose=True)
    agent.fit(train_loader, valid_loader, epochs=args.epochs, monitor=monitor, patience=args.patience)
        
def FineTune(args):
    """
    Wrapper to finetune a generator
    
    Arguments:
        args (NameSpace): namespace containing command line arguments
    """
    
    if args.pretrained_model :
        pt_path = os.path.join(args.base_dir, 'generators', args.pretrained_model +'.pkg')
        if not os.path.exists(pt_path):
            log.warning('%s does not exist, trying %s as prefix of path name.' % (pt_path, args.pretrained_model))
            pt_path = os.path.join(args.base_dir, 'generators', '_'.join([args.pretrained_model, args.mol_type, args.algorithm, 'PT']) +'.pkg')
        assert os.path.exists(pt_path), f'{pt_path} does not exist'
    else:
        raise ValueError('Missing --pretrained_model')

    ft_path = os.path.join(args.base_dir, 'generators', args.output_long)
            
    if args.mol_type == 'graph':
        voc, train_loader, valid_loader = DataPreparationGraph(args.voc_files, args.base_dir, args.input, args.batch_size)
        print('Fine-tuning graph-based model ...')
    else:
        voc, train_loader, valid_loader = DataPreparationSmiles(args.voc_files, args.base_dir, args.input, args.batch_size)
        print('Fine-tuning SMILES-based ({}) model ...'.format(args.algorithm))
    
    agent = SetGeneratorAlgorithm(voc, args.mol_type, args.algorithm, args.gpu)
    agent.loadStatesFromFile(pt_path)
    monitor = FileMonitor(ft_path, verbose=True)
    agent.fit(train_loader, valid_loader, epochs=args.epochs, monitor=monitor, patience=args.patience)
                              
def RLTrain(args):
    
    """
    Wrapper for the Reinforcement Learning
    
    Arguments:
        args (NameSpace): namespace containing command line arguments
    """

    if not args.targets:
        raise ValueError('At least on active or inactive target should be given for RL.')
                                                         
    if args.mol_type == 'graph':
        voc, train_loader, valid_loader = DataPreparationGraph(args.voc_files, args.base_dir, args.input, args.batch_size, unique_frags=True, n_samples=args.n_samples)
    else:
        voc, train_loader, valid_loader = DataPreparationSmiles(args.voc_files, args.base_dir, args.input, args.batch_size, unique_frags=True if args.algorithm != 'rnn' else False, n_samples=args.n_samples)
    
    # Initialize agent and prior by loading pretrained model
    agent = SetGeneratorAlgorithm(voc, args.mol_type, args.algorithm, args.gpu)
    ag_path = args.base_dir + '/generators/' + args.agent_model + '.pkg'       
    agent.loadStatesFromFile(ag_path)
    prior = SetGeneratorAlgorithm(voc, args.mol_type, args.algorithm, args.gpu)
    pr_path = args.base_dir + '/generators/' + args.prior_model + '.pkg'
    prior.loadStatesFromFile(pr_path)

    rl_path = args.base_dir + '/generators/' + args.output_long
    print(args.env_alg)
    
    # Create the desirability function
    environment = CreateDesirabilityFunction(
        args.base_dir,
        args.env_alg,
        args.env_task,
        args.scheme,
        active_targets=args.active_targets,
        inactive_targets=args.inactive_targets,
        window_targets=args.window_targets,
        activity_threshold=args.activity_threshold,
        qed=args.qed,
        unique=args.uniqueness, 
        sa_score=args.sa_score,
        ra_score=args.ra_score,
        ra_score_model=args.ra_score_model,
        mw=args.molecular_weight,
        mw_ths=args.mw_thresholds,
        logP=args.logP,
        logP_ths=args.logP_thresholds,
        tpsa=args.tpsa,
        tpsa_ths=args.tpsa_thresholds,
        sim_smiles=args.similarity_mol,
        sim_type=args.similarity_type,
        sim_th=args.similarity_threshold,
        sim_tw=args.similarity_tversky_weights,
        le=args.ligand_efficiency,
        le_ths=args.le_thresholds,
        lipe=args.lipophilic_efficiency,
        lipe_ths=args.lipe_thresholds,
    )

    # Initialize evolver algorithm
    ## first difference for v2 needs to be adapted
    explorer = InitializeEvolver(agent, environment, prior, args.mol_type, args.algorithm, args.batch_size, args.epsilon, args.beta, args.n_samples, args.gpu)
    monitor = FileMonitor(rl_path, verbose=True)
    explorer.fit(train_loader, valid_loader, epochs=args.epochs, patience=args.patience, monitor=monitor)


def TrainGenerator(args):
    
    """
    Wrapper to train a setup and train a generator
    
    Arguments:
        args (NameSpace): namespace containing command line arguments
    """
    
    args.gpu = [int(x) for x in args.gpu.split(',')]
    
    if not os.path.exists(args.base_dir + '/generators'):
        os.makedirs(args.base_dir + '/generators')  
    
    if args.mode == 'PT':
        log.info("Pretraining started.")
        try:
            PreTrain(args)
        except Exception as exp:
            log.exception("Something went wrong in the pretraining.")
            raise exp
        log.info("Pretraining finished.")
    elif args.mode == 'FT':
        log.info("Finetuning started.")
        try:
            FineTune(args)
        except Exception as exp:
            log.exception("Something went wrong in the finetuning.")
            raise exp
        log.info("Finetuning finished.")
    elif args.mode == 'RL' :
        log.info("Reinforcement learning started.")
        try:
            if args.algorithm in ('ved', 'attn'):
                raise NotImplementedError(f"The algorithm you specified does not support reinforcement learning: {args.algorithm}")
            RLTrain(args)
        except Exception as exp:
            log.exception("Something went wrong in reinforcement learning.")
            raise exp
        log.info("Reinforcement learning finised.")
    else:
        raise ValueError("--mode should be either 'PT', 'FT' or 'RL', you gave {}".format(args.mode))

if __name__ == "__main__":
    args = GeneratorArgParser()

    backup_msg = backUpFiles(args.base_dir, 'generators', (args.output_long,))
    

    if not os.path.exists(f'{args.base_dir}/generators'):
        os.mkdir(f'{args.base_dir}/generators')

    logSettings = enable_file_logger(
        os.path.join(args.base_dir,'generators'),
        args.output_long + '.log',
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args)
    )
    log = logSettings.log
    log.info(backup_msg)

    # Create json log file with used commandline arguments 
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(os.path.join(args.base_dir, 'generators', args.output_long + '.json'), 'w') as f:
        json.dump(vars(args), f)

    TrainGenerator(args)