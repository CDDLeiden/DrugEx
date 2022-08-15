#!/usr/bin/env python

import os
import os.path
import sys
import json
import random
import optuna
import argparse

import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier

import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

from drugex import DEFAULT_GPUS
from drugex.logs.utils import backUpFiles, enable_file_logger, commit_hash
from drugex.environment.data import QSARDataset
from drugex.environment.models import QSARModel, QSARDNN

def EnvironmentArgParser(txt=None):
    """ 
        Define and read command line arguments
    """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-ran', '--random_state', type=int, default=1, help="Seed for the random state")
    parser.add_argument('-i', '--input', type=str, default='dataset',
                        help="tsv file name that contains SMILES, target accession & corresponding data")
    parser.add_argument('-s', '--save_model', action='store_true',
                        help="If included then then the model will be trained on all data and saved")   
    parser.add_argument('-m', '--model_types', type=str, nargs='*', default=['RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN', 'MT_DNN'],
                        help="Modeltype, defaults to run all modeltypes, choose from: 'RF', 'XGB', 'DNN', 'SVM', 'PLS' (only with REG), 'NB' (only with CLS) 'KNN' or 'MT_DNN'") 
    parser.add_argument('-r', '--regression', type=str, default=None,
                        help="If True, only regression model, if False, only classification, default both")
    parser.add_argument('-t', '--targets', type=str, nargs='*', default=None,
                        help="Target indentifiers, defaults to all in the dataset") 
    parser.add_argument('-a', '--activity_threshold', type=float, default=6.5,
                        help="Activity threshold")
    parser.add_argument('-l', '--keep_low_quality', action='store_true',
                        help="If included keeps low quality data")
    parser.add_argument('-y', '--year', type=int, default=None,
                        help="Temporal split limit")  
    parser.add_argument('-n', '--test_size', type=str, default="0.1",
                        help="Random test split fraction if float is given and absolute size if int is given, used when no temporal split given.")
    parser.add_argument('-o', '--optimization', type=str, default=None,
                        help="Hyperparameter optimization, if None no optimization, if grid gridsearch, if bayes bayesian optimization")
    parser.add_argument('-ss', '--search_space', type=str, default=None,
                        help="search_space hyperparameter optimization json file name, if None default drugex.environment.search_space.json used")                  
    parser.add_argument('-c', '--model_evaluation', action='store_true',
                        help='If on, model evaluation through cross validation and independent test set is performed.')
    parser.add_argument('-ncpu', '--ncpu', type=int, default=8,
                        help="Number of CPUs")
    parser.add_argument('-gpu', '--gpu', type=str, default='1,2,3,4',
                        help="List of GPUs") 
    parser.add_argument('-bs', '--batch_size', type=int, default=2048,
                        help="Batch size for DNN")
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help="Number of epochs for DNN")
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
    
    if args.targets is None:
        df = pd.read_table('%s/data/%s' % (args.base_dir, args.input)).dropna(subset=['SMILES'])
        args.targets = df.accession.unique().tolist()

    # If no regression argument, does both regression and classification
    if args.regression is None: 
        args.regression = [True, False]
    elif args.regression.lower() == 'true':
        args.regression = [True]
    elif args.regression.lower() == 'false':
        args.regression = [False]
    else:
        sys.exit("invalid regression arg given")

    if '.' in args.test_size:
        args.test_size = float(args.test_size) 
    else: 
        args.test_size = int(args.test_size)

    return args


def Environ(args):
    """ 
        Optimize, evaluate and train estimators
    """
    args.devices = list(eval(args.gpu))
    torch.cuda.set_device(DEFAULT_GPUS[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.environ['OMP_NUM_THREADS'] = str(args.ncpu)
    
    if not os.path.exists(args.base_dir + '/envs'):
        os.makedirs(args.base_dir + '/envs') 
    
    if args.optimization == ('grid' or 'bayes'):
        grid_params = QSARModel.load_params_grid(args.search_space, args.optimization, args.model_types)

    for reg in args.regression:
        args.learning_rate = 1e-4 if reg else 1e-5 
        for target in args.targets:
            #prepare dataset for training QSAR model
            mydataset = QSARDataset(args.base_dir, args.input, target, reg = reg, timesplit=args.year,
                                    test_size=args.test_size, th = args.activity_threshold,
                                    keep_low_quality=args.keep_low_quality)
            mydataset.split_dataset()
            
            for model_type in args.model_types:
                if model_type == 'MT_DNN': print('MT DNN is not implemented yet')
                elif model_type not in ['RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN']: 
                    log.warning(f'Model type {model_type} does not exist')
                    continue
                if model_type == 'NB' and reg:
                    log.warning("NB with regression invalid, skipped.")
                    continue
                if model_type == 'PLS' and not reg:
                    log.warning("PLS with classification invalid, skipped.")
                    continue
                
                alg_dict = {
                    'RF' : RandomForestRegressor() if reg else RandomForestClassifier(),
                    'XGB': XGBRegressor(objective='reg:squarederror') if reg else \
                        XGBClassifier(objective='binary:loggeristic', use_label_encoder=False, eval_metric='loggerloss'),
                    'SVM': SVR() if reg else SVC(probability=True),
                    'PLS': PLSRegression(),
                    'NB': GaussianNB(),
                    'KNN': KNeighborsRegressor() if reg else KNeighborsClassifier()
                }

                #Create QSAR model object
                if model_type == 'DNN':
                    qsarmodel = QSARDNN(data=mydataset, alg=alg_dict[model_type], alg_name=model_type, n_jobs=args.ncpu)
                qsarmodel = QSARModel(data=mydataset, alg=alg_dict[model_type], alg_name=model_type, n_jobs=args.ncpu)

                #if desired run parameter optimization
                if args.optimization == 'grid':
                    search_space_gs = grid_params[grid_params[:,0] == model_type,1]
                    log.info(search_space_gs)
                    qsarmodel.grid_search(search_space_gs, args.save_model)
                elif args.optimization == 'bayes':
                    search_space_bs = grid_params[grid_params[:,0] == model_type,1]
                    if reg and model_type == "RF":
                        search_space_bs.update({'criterion' : ['categorical', ['squared_error', 'poisson']]})
                    elif model_type == "RF":
                        search_space_bs.update({'criterion' : ['categorical', ['gini', 'entropy']]})
                    qsarmodel.bayes_optimization(search_space_bs, 20, args.save_model)
                
                #initialize models from saved or default parameters

                if args.optimization is None and args.save_model:
                    qsarmodel.fit_model()
                
                if args.model_evaluation:
                    qsarmodel.model_evaluation()

               
if __name__ == '__main__':
    args = EnvironmentArgParser()

    #Set random seeds
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    os.environ['TF_DETERMINISTIC_OPS'] = str(args.random_state)

    # Backup files
    tasks = [ 'REG' if reg == True else 'CLS' for reg in args.regression ]
    file_prefixes = [ f'{alg}_{task}_{target}' for alg in args.model_types for task in tasks for target in args.targets]
    backup_msg = backUpFiles(args.base_dir, 'envs', tuple(file_prefixes), cp_suffix='_params')

    if not os.path.exists(f'{args.base_dir}/envs'):
        os.mkdir(f'{args.base_dir}/envs')

    logSettings = enable_file_logger(
        os.path.join(args.base_dir, 'envs'),
        'environ.log',
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args)
    )   

    log = logSettings.log
    log.info(backup_msg)

    #Add optuna logging
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Create json log file with used commandline arguments 
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(f'{args.base_dir}/envs/environ.json', 'w') as f:
        json.dump(vars(args), f)
        
    # logSettings for DNN model
    #TODO: set this only for DNN model, instead of global
    BATCH_SIZE = args.batch_size
    N_EPOCH = args.epochs
    
    #Optimize, evaluate and train estimators according to environment arguments
    Environ(args)
