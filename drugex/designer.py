#!/usr/bin/env python
import os
import json
import argparse
import pandas as pd

import math

from drugex.data.corpus.vocabulary import VocGraph, VocGPT, VocSmiles
from drugex.data.datasets import GraphFragDataSet, SmilesFragDataSet
from drugex.logs.utils import enable_file_logger, commit_hash
from train import SetGeneratorAlgorithm, CreateDesirabilityFunction

def DesignArgParser(txt=None):
    """ Define and read command line arguments """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains folders 'data' and 'output'")
    parser.add_argument('-k', '--keep_runid', action='store_true', help="If included, continue from last run")
    parser.add_argument('-p', '--pick_runid', type=int, default=None, help="Used to specify a specific run id")
    parser.add_argument('-d', '--debug', action='store_true')

    parser.add_argument('-g', '--generator', type=str, default='ligand_mf_brics_gpt_128',
                        help="Name of final generator model file without .pkg extension")
    parser.add_argument('-i', '--input_file', type=str, default='ligand_4:4_brics_test',
                        help="For v3, name of file containing fragments for generation without _graph.txt / _smi.txt extension") 
    parser.add_argument('-vfs', '--voc_files', type=str, nargs='*', default=['smiles'],
                        help="Names of voc files to use as vocabulary.")
    parser.add_argument('-n', '--num', type=int, default=1,
                        help="For v2 number of molecules to generate in total, for v3 number of molecules to generate per fragment")
    parser.add_argument('-gpu', '--gpu', type=str, default='1,2,3,4',
                        help="List of GPUs") 
    parser.add_argument('-bs', '--batch_size', type=int, default=1048,
                        help="Batch size")
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
        
    # Load parameters generator/environment from trained model
    with open(args.base_dir + '/generators/' + args.generator + '.json') as f:
        g_params = json.load(f)
    
    args.algorithm = g_params['algorithm']
    args.epsilon = g_params['epsilon']
    args.beta = g_params['beta']
    args.scheme = g_params['scheme']
    args.env_alg = g_params['env_alg']
    args.env_task = g_params['env_task']
    args.active_targets = g_params['active_targets']
    args.inactive_targets = g_params['inactive_targets']
    args.activity_threshold = g_params['activity_threshold']
    args.qed = g_params['qed']
    args.sa_score = g_params['sa_score']
    args.ra_score = g_params['ra_score']
    args.ra_score_model = g_params['ra_score_model']
    args.env_runid = g_params['env_runid']
    args.data_runid = g_params['data_runid']
    args.molecular_weight = g_params['molecular_weight']
    args.mw_thresholds = g_params['mw_thresholds']
    args.logP = g_params['logP']
    args.logP_thresholds = g_params['logP_thresholds']
    
    args.targets = args.active_targets + args.inactive_targets

    print(json.dumps(vars(args), sort_keys=False, indent=2))
    return args

def DesignerFragsDataPreparation(
    voc_files : list, 
    data_path : str, 
    input_file : str,
    gen_alg : str, 
    batch_size=128, 
    n_samples=-1):

    """
    Reads and preprocesses the vocabulary and input data for a graph-based generator

    Arguments:
        voc_files (list)            : list of vocabulary file prefixes
        data_path (str)             : name of the folder input files
        input_file (str)            : name of file containing input fragments
        gen_alg (str)               : generator algoritm
        batch_size (int), opt       : batch size
        n_samples (int), opt        : number of molecules to generate
    Returns:
        voc                         : atom vocabulary
        loader                      : torch DataLoader containing input fragements
    """

    mol_type = 'graph' if gen_alg == 'graph' else 'smiles'

    voc_paths = vocPaths(data_path, voc_files, mol_type)
    logSettings.log.info(f'Loading vocabulary from {voc_paths}')

    input_path = data_path + input_file
    assert os.path.exists(input_path)
    logSettings.log.info(f'Loading input fragments from {input_path}')

    if gen_alg == 'graph' :
        data_set = GraphFragDataSet(input_path)
        data_set.readVocs(voc_paths, VocGraph, max_len=80, n_frags=4)
    elif gen_alg == 'gpt' :
        data_set = SmilesFragDataSet(input_path)
        data_set.readVocs(voc_paths, VocGPT, src_len=100, trg_len=100)       
    else:
        data_set = SmilesFragDataSet(input_path)
        data_set.readVocs(voc_paths, VocSmiles, max_len=100)
    voc = data_set.getVoc()

    loader = data_set.asDataLoader(batch_size=batch_size, n_samples=n_samples)
    return voc, loader

def vocPaths(data_path : str, voc_files : list, mol_type : str):

    voc_paths = []
    for v in voc_files:
        path = data_path + f"{v}_{mol_type}_{logSettings.runID}.txt"
        if not os.path.exists(path):
            logSettings.log.warning(f'Reading voc_{mol_type}.txt instead of {path}')
            path = data_path + "voc_{mol_type}.txt"
        assert os.path.exists(path)
        voc_paths.append(path)

    return voc_paths

def Design(args):

    log = logSettings.log

    args.gpu = [int(x) for x in args.gpu.split(',')]

    data_path = args.base_dir + '/data/'
    
    if not os.path.exists(args.base_dir + '/new_molecules'):
        os.makedirs(args.base_dir + '/new_molecules')
    
    if args.algorithm != 'rnn':
        voc, loader = DesignerFragsDataPreparation(args.voc_files, 
            data_path,
            args.input_file, 
            args.algorithm, 
            args.batch_size, 
            args.num
            )
    else:
        voc_paths = vocPaths(data_path, args.voc_files, 'smiles')
        voc = VocSmiles(voc_paths, max_len=100)
    
    # Load generator model
    gen_path = args.base_dir + '/generators/' + args.generator + '.pkg'
    assert os.path.exists(gen_path)
    agent = SetGeneratorAlgorithm(voc, args.algorithm)
    agent.loadStatesFromFile(gen_path)
    # Set up environment-predictor
    env = CreateDesirabilityFunction(
        args.base_dir,
        args.env_alg,
        args.env_task,
        args.scheme,
        args.env_runid,
        active_targets=args.active_targets,
        inactive_targets=args.inactive_targets,
        activity_threshold=args.activity_threshold,
        qed=args.qed,
        sa_score=args.sa_score,
        ra_score=args.ra_score,
        ra_score_model=args.ra_score_model,
        mw=args.molecular_weight,
        mw_ths=args.mw_thresholds,
        logP=args.logP,
        logP_ths=args.logP_thresholds,
    )
    
    out = args.base_dir + '/new_molecules/' + args.generator + '.tsv'
    
    # Generate molecules and save them
    if args.algorithm == 'rnn':
        df = pd.DataFrame()
        repeat = 1
        batch_size = min(args.num, args.batch_size)
        if args.num > args.batch_size:
            repeat = math.ceil(args.num / batch_size)
        df['Smiles'], scores = agent.evaluate(batch_size, repeat = repeat, method=env)
        scores = pd.concat([df, scores],axis=1)
    else:
        frags, smiles, scores = agent.evaluate(loader, repeat=1, method=env)
        scores['Frags'], scores['SMILES'] = frags, smiles
    scores.to_csv(out, index=False, sep='\t', float_format='%.2f')


if __name__ == "__main__":

    args = DesignArgParser()

    logSettings = enable_file_logger(
        os.path.join(args.base_dir,'logs'),
        'design.log',
        args.keep_runid,
        args.pick_runid,
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args)
    )

    # Default input file prefix in case of pretraining and finetuning
    if args.data_runid is None:
        args.data_runid = logSettings.runID
        
    if args.env_runid is None:
        args.env_runid = logSettings.runID

    Design(args)
