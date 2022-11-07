#!/usr/bin/env python
import os
import json
import math
import argparse
import pandas as pd

from drugex.data.corpus.vocabulary import VocGraph, VocGPT, VocSmiles
from drugex.data.datasets import GraphFragDataSet, SmilesFragDataSet
from drugex.data.utils import getVocPaths
from drugex.logs.utils import enable_file_logger, commit_hash, backUpFiles
from drugex.train import SetGeneratorAlgorithm, CreateDesirabilityFunction

def DesignArgParser(txt=None):
    """ Define and read command line arguments """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains folders 'data' and 'output'")
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
    parser.add_argument('-m', '--modify', action='store_true',
                        help="If on, modifiers (defined in CreateDesirabilityFunction) are applied to predictor outputs, if not returns unmodified scores")
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()

    
        
    # Load parameters generator/environment from trained model
    designer_args = vars(args)
    train_parameters = ['mol_type', 'algorithm', 'epsilon', 'beta', 'scheme', 'env_alg', 'env_task', 'no_qsprpred', 
        'active_targets', 'inactive_targets', 'window_targets', 'activity_threshold', 'qed', 'sa_score', 'ra_score', 
        'ra_score_model', 'molecular_weight', 'mw_thresholds', 'logP', 'logP_thresholds' ]
    with open(args.base_dir + '/generators/' + args.generator + '.json') as f:
        train_args = json.load(f)
    for k, v in train_args.items():
        if k in train_parameters:
            designer_args[k] = v
    args = argparse.Namespace(**designer_args)
    
    args.targets = args.active_targets + args.inactive_targets + args.window_targets

    print(json.dumps(vars(args), sort_keys=False, indent=2))
    return args

def DesignerFragsDataPreparation(
    voc_files : list, 
    data_path : str, 
    input_file : str,
    mol_type : str,
    alg : str, 
    batch_size=128, 
    n_samples=-1):

    """
    Reads and preprocesses the vocabulary and input data for a graph-based generator

    Arguments:
        voc_files (list)            : list of vocabulary file prefixes
        data_path (str)             : name of the folder input files
        input_file (str)            : name of file containing input fragments
        mol_type (str)              : molecule type
        alg (str)                   : generator algoritm
        batch_size (int), opt       : batch size
        n_samples (int), opt        : number of molecules to generate
    Returns:
        voc                         : atom vocabulary
        loader                      : torch DataLoader containing input fragements
    """

    voc_paths = getVocPaths(data_path, voc_files, mol_type)

    try:
        input_path = data_path + input_file
        assert os.path.exists(input_path)
    except:
        input_path = data_path + '_'.join(input_file, 'test', mol_type,) + '.txt'
        assert os.path.exists(input_path)
    logSettings.log.info(f'Loading input fragments from {input_path}')

    if mol_type == 'graph' :
        data_set = GraphFragDataSet(input_path)
        if voc_paths:
            data_set.readVocs(voc_paths, VocGraph, max_len=80, n_frags=4)
    else:
        if gen_alg == 'trans' :
            data_set = SmilesFragDataSet(input_path)
            if voc_paths:
                data_set.readVocs(voc_paths, VocGPT, src_len=100, trg_len=100)       
        else:
            data_set = SmilesFragDataSet(input_path)
            if voc_paths:
                data_set.readVocs(voc_paths, VocSmiles, max_len=100)
    voc = data_set.getVoc()

    loader = data_set.asDataLoader(batch_size=batch_size, n_samples=n_samples)
    return voc, loader

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
            args.mol_type,
            args.algorithm, 
            args.batch_size, 
            args.num
            )
    else:
        voc_paths = getVocPaths(data_path, args.voc_files, 'smiles')
        voc = VocSmiles.fromFile(voc_paths[0], max_len=100)
    
    # Load generator model
    gen_path = args.base_dir + '/generators/' + args.generator + '.pkg'
    assert os.path.exists(gen_path)
    agent = SetGeneratorAlgorithm(voc, args.mol_type, args.algorithm, args.gpu)
    agent.loadStatesFromFile(gen_path)
    # Set up environment-predictor
    env = CreateDesirabilityFunction(
        args.base_dir,
        args.env_alg,
        args.env_task,
        args.no_qsprpred,
        args.scheme,
        active_targets=args.active_targets,
        inactive_targets=args.inactive_targets,
        window_targets=args.window_targets,
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
    if not args.modify:
        for scorer in env.scorers:
            scorer.modifier=None
    out = args.base_dir + '/new_molecules/' + args.generator + '.tsv'
    
    # Generate molecules and save them
    if args.algorithm == 'rnn':
        df = pd.DataFrame()
        batch_size = min(args.num, args.batch_size)
        df['Smiles'], scores = agent.evaluate(batch_size, num_smiles=args.num, method=env)
        scores = pd.concat([df, scores], axis=1)
    else:
        frags, smiles, scores = agent.evaluate(loader, repeat=1, method=env)
        scores['Frags'], scores['SMILES'] = frags, smiles
    if not args.modify:
        scores = scores.drop(columns=['DESIRE'])
    scores.to_csv(out, index=False, sep='\t', float_format='%.2f')


if __name__ == "__main__":

    args = DesignArgParser()

    backup_msg = backUpFiles(args.base_dir, 'new_molecules', (args.generator,))

    if not os.path.exists(f'{args.base_dir}/new_molecules'):
        os.mkdir(f'{args.base_dir}/new_molecules')

    logSettings = enable_file_logger(
        os.path.join(args.base_dir,'new_molecules'),
        'design.log',
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args)
    )

    log = logSettings.log
    log.info(backup_msg)

    Design(args)
