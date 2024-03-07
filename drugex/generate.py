#!/usr/bin/env python
import argparse
import json
import os

from drugex.data.corpus.vocabulary import VocGraph, VocSmiles
from drugex.data.datasets import GraphFragDataSet, SmilesFragDataSet
from drugex.data.utils import getVocPaths
from drugex.logs.utils import backUpFiles, enable_file_logger
from drugex.train import CreateEnvironment, SetUpGenerator, DataPreparation


def DesignArgParser(txt=None):
    """ Define and read command line arguments """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains folders 'data' and 'output'")
    # TODO: is the debug flag necessary?
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-g', '--generator', type=str, default='ligand_mf_brics_gpt_128',
                        help="Name of final generator model file without .pkg extension")
    parser.add_argument('-i', '--input_file', type=str, default='ligand_4:4_brics_test',
                        help="For v3, name of file containing fragments for generation without _test_graph.txt / _test_smiles.txt extension or full path")
    # TODO: Is reading voc files necessary? Is the vocabulary saved to the generator file?
    parser.add_argument('-vfs', '--voc_files', type=str, nargs='*', default=['smiles'],
                        help="Names of voc files to use as vocabulary.")

    parser.add_argument('-n', '--num', type=int, default=1,
                        help="Number of molecules to generate in total.")
    parser.add_argument('--keep_invalid', action='store_true',
                        help="If on, invalid molecules are kept in the output. Else, they are dropped.")
    parser.add_argument('--keep_duplicates', action='store_true',
                        help="If on, duplicate molecules are kept in the output. Else, they are dropped.")
    parser.add_argument('--keep_undesired', action='store_true',
                        help="If on, undesirable molecules are kept in the output. Else, they are dropped.")

    parser.add_argument('-gpu', '--use_gpus', type=str, default='1,2,3,4',
                        help="List of GPUs") 
    parser.add_argument('-bs', '--batch_size', type=int, default=1048,
                        help="Batch size")

    args = parser.parse_args()
    designer_args = vars(args)
    
    # Load parameters generator/environment from trained model    
    train_parameters = ['mol_type', 'algorithm', 'predictor', 'scheme', 'use_gru',
        'active_targets', 'inactive_targets', 'window_targets', 'activity_threshold',
        'qed', 'uniqueness', 'sa_score', 'ra_score', 'molecular_weight',
        'mw_thresholds', 'logP', 'logP_thresholds', 'tpsa', 'tpsa_thresholds',
        'similarity_mol', 'similarity_type', 'similarity_threshold',
        'similarity_tversky_weights', 'ligand_efficiency', 'le_thresholds',
        'lipophilic_efficiency', 'lipe_thresholds']
    with open(args.base_dir + '/generators/' + args.generator + '.json') as f:
        train_args = json.load(f)
    for k, v in train_args.items():
        if k in train_parameters:
            designer_args[k] = v
    args = argparse.Namespace(**designer_args)
    
    # Set target list
    args.targets = args.active_targets + args.inactive_targets + args.window_targets

    print(json.dumps(vars(args), sort_keys=False, indent=2))
    return args

def DesignerFragsDataPreparation(
    voc_files : list, 
    data_path : str, 
    input_file : str,
    mol_type : str):

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
        input_path = data_path + '_'.join([input_file, 'test', mol_type if mol_type == 'graph' else 'smiles']) + '.txt'
        assert os.path.exists(input_path), f'Input file {input_path} does not exist'
    logSettings.log.info(f'Loading input fragments from {input_path}')

    if mol_type == 'graph' :
        data_set = GraphFragDataSet(input_path)
        if voc_paths:
            # TODO: SOFTCODE number of fragments !!!!
            data_set.readVocs(voc_paths, VocGraph, max_len=80, n_frags=4)
    else:
        data_set = SmilesFragDataSet(input_path)
        if voc_paths:
            # TODO: SOFTCODE number of fragments !!!!
            data_set.readVocs(voc_paths, VocSmiles, max_len=100, encode_frags=True)

    return data_set

def Design(args):

    log = logSettings.log

    args.use_gpus = [int(x) for x in args.use_gpus.split(',')]

    data_path = args.base_dir + '/data/'
    
    if not os.path.exists(args.base_dir + '/new_molecules'):
        os.makedirs(args.base_dir + '/new_molecules')
    
    if args.algorithm != 'rnn':
        data_set = DesignerFragsDataPreparation(args.voc_files,
            data_path,
            args.input_file, 
            args.mol_type
            )
        voc = data_set.getVoc()
    else:
        voc_paths = DataPreparation(args.base_dir, args.voc_files, None, None, None, args.mol_type).getVocPaths()
        voc = VocSmiles.fromFile(voc_paths[0], False, max_len=100)
    
    # Load generator model
    gen_path = args.base_dir + '/generators/' + args.generator + '.pkg'
    assert os.path.exists(gen_path)
    setup_generator = SetUpGenerator(args)
    agent = setup_generator.setGeneratorAlgorithm(voc)
    agent = setup_generator.loadStatesFromFile(agent, gen_path)
  
    # Set up environment-predictor
    env = CreateEnvironment(
        base_dir=args.base_dir,
        predictor=args.predictor,
        scheme=args.scheme,
        active_targets=args.active_targets,
        inactive_targets=args.inactive_targets,
        window_targets=args.window_targets,
        activity_threshold=args.activity_threshold,
        qed=args.qed,
        unique=args.uniqueness,
        sa_score=args.sa_score,
        ra_score=args.ra_score,
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
        logger=log
    )
    
    out = args.base_dir + '/new_molecules/' + args.generator + '.tsv'
    
    # Generate molecules and save them
    if args.keep_invalid and not args.keep_duplicates:
        logSettings.log.warning('Ignoring dropping of duplicates because invalids are kept.')
    if args.keep_invalid and not args.keep_undesired:
        logSettings.log.warning('Ignoring dropping of undesirables because invalids are kept.')  

    gen_kwargs = dict(num_samples=args.num, batch_size=args.batch_size, n_proc=8,
        drop_invalid=not args.keep_invalid, no_multifrag_smiles=True, drop_duplicates=not args.keep_duplicates, drop_undesired=not args.keep_undesired, 
        evaluator=env, raw_scores=True)
    
    if args.algorithm != 'rnn':
        gen_kwargs['input_dataset'] = data_set
        gen_kwargs['keep_frags'] = True
    
    df_mols = agent.generate(**gen_kwargs)
    df_mols.to_csv(out, index=False, sep='\t', float_format='%.2f')


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
        vars(args)
    )

    log = logSettings.log
    log.info(backup_msg)

    Design(args)
