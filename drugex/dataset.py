import os
import json
import time
import argparse

import pandas as pd

from drugex.data.corpus.corpus import SequenceCorpus, ScaffoldGraphCorpus, ScaffoldSequenceCorpus
from drugex.data.processing import Standardization, CorpusEncoder, RandomTrainTestSplitter
from drugex.data.datasets import SmilesDataSet, SmilesFragDataSet, SmilesScaffoldDataSet, GraphFragDataSet, \
    GraphScaffoldDataSet
from drugex.logs.utils import enable_file_logger, commit_hash, backUpFiles
from drugex.data.fragments import FragmentPairsSplitter, SequenceFragmentEncoder, \
    GraphFragmentEncoder, FragmentCorpusEncoder
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.molecules.files.suppliers import SDFSupplier
from drugex.data.corpus.vocabulary import VocSmiles, VocGraph

def load_molecules(base_dir, input_file):
    """
    Loads raw SMILES from input file and transform to rdkit molecule
    Arguments:
        base_dir (str)            : base directory, needs to contain a folder data with input file
        input_file  (str)         : file containing SMILES, can be 'sdf.gz' or (compressed) 'tsv' or 'csv' file
    Returns:
        mols (lst)                : list of SMILES extracted from input_file
    """
    
    print('Loading molecules...')
    
    file_path = base_dir + '/data/' + input_file

    if input_file.endswith('.sdf.gz') or input_file.endswith('.sdf'):
        mols = SDFSupplier(file_path, hide_duplicates=True)
        mols = [x.smiles for x in mols.toList()]
    else:
        df = pd.read_csv(file_path, sep="\t", header=0, na_values=('nan', 'NA', 'NaN', '')).dropna(subset=[args.molecule_column])
        mols = df[args.molecule_column].tolist()
        
    return mols
    
def DatasetArgParser(txt=None):
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--input', type=str, default='LIGAND_RAW.tsv',
                        help="Input file containing raw data. tsv or sdf.gz format")   
    parser.add_argument('-o', '--output', type=str, default='ligand',
                        help="Prefix of output files")
    
    parser.add_argument('-mt', '--mol_type', type=str, default='smiles',
                        help="Type of molecular representation: 'graph' or 'smiles'")     
    parser.add_argument('-sm', '--smiles_corpus', action='store_true',
                        help="If on, molecules are not split to fragments and a smiles corpus is created (for v2)")
    parser.add_argument('-nof', '--no_fragmentation', action='store_true',
                        help="If on, molecules are directly encoded without fragmentation (for v3 scaffold-based molecule generation)")
    
    
    parser.add_argument('-fm', '--frag_method', type=str, default='brics',
                        help="Fragmentation method: 'brics' or 'recap'") 
    parser.add_argument('-nf', '--n_frags', type=int, default=4,
                        help="Number of largest leaf-fragments used per compound")
    parser.add_argument('-nc', '--n_combs', type=int, default=None,
                        help="Maximum number of leaf-fragments that are combined for each fragment-combinations. If None, default is {n_frags}")
    parser.add_argument('-np', '--n_proc', type=int, default=8,
                        help="Number of parallel processes to use for multi-core tasks. If not specified, this number is set to the number of available CPUs on the system.")
    parser.add_argument('-mc', '--molecule_column', type=str, default='SMILES',
                        help="Name of the column in CSV files that contains molecules.")
    parser.add_argument('-sv', '--save_voc', action='store_true',
                        help="If on, save voc file (should only be done for the pretraining set). Currently only works is --mol_type is 'smiles'.")   
    parser.add_argument('-sif', '--save_intermediate_files', action='store_true',
                        help="If on, intermediate files")
    parser.add_argument('-nfs', '--no_fragment_split', action='store_true',
                        help="If on, split fragment data sets to training, test and unique sets.")
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
        
    if args.n_combs is None:
        args.n_combs = args.n_frags
        
    return args

def save_encoded_data(collectors, file_base, mol_type, save_voc, output):
    for collector in collectors:
        if type(collector) == pd.DataFrame:
            collector.to_csv(os.path.join(file_base, f'{output}_{collector.name}_{mol_type if mol_type != "smiles" else "smi"}.txt'), header=True, index=False, sep='\t')
        else:
            collector.save()

    vocs = [x.getVoc() for x in collectors if hasattr(x, 'getVoc')]
    voc = sum(vocs[1:], start=vocs[0])
    if save_voc:
        voc.toFile(os.path.join(file_base, f'{output}_{mol_type}_voc.txt'))

def Dataset(args):
    """ 
    Prepare input files for DrugEx generators containing encoded molecules for three different cases:
    
    - SMILES w/o fragments: {output}_corpus.txt and [opt] {voc}_smiles.txt containing the SMILES-token-encoded molecules
                             and the token-vocabulary respectively
    - SMILES w/ fragments: {output}_{mf/sf}_{frag_method}_[train/test]_smi.txt and [opt] {voc}_smiles.txt containing
                             the SMILES-token-encoded fragment-molecule pairs for the train and test sets and 
                             the token-vocabulary respectively
    - Graph fragments: {output}_{mf/sf}_{frag_method}_[train/test]_graph.txt and [opt] {voc}_graph.txt containing the
                             encoded graph-matrices of fragement-molecule pairs for the train and test sets and
                             the token-vocabulary respectively   
    """
    
                        
    # load molecules
    tm_start = time.perf_counter()
    print('Dataset started. Loading molecules...')
    smiles = load_molecules(args.base_dir, args.input)

    print("Standardizing molecules...")
    standardizer = Standardization(n_proc=args.n_proc)
    smiles = standardizer.apply(smiles)

    file_base = os.path.join(args.base_dir, 'data')
    
    if args.smiles_corpus:
        # create sequence corpus and vocabulary (used only in v2 models)
        encoder = CorpusEncoder(
            SequenceCorpus,
            {
                'vocabulary': VocSmiles()
            },
            n_proc=args.n_proc
        )
        data_collector = SmilesDataSet(os.path.join(file_base, f'{args.output}_corpus.txt'))
        encoder.apply(smiles, collector=data_collector)

        df_data_collector = data_collector.getDataFrame()
        splitter = RandomTrainTestSplitter(0.1, 1e4)
        train, test = splitter(df_data_collector)
        train.name = 'train'
        test.name = 'test'
        mol_type = 'smiles'
        save_encoded_data([train, test, data_collector], file_base, mol_type, args.save_voc, args.output)
        
    elif args.no_fragmentation:
        # encode inputs to single fragment-molecule pair without fragmentation and splitting to subsets (only v3 models)
        if args.mol_type == 'graph':
            data_set = GraphScaffoldDataSet(os.path.join(file_base, f'{args.output}_graph.txt' ))
            encoder = CorpusEncoder(
                ScaffoldGraphCorpus,
                {
                    'vocabulary': VocGraph(),
                    'largest': max(smiles, key=len)
                },
                n_proc=args.n_proc
            )
            encoder.apply(smiles, collector=data_set)
            save_encoded_data([data_set], file_base, args.mol_type, args.save_voc, args.output)
        else:
            data_set = SmilesScaffoldDataSet(os.path.join(file_base, f'{args.output}_smi.txt' ))
            encoder = CorpusEncoder(
                ScaffoldSequenceCorpus,
                {
                    'vocabulary': VocSmiles(min_len=3),
                    'largest': max(smiles, key=len)
                },
                n_proc=args.n_proc
            )
            encoder.apply(smiles, collector=data_set)
            save_encoded_data([data_set], file_base, args.mol_type, args.save_voc, args.output)            

    else:
        # create encoded fragment-molecule pair files for train and test set (only v3 models)
        file_prefix = os.path.join(file_base, f'{args.output}')

        if args.n_combs > 1 :
            print('Breaking molecules to leaf fragments, making combinations and encoding...')
        else:
            print('Breaking molecules to leaf fragments and encoding...')

        # prepare splitter and collect intermediate files if required
        pair_collectors = dict()
        if args.save_intermediate_files:
            pair_collectors['train_collector'] = lambda x : x.to_csv(file_prefix + '_train.txt', sep='\t', index=False)
            pair_collectors['test_collector'] = lambda x : x.to_csv(file_prefix + '_test.txt', sep='\t', index=False)
            pair_collectors['unique_collector'] = lambda x : x.to_csv(file_prefix + '_unique.txt', sep='\t', index=False)
        splitter = FragmentPairsSplitter(0.1, 1e4, **pair_collectors) if not args.no_fragment_split else None
        fragmenter = Fragmenter(args.n_frags, args.n_combs, args.frag_method)

        if args.mol_type == 'graph':
            encoder = FragmentCorpusEncoder(
                fragmenter=fragmenter,
                encoder=GraphFragmentEncoder(
                    VocGraph(n_frags=args.n_frags)
                ),
                pairs_splitter=splitter,
                n_proc=args.n_proc
            )

            data_collectors = [GraphFragDataSet(file_prefix + f'_{split}_graph.txt') for split in ('test', 'train', 'unique')] if splitter else [GraphFragDataSet(file_prefix + f'_train_graph.txt') ]
            encoder.apply(smiles, encodingCollectors=data_collectors)

            save_encoded_data(data_collectors, file_base, args.mol_type, args.save_voc, args.output)
        elif args.mol_type == 'smiles':
            data_collectors = [SmilesFragDataSet(file_prefix + f'_{split}_smi.txt') for split in ('test', 'train', 'unique')] if splitter else [SmilesFragDataSet(file_prefix + f'_train_smi.txt')]
            encoder = FragmentCorpusEncoder(
                fragmenter=fragmenter,
                encoder=SequenceFragmentEncoder(
                    VocSmiles()
                ),
                pairs_splitter=splitter,
                n_proc=args.n_proc
            )
            encoder.apply(smiles, encodingCollectors=data_collectors)

            save_encoded_data(data_collectors, file_base, args.mol_type, args.save_voc, args.output)
        else:
            raise ValueError("--mol_type should either 'smiles' or 'graph', you gave '{}' ".format(args.mol_type))

    tm_finish = time.perf_counter()

    print(f"Dataset finished. Execution time: {tm_finish - tm_start:0.4f} seconds")
     

if __name__ == '__main__':

    args = DatasetArgParser()

    backup_msg = backUpFiles(args.base_dir, 'data', (args.output,))
    
    logSettings = enable_file_logger(
        os.path.join(args.base_dir, 'data'),
        'dataset.log',
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args)
    )
    log = logSettings.log
    log.info(backup_msg)

    # Create json log file with used commandline arguments 
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(os.path.join(args.base_dir, 'data', 'dataset.json'), 'w') as f:
        json.dump(vars(args), f)

    Dataset(args)