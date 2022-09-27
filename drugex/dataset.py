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
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.data.corpus.vocabulary import VocSmiles, VocGraph

def load_molecules(base_dir, input_file):
    """
    Loads raw SMILES from input file and transform to rdkit molecule
    Arguments:
        base_dir (str)            : base directory, needs to contain a folder data with input file
        input_file  (str)         : file containing SMILES, can be 'sdf.gz' or (compressed) 'tsv' or 'csv' file
    Returns:
        mols (list)                : list of SMILES extracted from input_file
    """
    
    print('Loading molecules...')
    df = pd.read_csv(base_dir + '/data/' + input_file, sep="\t", header=0, na_values=('nan', 'NA', 'NaN', '')).dropna(subset=[args.molecule_column])
    return df[args.molecule_column].tolist()
    
def DatasetArgParser(txt=None):
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--input', type=str, default='LIGAND_RAW.tsv',
                        help="Input file containing raw data. tsv or sdf.gz format")   
    parser.add_argument('-o', '--output', type=str, default='ligand',
                        help="Prefix of output files")   
    parser.add_argument('-nof', '--no_fragments', action='store_true', 
                        help="If on, molecules are not split to fragments and a smiles corpus is created (for v2)")
    
    # parser.add_argument('-m', '--mode', type=str, default='fragmentation',
    #                     help="'corpus' for encoding smiles without fragmentation (for v2), 'fragmetation' for fragmenting molecules and encoding the molecule-fragment pairs (for v3) or 'scaffold' creating dummy molecules from set input scaffolds and the molecule-fragment pairs")   
    
    parser.add_argument('-s', '--scaffolds', action='store_true',
                        help="In on, input smiles are treated as fragments instead of molecules")   
    parser.add_argument('-mt', '--mol_type', type=str, default='smiles',
                        help="Type of molecular representation: 'graph' or 'smiles'")     
    
    # parser.add_argument('-sm', '--smiles_corpus', action='store_true',
    #                     help="If on, molecules are not split to fragments and a smiles corpus is created (for v2)")
    # parser.add_argument('-nof', '--no_fragmentation', action='store_true',
    #                     help="If on, molecules are directly encoded without fragmentation (for v3 scaffold-based molecule generation)")
    
    parser.add_argument('-fm', '--frag_method', type=str, default='brics',
                        help="Fragmentation method: 'brics' or 'recap'") 
    parser.add_argument('-nf', '--n_frags', type=int, default=4,
                        help="Number of largest leaf-fragments used per compound")
    parser.add_argument('-nc', '--n_combs', type=int, default=None,
                        help="Maximum number of leaf-fragments that are combined for each fragment-combinations. If None, default is {n_frags}")
    parser.add_argument('-np', '--n_proc', type=int, default=8,
                        help="Number of parallel processes to use for multi-core tasks.")
    parser.add_argument('-cs', '--chunk_size', type=int, default=512,
                        help="Number of iitems to be given to each process for multi-core tasks. If not specified, this number is set to 512.")
    parser.add_argument('-mc', '--molecule_column', type=str, default='SMILES',
                        help="Name of the column in CSV files that contains molecules.")
    parser.add_argument('-sv', '--save_voc', action='store_true',
                        help="If on, save voc file (should only be done for the pretraining set). Currently only works is --mol_type is 'smiles'.")   
    parser.add_argument('-vf', '--voc_file', type=str, default=None,
                        help="Name of voc file molecules should adhere to (i.e. prior_smiles_voc), if molecule contains tokens not in voc it is discarded (only works is --mol_type is 'smiles')")
    parser.add_argument('-sif', '--save_intermediate_files', action='store_true',
                        help="If on, intermediate files")
    parser.add_argument('-nfs', '--no_fragment_split', action='store_true',
                        help="If on, split fragment data sets to training, test and unique sets.")
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")
    
    args = parser.parse_args()
        
    if args.n_combs is None:
        args.n_combs = args.n_frags
        
    return args

def save_vocabulary(vocs, file_base, mol_type, output):
    """
    Combines and saves vocabularies obtained during encoding into one single file.

    Args:
        vocs: `Vocabulary` instances that are to be combined and saved
        file_base: the base directory of the outputs
        mol_type: molecule input type ('smiles' or 'graph')
        output: output prefix

    Returns:
        `None`
    """

    voc = sum(vocs[1:], start=vocs[0])
    voc.toFile(os.path.join(file_base, f'{output}_{mol_type}_voc.txt'))

def v2Dataset(smiles, args):

    """ Prepare data for SMILES-based RNN generator (v2) """

    file_base = os.path.join(args.base_dir, 'data')

    # load get voc path if voc file given (used to filter out molecules with tokens not occuring in voc)
    if args.voc_file:
        voc_path = args.base_dir + '/data/' + args.voc_file

    # create sequence corpus and vocabulary (used only in v2 models)
    if args.voc_file:
        encoder = CorpusEncoder(
            SequenceCorpus,
            {
                'vocabulary': VocSmiles.fromFile(voc_path),
                'update_voc': False,
                'throw': True

            },
            n_proc=args.n_proc,
            chunk_size=args.chunk_size
        )
    else:
        encoder = CorpusEncoder(
            SequenceCorpus,
            {
                'vocabulary': VocSmiles(),

            },
            n_proc=args.n_proc,
            chunk_size=args.chunk_size
        )
    data_collector = SmilesDataSet(os.path.join(file_base, f'{args.output}_corpus.txt'), rewrite=True)
    encoder.apply(smiles, collector=data_collector)

    df_data_collector = data_collector.getData()
    splitter = RandomTrainTestSplitter(0.1, 1e4)
    train, test = splitter(df_data_collector)
    for df, name in zip([train, test], ['train', 'test']):
        df.to_csv(os.path.join(file_base, f'{args.output}_{name}_smi.txt'), header=True, index=False, sep='\t')

    if args.save_voc:
        save_vocabulary([data_collector.getVoc()], file_base, args.mol_type, args.output)

def v3Dataset(smiles, args):

    """ Prepare data for SMILES- or graph-based transformer generators (v3) """

    file_base = os.path.join(args.base_dir, 'data')
    file_prefix = os.path.join(file_base, f'{args.output}')

    # load get voc path if voc file given (used to filter out molecules with tokens not occuring in voc)
    if args.voc_file:
        voc_path = args.base_dir + '/data/' + args.voc_file

    if args.scaffolds:
        fragmenter = dummyMolsFromFragments()
        splitter = None

    else:
        fragmenter = Fragmenter(args.n_frags, args.n_combs, args.frag_method, max_bonds=75)
        pair_collectors = dict()
        if args.save_intermediate_files:
            pair_collectors['train_collector'] = lambda x : pd.DataFrame(x, columns=['Frags', 'Smiles']).to_csv(file_prefix + '_train.txt', sep='\t', index=False)
            pair_collectors['test_collector'] = lambda x : pd.DataFrame(x, columns=['Frags', 'Smiles']).to_csv(file_prefix + '_test.txt', sep='\t', index=False)
            pair_collectors['unique_collector'] = lambda x : pd.DataFrame(x, columns=['Frags', 'Smiles']).to_csv(file_prefix + '_unique.txt', sep='\t', index=False)
        splitter = FragmentPairsSplitter(0.1, 1e4, make_unique=True, **pair_collectors) if not args.no_fragment_split else None


    if args.mol_type == 'graph':
        
        encoder = FragmentCorpusEncoder(
            fragmenter=fragmenter,
            encoder=GraphFragmentEncoder(
                VocGraph(n_frags=args.n_frags)
            ),
            pairs_splitter=splitter,
            n_proc=args.n_proc,
            chunk_size=args.chunk_size
        )

        data_collectors = [GraphFragDataSet(file_prefix + f'_{split}_graph.txt', rewrite=True) for split in ('test', 'train', 'unique')] if splitter else [GraphFragDataSet(file_prefix + f'_graph.txt', rewrite=True) ]
        encoder.apply(smiles, encodingCollectors=data_collectors)
        if args.save_voc:
            save_vocabulary([x.getVoc() for x in data_collectors], file_base, args.mol_type, args.output)
    
    elif args.mol_type == 'smiles':

        if args.voc_file:
            encoder = FragmentCorpusEncoder(
                fragmenter=fragmenter,
                encoder=SequenceFragmentEncoder(
                    VocSmiles.fromFile(voc_path), 
                    update_voc = False, 
                    throw= True),
                pairs_splitter=splitter,
                n_proc=args.n_proc,
                chunk_size=args.chunk_size
            )
        else:
            encoder = FragmentCorpusEncoder(
                fragmenter=fragmenter,
                encoder=SequenceFragmentEncoder(
                    VocSmiles()
                ),
                pairs_splitter=splitter,
                n_proc=args.n_proc,
                chunk_size=args.chunk_size
            )

        data_collectors = [SmilesFragDataSet(file_prefix + f'_{split}_smi.txt', rewrite=True) for split in ('test', 'train', 'unique')] if splitter else [SmilesFragDataSet(file_prefix + f'_smi.txt')]
        encoder.apply(smiles, encodingCollectors=data_collectors)
        if args.save_voc:
            save_vocabulary([x.getVoc() for x in data_collectors], file_base, args.mol_type, args.output)
    else:
        raise ValueError("--mol_type should either 'smiles' or 'graph', you gave '{}' ".format(args.mol_type))

def Dataset(args):
                        
    # load molecules
    tm_start = time.perf_counter()
    print('Dataset started. Loading molecules...')
    smiles = load_molecules(args.base_dir, args.input)

    print("Standardizing molecules...")
    standardizer = Standardization(n_proc=args.n_proc, chunk_size=args.chunk_size)
    smiles = standardizer.apply(smiles)
    
    if args.no_fragments:
        v2Dataset(smiles, args)
    else:
        v3Dataset(smiles, args)

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
