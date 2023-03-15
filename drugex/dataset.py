import os
import json
import time
import argparse

import pandas as pd

from drugex.logs.utils import enable_file_logger, commit_hash, backUpFiles

from drugex.molecules.converters.fragmenters import Fragmenter, FragmenterWithSelectedFragment
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments

from drugex.data.corpus.corpus import SequenceCorpus
from drugex.data.processing import Standardization, CorpusEncoder, RandomTrainTestSplitter
from drugex.data.datasets import SmilesDataSet, SmilesFragDataSet, GraphFragDataSet
from drugex.data.fragments import FragmentPairsSplitter, SequenceFragmentEncoder, \
    GraphFragmentEncoder, FragmentCorpusEncoder
from drugex.data.corpus.vocabulary import VocSmiles, VocGraph
    
def DatasetArgParser():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # I/O parameters
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-i', '--input', type=str, default='LIGAND_RAW.tsv',
                        help="Input file containing raw data. tsv or sdf.gz format")  
    parser.add_argument('-mc', '--molecule_column', type=str, default='SMILES',
                        help="Name of the column in CSV files that contains molecules.") 
    parser.add_argument('-vf', '--voc_file', type=str, default=None,
                        help="Name of voc file molecules should adhere to (i.e. prior_smiles_voc), if molecule contains tokens not in voc it is discarded (only works is --mol_type is 'smiles')")
    parser.add_argument('-o', '--output', type=str, default='ligand',
                        help="Prefix of output files")   
    parser.add_argument('-sif', '--save_intermediate_files', action='store_true',
                        help="If on, intermediate files are saved if --no_fragments is off: (train/test/unique) fragments-molecules pairs without encoding.")
    
    # Output data type parameters
    parser.add_argument('-mt', '--mol_type', type=str, default='smiles',
                        help="Type of molecular representation: 'graph' or 'smiles'")     
    parser.add_argument('-nof', '--no_fragments', action='store_true', 
                        help="If on, molecules are not split to fragments and a smiles corpus is created (for RNN-based models)")
    parser.add_argument('-sf', '--selected_fragment', type=str, default=None,
                        help="If specified, only fragments-molecules with the selected fragment in the input fragments are used. Only works if --no_fragments is off.")
    parser.add_argument('-sfe', '--selected_fragment_exclusive', action='store_true',
                        help="If on, only fragments-molecules with the exclusively the selected fragment in the input fragments are used. Only works if --no_fragments is off.")                        
    parser.add_argument('-s', '--scaffolds', action='store_true',
                        help="In on, input smiles are treated as fragments instead of molecules. Only works if --no_fragments is off.")   

    # Fragmentation parameters
    parser.add_argument('-fm', '--frag_method', type=str, default='brics',
                        help="Fragmentation method: 'brics' or 'recap'") 
    parser.add_argument('-nf', '--n_frags', type=int, default=4,
                        help="Number of largest fragments used per compound")
    parser.add_argument('-nc', '--n_combs', type=int, default=None,
                        help="Maximum number of fragments that are combined for each fragments-molecule pair. If None, default is {n_frags}")
    parser.add_argument('-nfs', '--no_fragment_split', action='store_true',
                        help="If off, split fragment data sets to training, test and unique sets.")   
        
    # General parameters
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-np', '--n_proc', type=int, default=8,
                        help="Number of parallel processes to use for multi-core tasks.")
    parser.add_argument('-cs', '--chunk_size', type=int, default=512,
                        help="Number of iitems to be given to each process for multi-core tasks. If not specified, this number is set to 512.")
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")
    
    args = parser.parse_args()
        
    if args.n_combs is None:
        args.n_combs = args.n_frags
        
    return args

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

class Dataset():
    def __init__(self, args):
        # Set aatributes from args
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.file_base = os.path.join(self.base_dir, 'data', self.output)

    def setVocabulary(self):
        """ 
        Set up vocabulary for sequence-based datasets.    
        Returns
        -------
        voc : VocSmiles or None
            Vocabulary object
        update_voc : bool
            If True, update vocabulary
        """
        if self.voc_file:
            voc_path = os.path.join(self.base_dir, 'data', self.voc_file)
            voc = VocSmiles.fromFile(voc_path, not self.no_fragments, min_len=self.min_len)
            log.info(f'Successfully loaded vocabulary file: {voc_path}. Note: Molecules with unknown tokens will be discarded.')
        else:
            log.warning(f'No vocabulary specified. A new vocabulary will be created and saved to {self.file_base}.')
            voc = VocSmiles(not self.no_fragments, min_len=self.min_len)
        return voc


class SequenceDataset(Dataset):

    def __init__(self, args):
        super().__init__(args)
        self.min_len = 10
        self.splitter = RandomTrainTestSplitter(0.1, 1e4)

    def __call__(self, smiles_list):

        """ Prepare data for SMILES-based RNN generator """

        # Set up vocabulary
        voc = self.setVocabulary()

        # Encode molecules
        encoder = CorpusEncoder(
            SequenceCorpus,
            {
                'vocabulary': voc,
                'update_voc': False,
                'throw': True

            },
            n_proc=self.n_proc,
            chunk_size=self.chunk_size
        )   

        data_collector = SmilesDataSet(f'{self.file_base}_corpus.txt', rewrite=True)
        encoder.apply(smiles_list, collector=data_collector)
        df_data_collector = pd.DataFrame(data_collector.getData(), columns=data_collector.getColumns())

        # Split data into training and test sets
        train, test = self.splitter(df_data_collector)

        # Save data
        for df, name in zip([train, test], ['train', 'test']):
            df.to_csv(f'{self.file_base}_{name}_smiles.txt', header=True, index=False, sep='\t')


class FragmentDataset(Dataset):
    
    def __init__(self, args):
        super().__init__(args)    

        # Set up fragmenter
        if args.scaffolds:
            self.fragmenter = dummyMolsFromFragments()
        elif args.selected_fragment:
            self.fragmenter = FragmenterWithSelectedFragment(args.selected_fragment, args.n_frags, args.n_combs, args.frag_method, max_bonds=75, exclusive=args.selected_fragment_exclusive)
        else:
            self.fragmenter = Fragmenter(args.n_frags, args.n_combs, args.frag_method, max_bonds=75)

        # Set up subset splitter
        if self.scaffolds or self.no_fragment_split:
            self.splitter = None
        else:
            pair_collectors = self.setPairCollectors()
            self.splitter = FragmentPairsSplitter(0.1, 1e4, make_unique=True, **pair_collectors) 

    def setPairCollectors(self):
        """ 
        Set up pair collectors for fragment-based datasets.    
        Returns
        -------
        pair_collectors : dict
            Dictionary containing pair collectors
        """

        pair_collectors = dict()
        if self.save_intermediate_files:
            pair_collectors['train_collector'] = lambda x : pd.DataFrame(x, columns=['Frags', 'SMILES']).to_csv(f'{self.file_base}_train.txt', sep='\t', index=False)
            pair_collectors['test_collector'] = lambda x : pd.DataFrame(x, columns=['Frags', 'SMILES']).to_csv(f'{self.file_base}_test.txt', sep='\t', index=False)
            pair_collectors['unique_collector'] = lambda x : pd.DataFrame(x, columns=['Frags', 'SMILES']).to_csv(f'{self.file_base}_unique.txt', sep='\t', index=False)
        
        return pair_collectors


class FragSequenceDataset(FragmentDataset):
    def __init__(self, args):
        super().__init__(args)        
        # Set up minimum molecule length
        self.min_len = 2 if self.scaffolds else 10

    def __call__(self, smiles_list):
            
        """ Prepare data for SMILES-based transformer generator """
        
        voc = self.setVocabulary()
        
        encoder = FragmentCorpusEncoder(
            fragmenter=self.fragmenter,
            encoder=SequenceFragmentEncoder(
                voc,
                update_voc=False,
                throw=True
            ),
            pairs_splitter=self.splitter,
            n_proc=self.n_proc,
            chunk_size=self.chunk_size
        )    

        if self.splitter:
            # Set up collectors for the different subsets
            # Vocabulary is saved only once with the training set
            data_collectors = [SmilesFragDataSet(f'{self.file_base}_train_smiles.txt', rewrite=True, voc_file=f'{self.file_base}_smiles.txt.vocab', save_voc=True)]
            data_collectors += [ SmilesFragDataSet(f'{self.file_base}_{split}_smiles.txt', rewrite=True, save_voc=False) for split in ('test', 'unique')]
        else:
            # Set up collector for the whole dataset and save vocabulary
            data_collectors = [SmilesFragDataSet(f'{self.file_base}_smiles.txt', rewrite=True, save_voc=True)]

        encoder.apply(smiles_list, encodingCollectors=data_collectors)

class FragGraphDataset(FragmentDataset):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, smiles_list):
        
        """ Prepare data for graph-based transformer generator """
                
        encoder = FragmentCorpusEncoder(
            fragmenter=self.fragmenter,
            encoder=GraphFragmentEncoder(
                VocGraph(n_frags=self.n_frags)
            ),
            pairs_splitter=self.splitter,
            n_proc=self.n_proc,
            chunk_size=self.chunk_size
        )       

        if self.splitter:
            # Set up collectors for the different subsets
            # Vocabulary is saved only once with the training set
            data_collectors = [GraphFragDataSet(f'{self.file_base}_train_graph.txt', rewrite=True, voc_file=f'{self.file_base}_graph.txt.vocab', save_voc=True)]
            data_collectors += [ GraphFragDataSet(f'{self.file_base}_{split}_graph.txt', rewrite=True, save_voc=False) for split in ('test', 'unique')]
        else:
            # Set up collector for the whole dataset and save vocabulary
            data_collectors = [GraphFragDataSet(f'{self.file_base}_graph.txt', rewrite=True, save_voc=True)]

        encoder.apply(smiles_list, encodingCollectors=data_collectors)

if __name__ == '__main__':

    # Parse commandline arguments
    args = DatasetArgParser()

    # Backup files
    backup_msg = backUpFiles(args.base_dir, 'data', (args.output,))
    
    # Set up logging
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
    with open(os.path.join(args.base_dir, 'data', 'dataset.json'), 'w') as f:
        json.dump(vars(args), f)

    # Load molecules
    tm_start = time.perf_counter()
    print('Dataset started. Loading molecules...')
    smiles = load_molecules(args.base_dir, args.input)

    # Standardize molecules
    print("Standardizing molecules...")
    standardizer = Standardization(n_proc=args.n_proc, chunk_size=args.chunk_size)
    smiles = standardizer.apply(smiles)

    # Select dataset type
    if args.no_fragments:
        # SMILES + no fragments --> SequenceRNN
        dataset = SequenceDataset(args)
    elif args.mol_type == 'smiles':
        # SMILES + fragments --> SequenceTransformer
        dataset = FragSequenceDataset(args)
    elif args.mol_type == 'graph':
        # Graphs + fragments --> GraphTransformer
        dataset = FragGraphDataset(args)
    else:
        raise ValueError(f"Unknown molecule type: {args.mol_type}")
    
    # Create dataset
    dataset(smiles)

    tm_finish = time.perf_counter()
    print(f"Dataset finished. Execution time: {tm_finish - tm_start:0.4f} seconds")