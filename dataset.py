import os
import logging
import logging.config
from datetime import datetime
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from drugex.corpus.corpus import SequenceCorpus
from drugex.logs import utils
from drugex.logs.utils import enable_file_logger
from drugex.datasets.fragments import FragmentPairsSplitter, FragmentPairsEncodedSupplier, SequenceFragmentEncoder, \
    GraphFragmentEncoder
from drugex.molecules.converters.default import Identity
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.molecules.converters.standardizers import DrExStandardizer
from drugex.molecules.files.suppliers import SDFSupplier
from drugex.molecules.fragments import FragmentPairsSupplier
from drugex.molecules.parallel import ParallelSupplierEvaluator, ListCollector
from drugex.molecules.suppliers import StandardizedSupplier, DataFrameSupplier
from drugex.corpus.vocabulary import VocSmiles, VocGraph
import numpy as np

rdBase.DisableLog('rdApp.info')
rdBase.DisableLog('rdApp.warning')

import argparse
import json

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
        # TODO: could be parallel as well
        mols = SDFSupplier(file_path, hide_duplicates=True)
        mols = [x.smiles for x in mols.get()]
    else:
        df = pd.read_csv(file_path, sep="\t", header=0)
        supplier = ParallelSupplierEvaluator(
            DataFrameSupplier,
            # n_proc=4,
            # chunks=df.shape[0] // 10000,
            kwargs={"mol_col" : "CANONICAL_SMILES", "converter" : Identity()}
        )
        mols = supplier.get(df)
        # mols = CSVSupplier(
        #     file_path,
        #     mol_col='CANONICAL_SMILES',
        #     sep='\t',
        #     hide_duplicates=True
        # )
        
    return mols

def corpus(base_dir, smiles, output, voc_file, save_voc):
    """
    Tokenizes SMILES and returns corpus of tokenized SMILES and vocabulary of all the unique tokens
    Arguments:
        base_dir (str)            : base directory, needs to contain a folder data with .tsv file containing dataset
        smiles  (str)             : list of standardized SMILES
        output (str)              : name of output corpus file
        voc_file (str)            : name of output voc_file
        save_voc (bool)           : if true save voc file (should only be true for the pre-training set)
    """
    
    print('Creating the corpus...')
    evaluator = ParallelSupplierEvaluator(
        SequenceCorpus,
        return_suppliers=True
    )
    words = set()
    data = []
    for result in evaluator.get(smiles):
        data.extend(result[0])
        words.update(result[1].getVoc().words)
    voc = VocSmiles(words)

    # save corpus data
    with open(base_dir + f'/data/{output}_corpus_{logSettings.runID}.txt', "w", encoding="utf-8") as outfile:
        outfile.writelines(["Smiles\tToken\n"])
        outfile.writelines(data)
    
    # save voc file
    if save_voc:
        print('Saving vocabulary...')
        voc.toFile(os.path.join(
            base_dir,
            f'data/{voc_file}_smiles_{logSettings.runid}.txt')
        )

# def graph_corpus(input, output, suffix='sdf'):
#     metals = {'Na', 'Zn', 'Li', 'K', 'Ca', 'Mg', 'Ag', 'Cs', 'Ra', 'Rb', 'Al', 'Sr', 'Ba', 'Bi'}
#     voc = VocGraph('data/voc_graph.txt')
#     inf = gzip.open(input)
#     if suffix == 'sdf':
#         mols = Chem.ForwardSDMolSupplier(inf)
#         total = 2e6
#     else:
#         mols = pd.read_table(input).drop_duplicates(subset=['Smiles']).dropna(subset=['Smiles'])
#         total = len(mols)
#         mols = mols.iterrows()
#     vals = {}
#     exps = {}
#     codes, ids = [], []
#     chooser = rdMolStandardize.LargestFragmentChooser()
#     disconnector = rdMolStandardize.MetalDisconnector()
#     normalizer = rdMolStandardize.Normalizer()
#     for i, mol in enumerate(tqdm(mols, total=total)):
#         if mol is None: continue
#         if suffix != 'sdf':
#             idx = mol[1]['Molecule ChEMBL ID']

#             mol = Chem.MolFromSmiles(mol[1].Smiles)
#         else:
#             idx = mol.GetPropsAsDict()
#             idx = idx['chembl_id']
#         try:
#             mol = disconnector.Disconnect(mol)
#             mol = normalizer.normalize(mol)
#             mol = chooser.choose(mol)
#             mol = disconnector.Disconnect(mol)
#             mol = normalizer.normalize(mol)
#         except:
#             print(idx)
#         symb = [a.GetSymbol() for a in mol.GetAtoms()]
#         # Nr. of the atoms
#         bonds = mol.GetBonds()
#         if len(bonds) < 4 or len(bonds) >= 63: continue
#         if {'C'}.isdisjoint(symb): continue
#         if not metals.isdisjoint(symb): continue

#         smile = Chem.MolToSmiles(mol)
#         try:
#             s0 = smile.replace('[O]', 'O').replace('[C]', 'C') \
#                  .replace('[N]', 'N').replace('[B]', 'B') \
#                  .replace('[2H]', '[H]').replace('[3H]', '[H]')
#             s0 = Chem.CanonSmiles(s0, 0)
#             code = voc.encode([smile])
#             s1 = voc.decode(code)[0]
#             assert s0 == s1
#             codes.append(code[0].reshape(-1).tolist())
#             ids.append(idx)
#         except Exception as ex:
#             print(ex)
#             print('Parse Error:', idx)
#     df = pd.DataFrame(codes, index=ids, columns=['C%d' % i for i in range(64*4)])
#     df.to_csv(output, sep='\t', index=True)
#     print(vals)
#     print(exps)

class FragmentCollector(ListCollector):

    def __call__(self, result):
        result_flat = []
        for x in result:
            result_flat.extend(x)
        self.result.extend(result_flat)

def pair_frags(smiles, out, n_frags, n_combs, method='recap', save_file=False):
    """
    Break molecules into leaf fragments and if is_mf combine those fragments to get larger fragments
    Arguments:
        smiles (list)             : list of SMILES
        out  (str)                : name of output file
        n_frags (int)             : how many leaf-fragments are generated per compound
        n_combs (int)             : maximum number of leaf-fragments that are combined for each fragment-combinations
        method (str)              : whether to use Recap or BRICKS for fragmenting
        save_file (bool)          : save output file
    Returns:
        pairs (list)         : list of tuples containing fragment-molecule pairs
    """
    
    if n_combs > 1 :
        print('Breaking molecules to leaf fragments and making combinations...')
    else:
        print('Breaking molecules to leaf fragments...')
    
    evaluator = ParallelSupplierEvaluator(
        FragmentPairsSupplier,
        return_unique=False,
        result_collector=FragmentCollector(),
        kwargs={
            "fragmenter" : Fragmenter(n_frags, n_combs, method)
        },
        chunks=100
    )
    pairs = evaluator.get(smiles)

    if save_file:
        df = pd.DataFrame(pairs, columns=['Frags', 'Smiles'])
        df.to_csv(out, sep='\t',  index=False)
    
    return pairs

def train_test_split(pairs, file_base, save_files=False):
    """
    Splits fragment-molecule pairs into a train and test set
    Arguments:
        pairs (list)         : list containing fragments
        file_base (str)           : base of input and output files
        save_files (bool)         : save output files
    Returns:
        train (pd.DataFrame)      : dataframe containing train set fragment-molecule pairs
        test (pd.DataFrame)       : dataframe containing test set fragment-molecule pairs
        unique (pd.DataFrame)     : dataframe containing a fragment-molecule pair per unique fragment-combination
    """

    collectors = dict()
    if save_files:
        collectors['train_collector'] = lambda x : x.to_csv(file_base + f'_train_{logSettings.runID}.txt', sep='\t', index=False)
        collectors['test_collector'] = lambda x : x.to_csv(file_base + f'_test_{logSettings.runID}.txt', sep='\t', index=False)
        collectors['unique_collector'] = lambda x : x.to_csv(file_base + f'_unique_{logSettings.runID}.txt', sep='\t', index=False)
    splitter = FragmentPairsSplitter(0.1, 1e4, **collectors)
    test, train, unique = splitter(
        pairs
    )

    return train, test, unique
    
def pair_encode(df, mol_type, file_base, n_frags=4, voc_file='voc', save_voc=False):
    
    """
    Wrapper to encode fragment-molecule pairs in either SMILES-tokens or graph-matrices
    Arguments:
        df (pd.DataFrame)         : dataframe containing fragment-molecule pairs (Fragments in column 'Frags' and SMILES of the output molecule in column 'Smiles')
        mol_type (str)            : molecular representation type
        file_base (str)           : base of output file
        n_frags (int)             : maximum number of fragments used as input per molecule
        voc_file (str)            : name of output voc_file
        save_voc (bool)           : if true save voc file (should only be true for the pre-training set)
    """
    
    if mol_type == 'smiles' :
        pair_smiles_encode(df, file_base, voc_file, save_voc)   
    elif mol_type == 'graph' :
        pair_graph_encode(df, file_base, n_frags, voc_file, save_voc)
    else:
        raise ValueError("--mol_type should either 'smiles' or 'graph', you gave '{}' ".format(mol_type))


def pair_graph_encode(df, file_base, n_frags, voc_file, save_voc):
    """
    Encodes fragments and molecules to graph-matrices.
    Arguments:
        df (pd.DataFrame)         : dataframe containing fragment-molecule pairs
        file_base (str)           : base of output file
        n_frags (int)             : maximum number of fragments used as input per molecule
        voc_file (str)            : name of output vocabulary file
        save_voc (bool)           : if true save voc file (should only be true for the pre-training set)
    """

    outfile = file_base + f'_graph_{logSettings.runID}.txt'
    print(f'Encoding fragments and molecules to graph-matrices. Output: {outfile}')
    # initialize vocabulary
    voc = VocGraph(n_frags=n_frags)

    if save_voc:
        voc.toFile(os.path.dirname(file_base) + '/%s_graph_%s.txt' % (voc_file, logSettings.runID))

    evaluator = ParallelSupplierEvaluator(
        FragmentPairsEncodedSupplier,
        kwargs={'encoder': GraphFragmentEncoder(voc)},
        return_unique=False
    )

    # create columns for fragments
    col = ['C%d' % d for d in range(voc.maxLen*5)]
    codes = []
    for mol, code in evaluator.get(df):
        if not code:
            continue
        codes.append(code)
    
    codes = pd.DataFrame(codes, columns=col)
    codes.to_csv(outfile, sep='\t', index=False)

def pair_smiles_encode(df, file_base, voc_file, save_voc):
    """
    Encodes fragments and molecules to SMILES-tokens.
    Arguments:
        df (pd.DataFrame)         : dataframe containing fragment-molecule pairs
        file_base (str)           : base of output file
        voc_file (str)            : name of output voc_file
        save_voc (bool)          : if true save voc file (should only be true for the pre-training set)
    """
    
    outpath = file_base + '_smi_%s.txt' % logSettings.runID
    print(f'Encoding fragments and molecules to SMILES-tokens. Output: {outpath}')

    evaluator = ParallelSupplierEvaluator(
        FragmentPairsEncodedSupplier,
        kwargs={'encoder': SequenceFragmentEncoder()},
        return_suppliers=True
    )

    words = set()
    codes = []
    for result, supplier in evaluator.get(df):
        codes.extend(
            [
                (
                    " ".join(x[1]),
                    " ".join(x[0])
                )
                for x in result if x[0] and x[1]
            ]
        )
        words.update(supplier.encoder.getVoc().words)
            
    # save voc file
    if save_voc:
        print('Saving vocabulary...')
        voc = VocSmiles(words)
        voc.toFile(os.path.dirname(file_base) + '/%s_smiles_%s.txt' % (voc_file, logSettings.runID))
    
    codes = pd.DataFrame(codes, columns=['Input', 'Output'])
    codes.to_csv(outpath, sep='\t', index=False)
    
    
def DatasetArgParser(txt=None):
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-k', '--keep_runid', action='store_true', help="If included, continue from last run")
    parser.add_argument('-p', '--pick_runid', type=int, default=None, help="Used to specify a specific run id")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--input', type=str, default='LIGAND_RAW.tsv',
                        help="Input file containing raw data. tsv or sdf.gz format")   
    parser.add_argument('-o', '--output', type=str, default='ligand',
                        help="Prefix of output files")
    parser.add_argument('-mt', '--mol_type', type=str, default='smiles',
                        help="Type of molecular representation: 'graph' or 'smiles'")     
    parser.add_argument('-nof', '--no_frags', action='store_true',
                        help="If on, molecules are not split to fragments and a corpus is created")
    
    parser.add_argument('-fm', '--frag_method', type=str, default='brics',
                        help="Fragmentation method: 'brics' or 'recap'") 
    parser.add_argument('-nf', '--n_frags', type=int, default=4,
                        help="Number of largest leaf-fragments used per compound")
    parser.add_argument('-nc', '--n_combs', type=int, default=None,
                        help="Maximum number of leaf-fragments that are combined for each fragment-combinations. If None, default is {n_frags}")

    parser.add_argument('-vf', '--voc_file', type=str, default='voc',
                        help="Name for voc file, used to save voc tokens") 
    parser.add_argument('-sv', '--save_voc', action='store_true',
                        help="If on, save voc file (should only be done for the pretraining set). Currently only works is --mol_type is 'smiles'.")   
    parser.add_argument('-sif', '--save_intermediate_files', action='store_true',
                        help="If on, intermediate files")
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
        
    if args.n_combs is None:
        args.n_combs = args.n_frags
        
    return args    

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
    smiles = load_molecules(args.base_dir, args.input)

    print("Standardizing molecules...")
    standardizer = ParallelSupplierEvaluator(
        StandardizedSupplier,
        kwargs={
            "standardizer": DrExStandardizer(input='SMILES')
        }
    )
    smiles = standardizer.get(np.asarray(list(smiles)))

    if args.no_frags:
        if args.mol_type == 'graph':
            raise ValueError("To apply --no_frags, --mol_type needs to be 'smiles'")
        # create corpus (only used in v2), vocab (only used in v2)  
        corpus(args.base_dir, smiles, args.output, args.voc_file, args.save_voc)
        
    else:
        # create encoded fragment-molecule pair files for train and test set (for v3)
        file_base = '%s/data/%s_%d:%d_%s' % (args.base_dir, args.output, args.n_frags, args.n_combs, args.frag_method)
        
        # create fragment-molecule pairs
        pairs = pair_frags(
            smiles,
            '%s_%s.txt' % (file_base, logSettings.runID),
            args.n_frags,
            args.n_combs,
            method=args.frag_method,
            save_file=args.save_intermediate_files
        )
        
        # split fragment-molecule pairs into train and test set
        df_train, df_test, df_unique =  train_test_split(pairs, file_base, save_files=args.save_intermediate_files)
        
        # encode pairs to SMILES-tokens or graph-matrices
        pair_encode(df_train, args.mol_type, file_base + '_train', 
                    n_frags=args.n_frags, voc_file=args.voc_file, save_voc=args.save_voc)
        pair_encode(df_test, args.mol_type, file_base + '_test', 
                    n_frags=args.n_frags, voc_file=args.voc_file, save_voc=args.save_voc)
        pair_encode(df_unique, args.mol_type, file_base + '_unique', 
                    n_frags=args.n_frags, voc_file=args.voc_file, save_voc=args.save_voc)

    print("Dataset finished.")

if __name__ == '__main__':

    args = DatasetArgParser()

    # enable logger and get logSettings
    logSettings = enable_file_logger(
        os.path.join(args.base_dir,'logs'),
        'dataset.log',
        args.keep_runid,
        args.pick_runid,
        args.debug,
        __name__,
        utils.commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args)
    )

    # Create json log file with used commandline arguments 
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open('%s/logs/%s/data_args.json' % (args.base_dir, logSettings.runID), 'w') as f:
        json.dump(vars(args), f)

    Dataset(args)