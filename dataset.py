import os
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Recap, BRICS
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm
from utils import VocSmiles, VocGraph
import utils
import re
import numpy as np
from itertools import combinations
import gzip
import getopt, sys
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
        mols (lst)                : list of rdkit-molecules
    """
    
    print('Loading molecules...')
    
    file_path = base_dir + '/data/' + input_file
    
    if input_file.endswith('sdf.gz'):
        # read molecules from file
        inf = gzip.open(file_path)
        mols = Chem.ForwardSDMolSupplier(inf)
    else:
        # read molecules from file and drop duplicate SMILES
        df = pd.read_csv(file_path, sep='\t' if '.tsv' in input_file else ',',
                         usecols= lambda x : x.upper() in ['SMILES']) 
        #df.columns.str.upper()
        df = df.SMILES.dropna().drop_duplicates()
        mols = [Chem.MolFromSmiles(s) for s in df]    
        
    return mols

def standardize_mol(mols):
    """
    Standardizes SMILES and removes fragments
    Arguments:
        mols (lst)                : list of rdkit-molecules
    Returns:
        smiles (set)              : set of SMILES
    """

    print('Standardizing molecules...')
    
    charger = rdMolStandardize.Uncharger()
    chooser = rdMolStandardize.LargestFragmentChooser()
    disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    smiles = set()
    carbon = Chem.MolFromSmarts('[#6]')
    salts = Chem.MolFromSmarts('[Na,Zn]')
    for mol in tqdm(mols):
        try:
            mol = disconnector.Disconnect(mol)
            mol = normalizer.normalize(mol)
            mol = chooser.choose(mol)
            mol = charger.uncharge(mol)
            mol = disconnector.Disconnect(mol)
            mol = normalizer.normalize(mol)
            smileR = Chem.MolToSmiles(mol, 0)
            # remove SMILES that do not contain carbon
            if len(mol.GetSubstructMatches(carbon)) == 0:
                continue
            # remove SMILES that still contain salts
            if len(mol.GetSubstructMatches(salts)) > 0:
                continue
            smiles.add(Chem.CanonSmiles(smileR))
        except:
            print('Parsing Error:', Chem.MolToSmiles(mol))

    return smiles


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
    voc = VocSmiles()
    # set of unique tokens
    words = set()
    # original SMILES
    canons = []
    # tokenized SMILES
    tokens = []
    for smile in tqdm(smiles):
        token = voc.split(smile)
        # keep SMILES within certain length
        if 10 < len(token) <= 100:
            words.update(token)
            canons.append(smile)
            tokens.append(' '.join(token))
    
    # save voc file
    if save_voc:
        print('Saving vocabulary...')
        log = open(base_dir + '/data/%s_smiles.txt' % voc_file, 'w')
        log.write('\n'.join(sorted(words)))
        log.close()

    log = pd.DataFrame()
    log['Smiles'] = canons
    log['Token'] = tokens
    log.drop_duplicates(subset='Smiles')
    log.to_csv(base_dir + '/data/' + output + '_corpus.txt', sep='\t', index=False)


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
        df (pd.DataFrame)         : dataframe containing fragment-molecule pairs
    """
    
    if n_combs > 1 :
        print('Breaking molecules to leaf fragments and making combinations...')
    else:
        print('Breaking molecules to leaf fragments...')
    
    pairs = []
    for i, smile in enumerate(tqdm(smiles)):
        # replace some tokens in SMILES
        smile = utils.clean_mol(smile)
        mol = Chem.MolFromSmiles(smile)
        # break SMILES up into leaf fragments
        if method == 'recap':
            frags = np.array(sorted(Recap.RecapDecompose(mol).GetLeaves().keys()))
        else:
            frags = BRICS.BRICSDecompose(mol)
            frags = np.array(sorted({re.sub(r'\[\d+\*\]', '*', f) for f in frags}))
        if len(frags) == 1: continue
        # replace connection tokens with [H]
        du, hy = Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]')
        subs = np.array([Chem.MolFromSmiles(f) for f in frags])
        subs = np.array([Chem.RemoveHs(Chem.ReplaceSubstructs(f, du, hy, replaceAll=True)[0]) for f in subs])
        subs = np.array([m for m in subs if m.GetNumAtoms() > 1])
        # remove fragments that contain other fragments (or are contained in other fragments?)
        match = np.array([[m.HasSubstructMatch(f) for f in subs] for m in subs])
        frags = subs[match.sum(axis=0) == 1]
        # sort the fragments and only keep n_frag largest ones
        frags = sorted(frags, key=lambda x:-x.GetNumAtoms())[:n_frags]
        frags = [Chem.MolToSmiles(Chem.RemoveHs(f)) for f in frags]

        max_comb = min(n_combs, len(frags))
        for ix in range(1, max_comb+1):
            # combine leaf fragments into larger fragments
            combs = combinations(frags, ix)
            for comb in combs:
                comb_frags = '.'.join(comb)
                #remove pair of fragment combinations if longer than original SMILES 
                if len(comb_frags) > len(smile): continue
                # check if substructure is in original molecule
                if mol.HasSubstructMatch(Chem.MolFromSmarts(comb_frags)):
                    pairs.append([comb_frags, smile])
    df = pd.DataFrame(pairs, columns=['Frags', 'Smiles'])
    if save_file:
        df.to_csv(out, sep='\t',  index=False)
    
    return df

def train_test_split(df, file_base, save_files=False):
    """
    Splits fragment-molecule pairs into a train and test set
    Arguments:
        df (pd.DataFrame)         : dataframe containing fragment-molecule pairs
        file_base (str)           : base of input and output files
        save_files (bool)         : save output files
    Returns:
        train (pd.DataFrame)      : dataframe containing train set fragment-molecule pairs
        test (pd.DataFrame)       : dataframe containing test set fragment-molecule pairs
    """
    frags = set(df.Frags)
    if len(frags) > int(1e5):
        print('WARNING: to speed up the training, the test set size is 10 0000 instead of {}!'.format(len(frags)//10))
        test_in = df.Frags.drop_duplicates().sample(int(1e4))
    else:
        test_in = df.Frags.drop_duplicates().sample(len(frags) // 10)
    test = df[df.Frags.isin(test_in)]
    train = df[~df.Frags.isin(test_in)]
    if save_files:
        test.to_csv(file_base + '_test.txt', sep='\t', index=False)
        train.to_csv(file_base + '_train.txt', sep='\t', index=False)
    
    return train, test 
    
def pair_encode(df, mol_type, file_base, n_frags=4, voc_file='voc', save_voc=False):
    
    """
    Wrapper to encode fragement-molecule pairs in either SMILES-tokens or graph-matrices
    Arguments:
        df (pd.DataFrame)         : dataframe containing fragment-molecule pairs
        mol_type (str)            : molecular representation type
        file_base (str)           : base of output file
        n_frags (int)             : maximum number of fragments used as input per molecule
        voc_file (str)            : name of output voc_file
        save_voc (bool)           : if true save voc file (should only be true for the pre-training set)
    """
    
    if mol_type == 'smiles' :
        pair_smiles_encode(df, file_base, voc_file, save_voc)   
    elif mol_type == 'graph' :
        pair_graph_encode(df, file_base, n_frags) 
    else:
        raise ValueError("--mol_type should either 'smiles' or 'graph', you gave '{}' ".format(mol_type))


def pair_graph_encode(df, file_base, n_frags):
    """
    Encodes fragments and molecules to graph-matrices.
    Arguments:
        df (pd.DataFrame)         : dataframe containing fragment-molecule pairs
        file_base (str)           : base of output file
        n_frags (int)             : maximum number of fragments used as input per molecule
    """    
    
    print('Encoding fragments and molecules to graph-matrices...')
    # initialize vocabulary
    # TO DO: VocGraph doesn't work w/o file >> TO DO: Modify VocGraph to work w/o it OR write voc_graph.txt before this in dataset.py 
    voc = VocGraph(os.path.dirname(file_base) + '/voc_graph.txt', n_frags=4)
    
    # create columns for fragments
    col = ['C%d' % d for d in range(voc.max_len*5)]
    codes = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        frags, smile = row.Frags, row.Smiles
        mol = Chem.MolFromSmiles(smile)
        total = mol.GetNumBonds()
        if total >= 75 or smile == frags:
            continue
        try:
            output = voc.encode([smile], [frags])
            f, s = voc.decode(output)

            assert smile == s[0]
            #assert f == frag[0]
            code = output[0].reshape(-1).tolist()
            codes.append(code)
        
        except:
            print(i, frags, smile)
    
    codes = pd.DataFrame(codes, columns=col)
    codes.to_csv(file_base + '_graph.txt', sep='\t', index=False)

def pair_smiles_encode(df, file_base, voc_file, save_voc):
    """
    Encodes fragments and molecules to SMILES-tokens.
    Arguments:
        df (pd.DataFrame)         : dataframe containing fragment-molecule pairs
        file_base (str)           : base of output file
        voc_file (str)            : name of output voc_file
        save_voc (bool)          : if true save voc file (should only be true for the pre-training set)
    """
    
    print('Encoding fragments and molecules to SMILES-tokens...')
    # initialize vocabulary
    voc = VocSmiles()

    col = ['Input', 'Output']
    codes = []
    words = set()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        frag, smile = row.Frags, row.Smiles
        token_mol = voc.split(smile)
        ## currently not checking if molecules contain carbons
        if 10 < len(token_mol) <= 100:
            words.update(token_mol)
            token_sub = voc.split(frag)
            words.update(token_sub)
            codes.append([' '.join(token_sub), ' '.join(token_mol)])
            
    # save voc file
    if save_voc:
        print('Saving vocabulary...')
        log = open( os.path.dirname(file_base) + '/%s_smiles.txt' % voc_file, 'w')
        log.write('\n'.join(sorted(words)))
        log.close()
    
    codes = pd.DataFrame(codes, columns=col)
    codes.to_csv(file_base + '_smi.txt', sep='\t', index=False)
    
    
def DatasetArgParser(txt=None):
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-i', '--input', type=str, default='LIGAND_RAW.tsv',
                        help="Input file containing raw data. tsv or sdf.gz format")   
    parser.add_argument('-o', '--output', type=str, default='ligand',
                        help="Prefix of output files")
    parser.add_argument('-mt', '--mol_type', type=str, default='smiles',
                        help="Type of molecular representation: 'graph' or 'smiles'")     
    parser.add_argument('-nof', '--no_frags', action='store_true',
                        help="If on, molecules are not split to fragments and a corpus is create")
    
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
    
    if args.no_git is False:
        args.git_commit = utils.commit_hash(os.path.dirname(os.path.realpath(__file__)))
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(args.base_dir + '/data_args.json', 'w') as f:
        json.dump(vars(args), f)
        
    return args    

def Dataset(args):
    """ 
    Prepare input files for DrugEx generators containing encoded molecules for three different cases:
    
    - SMILES w/o fragments: {output}_corpus.txt and [opt] {voc}_smiles.txt containing the SMILES-token-encoded molecules and the token-vocabulary respectively
    - SMILES w/ fragments: {output}_{mf/sf}_{frag_method}_[train/test]_smi.txt and [opt] {voc}_smiles.txt containing the SMILES-token-encoded fragment-molecule pairs for the train and test sets and the token-vocabulary respectively
    - Graph fragments: {output}_{mf/sf}_{frag_method}_[train/test]_graph.txt and [opt] {voc}_graph.txt containing the encoded graph-matrices of fragement-molecule pairs for the train and test sets and the token-vocabulary respectively   
    """
    
                        
    # load molecules
    mols = load_molecules(args.base_dir, args.input)              
    # standardize smiles and remove salts
    smiles = standardize_mol(mols)

    if args.no_frags:
        if args.mol_type == 'graph':
            raise ValueError("To apply --no_frags, --mol_type needs to be 'smiles'")
        # create corpus (only used in v2), vocab (only used in v2)  
        corpus(args.base_dir, smiles, args.output, args.voc_file, args.save_voc)
        
    else:
        # create encoded fragment-molecule pair files for train and test set (for v3)
        file_base = '%s/data/%s_%d:%d_%s' % (args.base_dir, args.output, args.n_frags, args.n_combs, args.frag_method)
        
        # create fragment-molecule pairs
        df_pairs = pair_frags(smiles, file_base + '.txt', args.n_frags, args.n_combs,
                              method=args.frag_method, save_file=args.save_intermediate_files)
        
        # split fragment-molecule pairs into train and test set
        df_train, df_test =  train_test_split(df_pairs, file_base, save_files=args.save_intermediate_files)
        
        # encode pairs to SMILES-tokens or graph-matrices
        pair_encode(df_train, args.mol_type, file_base + '_train', 
                    n_frags=args.n_frags, voc_file=args.voc_file, save_voc=args.save_voc)
        pair_encode(df_test, args.mol_type, file_base + '_test', 
                    n_frags=args.n_frags, voc_file=args.voc_file, save_voc=args.save_voc)

if __name__ == '__main__':

    args = DatasetArgParser()
    Dataset(args)