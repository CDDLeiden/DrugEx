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


def standardize_mol(base_dir, input, suffix='sdf'):
    """
    Standardizes SMILES and removes fragments
    Arguments:
        base_dir (str)            : base directory, needs to contain a folder data with .tsv file containing dataset
        input  (str)              : file containing SMILES
        suffix (str)              : suffix of input file
    """
    if suffix =='sdf':
        # read molecules from file
        inf = gzip.open(base_dir + '/data/' + input)
        mols = Chem.ForwardSDMolSupplier(inf)
        # mols = [mol for mol in suppl]
    else:
        # read molecules from file and drop duplicate SMILES
        df = pd.read_table(base_dir + '/data/' + input)
        df.columns.str.upper()
        df = df.SMILES.dropna().drop_duplicates()
        mols = [Chem.MolFromSmiles(s) for s in df]

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
        log = open(base_dir + '/data/%s_smiles.txt' % voc_file, 'w')
        log.write('\n'.join(sorted(words)))
        log.close()

    log = pd.DataFrame()
    log['Smiles'] = canons
    log['Token'] = tokens
    log.drop_duplicates(subset='Smiles')
    log.to_csv(base_dir + '/data/' + output + '_corpus.txt', sep='\t', index=False)


def graph_corpus(input, output, suffix='sdf'):
    metals = {'Na', 'Zn', 'Li', 'K', 'Ca', 'Mg', 'Ag', 'Cs', 'Ra', 'Rb', 'Al', 'Sr', 'Ba', 'Bi'}
    voc = VocGraph('data/voc_graph.txt')
    inf = gzip.open(input)
    if suffix == 'sdf':
        mols = Chem.ForwardSDMolSupplier(inf)
        total = 2e6
    else:
        mols = pd.read_table(input).drop_duplicates(subset=['Smiles']).dropna(subset=['Smiles'])
        total = len(mols)
        mols = mols.iterrows()
    vals = {}
    exps = {}
    codes, ids = [], []
    chooser = rdMolStandardize.LargestFragmentChooser()
    disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    for i, mol in enumerate(tqdm(mols, total=total)):
        if mol is None: continue
        if suffix != 'sdf':
            idx = mol[1]['Molecule ChEMBL ID']

            mol = Chem.MolFromSmiles(mol[1].Smiles)
        else:
            idx = mol.GetPropsAsDict()
            idx = idx['chembl_id']
        try:
            mol = disconnector.Disconnect(mol)
            mol = normalizer.normalize(mol)
            mol = chooser.choose(mol)
            mol = disconnector.Disconnect(mol)
            mol = normalizer.normalize(mol)
        except:
            print(idx)
        symb = [a.GetSymbol() for a in mol.GetAtoms()]
        # Nr. of the atoms
        bonds = mol.GetBonds()
        if len(bonds) < 4 or len(bonds) >= 63: continue
        if {'C'}.isdisjoint(symb): continue
        if not metals.isdisjoint(symb): continue

        smile = Chem.MolToSmiles(mol)
        try:
            s0 = smile.replace('[O]', 'O').replace('[C]', 'C') \
                 .replace('[N]', 'N').replace('[B]', 'B') \
                 .replace('[2H]', '[H]').replace('[3H]', '[H]')
            s0 = Chem.CanonSmiles(s0, 0)
            code = voc.encode([smile])
            s1 = voc.decode(code)[0]
            assert s0 == s1
            codes.append(code[0].reshape(-1).tolist())
            ids.append(idx)
        except Exception as ex:
            print(ex)
            print('Parse Error:', idx)
    df = pd.DataFrame(codes, index=ids, columns=['C%d' % i for i in range(64*4)])
    df.to_csv(output, sep='\t', index=True)
    print(vals)
    print(exps)


def pair_frags(smiles, out, n_frags, method='Recap', is_mf=True):
    """
    Break molecules into leaf fragments and if is_mf combine those fragments to get larger fragments
    Arguments:
        smiles (list)             : list of SMILES
        out  (str)                : name of output file
        n_frags (int)             : how many fragments to save
        method (str)              : whether to use Recap or BRICKS for fragmenting
        is_mf (bool)              : wheter to combine leaf fragments
    """
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

        max_comb = len(frags) if is_mf else 1
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
    df.to_csv(out, sep='\t',  index=False)


def pair_graph_encode(fname, voc, out):
    # read fragments (input) and original SMILES (output)
    df = pd.read_table(fname)
    # create columns for fragments
    col = ['C%d' % d for d in range(voc.max_len*5)]
    codes = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        frags, smile = row.Frags, row.Smiles
        # smile = voc_smi.decode(row.Output.split(' '))
        # frag = voc_smi.decode(row.Input.split(' '))
        mol = Chem.MolFromSmiles(smile)
        total = mol.GetNumBonds()
        if total >= 75 or smile == frags:
            continue
        try:
            # s = utils.clean_mol(smile)
            # f = utils.clean_mol(frag, is_deep=False)
            output = voc.encode([smile], [frags])
            f, s = voc.decode(output)

            assert smile == s[0]
            # assert f == frag[0]
            code = output[0].reshape(-1).tolist()
            codes.append(code)
        except:
            print(i, frags, smile)
    codes = pd.DataFrame(codes, columns=col)
    codes.to_csv(out, sep='\t', index=False)


def pair_smiles_encode(fname, voc, out, words):
    """
    
    Arguments:
        fname (list)             : name of file containing fragments and original SMILES
        voc (class)              : instance of VocSmiles
        out  (str)               : name of output file
        words (set)              : set of unique tokens that occur in the dataset
    """
    # read fragments and original SMILES
    df = pd.read_table(fname)
    col = ['Input', 'Output']
    codes = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        frag, smile = row.Frags, row.Smiles
        token_mol = voc.split(smile)
        ## currently not checking if molecules contain carbons
        if 10 < len(token_mol) <= 100:
            words.update(token_mol)
            token_sub = voc.split(frag)
            words.update(token_sub)
            codes.append([' '.join(token_sub), ' '.join(token_mol)])

    codes = pd.DataFrame(codes, columns=col)
    codes.to_csv(out, sep='\t', index=False)

    return words

def train_test_split(fname, out):
    # split into train and test
    df = pd.read_table(fname)
    frags = set(df.Frags)
    test_in = df.Frags.drop_duplicates().sample(len(frags) // 10)
    test = df[df.Frags.isin(test_in)]
    train = df[~df.Frags.isin(test_in)]
    test.to_csv(out + '_test.txt', sep='\t', index=False)
    train.to_csv(out + '_train.txt', sep='\t', index=False)
    
    
def DatasetArgParser(txt=None):
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-i', '--input', type=str, default='LIGAND_RAW.tsv',
                        help="Input file containing raw data. tsv or sdf.gz format")   
    parser.add_argument('-o', '--output', type=str, default='ligand',
                        help="Prefix of output files")
    parser.add_argument('-m', '--method', type=str, default='brics',
                        help="Method: 'brics' or 'recap'") 
    parser.add_argument('-mf', '--is_mf', type=bool, default=True,
                        help="If on, uses multiple fragments and largest 4 BRICS fragments are combined to form the output")
    parser.add_argument('-nf', '--n_frags', type=int, default=4,
                        help="If multiple fragments is true, sets how many of the largest BRICS fragments kept, and combined to form the output")  
    parser.add_argument('-v2', '--version_2', action='store_true',
                        help="If on, data processing for v2, else for v3.")
    parser.add_argument('-vf', '--voc_file', type=str, default='voc',
                        help="Name for voc file, used to save voc tokens") 
    parser.add_argument('-sv', '--save_voc', action='store_true',
                        help="If on, save voc file (should only be done for the pretraining set)")     
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
    if args.no_git is False:
        args.git_commit = utils.commit_hash(os.path.dirname(os.path.realpath(__file__)))
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(args.base_dir + '/data_args.json', 'w') as f:
        json.dump(vars(args), f)
        
    return args

def Dataset(args):
    if args.input.endswith('tsv') : suffix = 'tsv'
    elif args.input.endswith('sdf.gz'): suffix = 'sdf'
    else: sys.exit('Wrong input file format')

    # standardize smiles and remove salts
    smiles_std = standardize_mol(args.base_dir, args.input, suffix=suffix)

    # create corpus (only used in v2), vocab (only used in v2) and list of SMILES of length 10-100   
    corpus(args.base_dir, smiles_std, args.output, args.voc_file, args.save_voc)

    ## TODO rename version_2 ect. at some point
    if args.version_2 is False:
        out = '%s/data/%s_%s_%s.txt' % (args.base_dir, args.output, 'mf' if args.is_mf else 'sf', args.method)
        #create fragments from SMILES
        pair_frags(smiles_std, out, args.n_frags, method=args.method, is_mf=args.is_mf)

        inp = '%s/data/%s_%s_%s.txt' % (args.base_dir, args.output, 'mf' if args.is_mf else 'sf', args.method)
        out = '%s/data/%s_%s_%s' % (args.base_dir, args.output, 'mf' if args.is_mf else 'sf', args.method)
        train_test_split(inp, out)

        voc_smi = VocSmiles()
        ## need to see if also works without initializing vocab
        voc = VocGraph(args.base_dir +'/data/voc_graph.txt', n_frags=args.n_frags)

        # set for unique tokens that occur in train & test (not pretty)
        words = set()
        for ds in ['train', 'test']:
            pair_graph_encode(out + '_%s.txt' % ds, voc, out + '_%s_graph.txt' % ds)
            words = pair_smiles_encode(out + '_%s.txt' % ds, voc_smi, out + '_%s_smi.txt' % ds, words)    
        # save voc file
        if args.save_voc:
            log = open(args.base_dir + '/data/%s_smiles.txt' % args.voc_file, 'w')
            log.write('\n'.join(sorted(words)))
            log.close()
        print('Dataset prepreparation starting from {} for version 3 finished!'.format(args.input)) 
    else:
        print('Dataset prepreparation starting from {} for version 2 finished!'.format(args.input))

if __name__ == '__main__':

    args = DatasetArgParser()
    Dataset(args)


