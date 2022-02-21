import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Recap, BRICS
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm
from utils import VocSmiles as Voc
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


def corpus(input, output, suffix='sdf'):
    if suffix =='sdf':
        inf = gzip.open(input)
        mols = Chem.ForwardSDMolSupplier(inf)
        # mols = [mol for mol in suppl]
    else:
        df = pd.read_table(input).SMILES.dropna()
        mols = [Chem.MolFromSmiles(s) for s in df]
    voc = Voc('data/voc_smiles.txt')
    charger = rdMolStandardize.Uncharger()
    chooser = rdMolStandardize.LargestFragmentChooser()
    disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    words = set()
    canons = []
    tokens = []
    smiles = set()
    for mol in tqdm(mols):
        try:
            mol = disconnector.Disconnect(mol)
            mol = normalizer.normalize(mol)
            mol = chooser.choose(mol)
            mol = charger.uncharge(mol)
            mol = disconnector.Disconnect(mol)
            mol = normalizer.normalize(mol)
            smileR = Chem.MolToSmiles(mol, 0)
            smiles.add(Chem.CanonSmiles(smileR))
        except:
            print('Parsing Error:') #, Chem.MolToSmiles(mol))

    for smile in tqdm(smiles):
        token = voc.split(smile) + ['EOS']
        if {'C', 'c'}.isdisjoint(token):
            print('Warning:', smile)
            continue
        if not {'[Na]', '[Zn]'}.isdisjoint(token):
            print('Redudent', smile)
            continue
        if 10 < len(token) <= 100:
            words.update(token)
            canons.append(smile)
            tokens.append(' '.join(token))
    log = open(output + '_voc.txt', 'w')
    log.write('\n'.join(sorted(words)))
    log.close()

    log = pd.DataFrame()
    log['Smiles'] = canons
    log['Token'] = tokens
    log.drop_duplicates(subset='Smiles')
    log.to_csv(output + '_corpus.txt', sep='\t', index=False)


def graph_corpus(input, output, suffix='sdf'):
    metals = {'Na', 'Zn', 'Li', 'K', 'Ca', 'Mg', 'Ag', 'Cs', 'Ra', 'Rb', 'Al', 'Sr', 'Ba', 'Bi'}
    voc = utils.VocGraph('data/voc_graph.txt')
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


def pair_frags(fname, out, voc, method='Recap', is_mf=True):
    smiles = pd.read_table(fname).Smiles.dropna()
    pairs = []
    for i, smile in enumerate(tqdm(smiles)):
        smile = utils.clean_mol(smile)
        mol = Chem.MolFromSmiles(smile)
        if method == 'recap':
            frags = np.array(sorted(Recap.RecapDecompose(mol).GetLeaves().keys()))
        else:
            frags = BRICS.BRICSDecompose(mol)
            frags = np.array(sorted({re.sub(r'\[\d+\*\]', '*', f) for f in frags}))
        if len(frags) == 1: continue
        du, hy = Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]')
        subs = np.array([Chem.MolFromSmiles(f) for f in frags])
        subs = np.array([Chem.RemoveHs(Chem.ReplaceSubstructs(f, du, hy, replaceAll=True)[0]) for f in subs])
        subs = np.array([m for m in subs if m.GetNumAtoms() > 1])
        match = np.array([[m.HasSubstructMatch(f) for f in subs] for m in subs])
        frags = subs[match.sum(axis=0) == 1]
        frags = sorted(frags, key=lambda x:-x.GetNumAtoms())[:voc.n_frags]
        frags = [Chem.MolToSmiles(Chem.RemoveHs(f)) for f in frags]

        max_comb = len(frags) if is_mf else 1
        for ix in range(1, max_comb+1):
            combs = combinations(frags, ix)
            for comb in combs:
                input = '.'.join(comb)
                if len(input) > len(smile): continue
                if mol.HasSubstructMatch(Chem.MolFromSmarts(input)):
                    pairs.append([input, smile])
    df = pd.DataFrame(pairs, columns=['Frags', 'Smiles'])
    df.to_csv(out, sep='\t',  index=False)


def pair_graph_encode(fname, voc, out):
    df = pd.read_table(fname)
    col = ['C%d' % d for d in range(voc.max_len*5)]
    codes = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        frag, smile = row.Frags, row.Smiles
        # smile = voc_smi.decode(row.Output.split(' '))
        # frag = voc_smi.decode(row.Input.split(' '))
        mol = Chem.MolFromSmiles(smile)
        total = mol.GetNumBonds()
        if total >= 75 or smile == frag:
            continue
        try:
            # s = utils.clean_mol(smile)
            # f = utils.clean_mol(frag, is_deep=False)
            output = voc.encode([smile], [frag])
            f, s = voc.decode(output)

            assert smile == s[0]
            # assert f == frag[0]
            code = output[0].reshape(-1).tolist()
            codes.append(code)
        except:
            print(i, frag, smile)
    codes = pd.DataFrame(codes, columns=col)
    codes.to_csv(out, sep='\t', index=False)


def pair_smiles_encode(fname, voc, out):
    df = pd.read_table(fname)
    col = ['Input', 'Output']
    codes = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        frag, smile = row.Frags, row.Smiles
        mol = voc.split(smile)
        if len(mol) > 100: continue
        sub = voc.split(frag)
        codes.append([' '.join(sub), ' '.join(mol)])
    codes = pd.DataFrame(codes, columns=col)
    codes.to_csv(out, sep='\t', index=False)


def pos_neg_split():
    pair = ['Target ChEMBL ID', 'Smiles', 'pChEMBL Value', 'Comment',
            'Standard Type', 'Standard Relation']
    obj = pd.read_table('data/LIGAND.tsv').dropna(subset=pair[1:2])
    df = obj[obj[pair[0]] == 'CHEMBL251']
    df = df[pair].set_index(pair[1])
    numery = df[pair[2]].groupby(pair[1]).mean().dropna()

    comments = df[(df.Comment.str.contains('Not Active') == True)]
    inhibits = df[(df['Standard Type'] == 'Inhibition') & df['Standard Relation'].isin(['<', '<='])]
    relations = df[df['Standard Type'].isin(['EC50', 'IC50', 'Kd', 'Ki']) & df['Standard Relation'].isin(['>', '>='])]
    binary = pd.concat([comments, inhibits, relations], axis=0)
    binary = binary[~binary.index.isin(numery.index)]
    binary[pair[2]] = 3.99
    binary = binary[pair[2]].groupby(binary.index).first()
    df = numery.append(binary)
    pos = {utils.clean_mol(s) for s in df[df >=6.5].index}
    neg = {utils.clean_mol(s) for s in df[df < 6.5].index}.difference(pos)
    oth = obj[~obj.Smiles.isin(df.index)].Smiles
    oth = {utils.clean_mol(s) for s in oth}.difference(pos).difference(neg)
    for data in ['pos', 'neg', 'oth']:
        file = open('data/ligand_%s.tsv' % data, 'w')
        file.write('Smiles\n')
        file.write('\n'.join(eval(data)))
        file.close()


def train_test_split(fname, out):
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
    parser.add_argument('-f', '--is_mf', type=bool, default=True,
                        help="??") 
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
    args.git_commit = utils.commit_hash(os.path.dirname(os.path.realpath(__file__)))
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(args.base_dir + '/data_args.json', 'w') as f:
        json.dump(vars(args), f)
        
    return args

def Dataset(args):
    
    if args.input.endswith('tsv') : suffix = 'tsv'
    elif args.input.endswith('sdf.gz'): suffix = 'sdf'
    else: sys.exit('Wrong input file format')
        
    corpus(args.base_dir + '/data/' + args.input, 
           args.base_dir + '/data/' + args.output,
           suffix=suffix)

    voc = utils.VocGraph(args.base_dir +'/data/voc_graph.txt', n_frags=4)
    voc_smi = utils.VocSmiles(args.base_dir + '/data/voc_smiles.txt')
    
    inp = '%s/data/%s_corpus.txt' % (args.base_dir, args.output)
    out = '%s/data/%s_%s_%s.txt' % (args.base_dir, args.output, 'mf' if args.is_mf else 'sf', args.method)
    pair_frags(inp, out, voc, method=args.method, is_mf=args.is_mf)
    
    inp = '%s/data/%s_%s_%s.txt' % (args.base_dir, args.output, 'mf' if args.is_mf else 'sf', args.method)
    out = '%s/data/%s_%s_%s' % (args.base_dir, args.output, 'mf' if args.is_mf else 'sf', args.method)
    train_test_split(inp, out)
    
    for ds in ['train', 'test']:
        pair_graph_encode(out + '_%s.txt' % ds, voc, out + '_%s_code.txt' % ds)
        pair_smiles_encode(out + '_%s.txt' % ds, voc_smi, out + '_%s_smi.txt' % ds)
    #pos_neg_split() # TO DO : think about this function
    

if __name__ == '__main__':
    
    args = DatasetArgParser()
    Dataset(args)

