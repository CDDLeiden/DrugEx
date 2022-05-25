import torch
import re
from rdkit import Chem
import numpy as np
import pandas as pd
import utils
from typing import List, Iterable, Optional


# class Voc(object):
#     def __init__(self, init_from_file=None, src_len=1000, trg_len=100):
#         self.control = ('_', 'GO', 'EOS')
#         self.words = list(self.control) + ['.']
#         self.src_len = src_len
#         self.trg_len = trg_len
#         if init_from_file: self.init_from_file(init_from_file)
#         self.size = len(self.words)
#         self.tk2ix = dict(zip(self.words, range(len(self.words))))
#         self.ix2tk = {v: k for k, v in self.tk2ix.items()}
#
#     def split(self, seq, is_smiles=True):
#         """Takes a SMILES and return a list of characters/tokens"""
#         tokens = []
#         if is_smiles:
#             regex = '(\[[^\[\]]{1,6}\])'
#             seq = re.sub('\[\d+', '[', seq)
#             seq = seq.replace('Br', 'R').replace('Cl', 'L')
#             for word in re.split(regex, seq):
#                 if word == '' or word is None: continue
#                 if word.startswith('['):
#                     tokens.append(word)
#                 else:
#                     for i, char in enumerate(word):
#                         tokens.append(char)
#         else:
#             for token in seq:
#                 token.append('|' + token)
#         return tokens
#
#     def encode(self, input, is_smiles=True):
#         """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
#         seq_len = self.trg_len if is_smiles else self.src_len
#         output = torch.zeros(len(input), seq_len).long()
#         for i, seq in enumerate(input):
#             # print(i, len(seq))
#             for j, char in enumerate(seq):
#                 output[i, j] = self.tk2ix[char] if is_smiles else self.tk2ix['|' + char]
#         return output
#
#     def decode(self, matrix, is_smiles=True):
#         """Takes an array of indices and returns the corresponding SMILES"""
#         chars = []
#         for i in matrix:
#             token = self.ix2tk[i.item()]
#             if token == 'EOS': break
#             if token in self.control: continue
#             chars.append(token)
#         seqs = "".join(chars)
#         if is_smiles:
#             seqs = seqs.replace('L', 'Cl').replace('R', 'Br')
#         else:
#             seqs = seqs.replace('|', '')
#         return seqs
#
#     def init_from_file(self, file):
#         """Takes a file containing \n separated characters to initialize the vocabulary"""
#         with open(file, 'r') as f:
#             chars = f.read().split()
#             assert len(set(chars)) == len(chars)
#             self.words += chars


# class VocGraph:
#     def __init__(self, init_from_file=None, max_len=80, n_frags=4):
#         self.control = ('EOS', 'GO')
#         self.words = list(self.control)
#         self.max_len = max_len
#         self.n_frags = n_frags
#         self.tk2ix = {'EOS': 0, 'GO': 1}
#         self.ix2nr = {0: 0, 1: 0}
#         self.ix2ch = {0: 0, 1: 0}
#         if init_from_file: self.init_from_file(init_from_file)
#         self.size = len(self.words)
#         self.E = {0: '', 1: '+', -1: '-'}
#
#     def init_from_file(self, file):
#         chars = []
#         df = pd.read_table(file)
#         self.masks = torch.zeros(len(df) + len(self.control)).long()
#         for i, row in df.iterrows():
#             self.masks[i + len(self.control)] = row.Val
#             ix = i + len(self.control)
#             self.tk2ix[row.Word] = ix
#             self.ix2nr[ix] = row.Nr
#             self.ix2ch[ix] = row.Ch
#             chars.append(row.Word)
#         assert len(set(chars)) == len(chars)
#         self.words += chars
#
#     def get_atom_tk(self, atom):
#         sb = atom.GetSymbol() + self.E[atom.GetFormalCharge()]
#         val = atom.GetExplicitValence() + atom.GetImplicitValence()
#         tk = str(val) + sb
#         return self.tk2ix[tk]
#
#     def encode(self, smiles, subs):
#         output = np.zeros([len(smiles), self.max_len-self.n_frags-1, 5], dtype=np.long)
#         connect = np.zeros([len(smiles), self.n_frags+1, 5], dtype=np.long)
#         for i, s in enumerate(smiles):
#             mol = Chem.MolFromSmiles(s)
#             sub = Chem.MolFromSmiles(subs[i])
#             # Chem.Kekulize(sub)
#             sub_idxs = mol.GetSubstructMatches(sub)
#             for sub_idx in sub_idxs:
#                 sub_bond = [mol.GetBondBetweenAtoms(
#                     sub_idx[b.GetBeginAtomIdx()],
#                     sub_idx[b.GetEndAtomIdx()]).GetIdx() for b in sub.GetBonds()]
#                 sub_atom = [mol.GetAtomWithIdx(ix) for ix in sub_idx]
#                 split_bond = {b.GetIdx() for a in sub_atom for b in a.GetBonds() if b.GetIdx() not in sub_bond}
#                 single = sum([int(mol.GetBondWithIdx(b).GetBondType()) for b in split_bond])
#                 if single == len(split_bond): break
#             frags = Chem.FragmentOnBonds(mol, list(split_bond))
#
#             Chem.MolToSmiles(frags)
#             rank = eval(frags.GetProp('_smilesAtomOutputOrder'))
#             mol_idx = list(sub_idx) + [idx for idx in rank if idx not in sub_idx and idx < mol.GetNumAtoms()]
#             frg_idx = [i+1 for i, f in enumerate(Chem.GetMolFrags(sub)) for _ in f]
#
#             Chem.Kekulize(mol)
#             m, n, c = [(self.tk2ix['GO'], 0, 0, 0, 1)], [], [(self.tk2ix['GO'], 0, 0, 0, 0)]
#             mol2sub = {ix: i for i, ix in enumerate(mol_idx)}
#             for j, idx in enumerate(mol_idx):
#                 atom = mol.GetAtomWithIdx(idx)
#                 bonds = sorted(atom.GetBonds(), key=lambda x: mol2sub[x.GetOtherAtomIdx(idx)])
#                 bonds = [b for b in bonds if j > mol2sub[b.GetOtherAtomIdx(idx)]]
#                 n_split = sum([1 if b.GetIdx() in split_bond else 0 for b in bonds])
#                 tk = self.get_atom_tk(atom)
#                 for k, bond in enumerate(bonds):
#                     ix2 = mol2sub[bond.GetOtherAtomIdx(idx)]
#                     is_split = bond.GetIdx() in split_bond
#                     if idx in sub_idx:
#                         is_connect = is_split
#                     elif len(bonds) == 1:
#                         is_connect = False
#                     elif n_split == len(bonds):
#                         is_connect = is_split and k != 0
#                     else:
#                         is_connect = False
#                     if bond.GetIdx() in sub_bond:
#                         bin, f = m, frg_idx[j]
#                     elif is_connect:
#                         bin, f = c, 0
#                     else:
#                         bin, f = n, 0
#                     if bond.GetIdx() in sub_bond or not is_connect:
#                         tk2 = tk
#                         tk = self.tk2ix['*']
#                     else:
#                         tk2 = self.tk2ix['*']
#                     bin.append((tk2, j, ix2, int(bond.GetBondType()), f))
#                 if tk != self.tk2ix['*']:
#                     bin, f = (m, frg_idx[j]) if idx in sub_idx else (n, f)
#                     bin.append((tk, j, j, 0, f))
#             output[i, :len(m+n), :] = m+n
#             if len(c) > 0:
#                 connect[i, :len(c)] = c
#         return np.concatenate([output, connect], axis=1)
#
#     def decode(self, matrix):
#         frags, smiles = [], []
#         for m, adj in enumerate(matrix):
#             # print('decode: ', m)
#             emol = Chem.RWMol()
#             esub = Chem.RWMol()
#             try:
#                 for atom, curr, prev, bond, frag in adj:
#                     atom, curr, prev, bond, frag = int(atom), int(curr), int(prev), int(bond), int(frag)
#                     if atom == self.tk2ix['EOS']: continue
#                     if atom == self.tk2ix['GO']: continue
#                     if atom != self.tk2ix['*']:
#                         a = Chem.Atom(self.ix2nr[atom])
#                         a.SetFormalCharge(self.ix2ch[atom])
#                         emol.AddAtom(a)
#                         if frag != 0: esub.AddAtom(a)
#                     if bond != 0:
#                         b = Chem.BondType(bond)
#                         emol.AddBond(curr, prev, b)
#                         if frag != 0: esub.AddBond(curr, prev, b)
#                 Chem.SanitizeMol(emol)
#                 Chem.SanitizeMol(esub)
#             except Exception as e:
#                 print(adj)
#                 # raise e
#             frags.append(Chem.MolToSmiles(esub))
#             smiles.append(Chem.MolToSmiles(emol))
#         return frags, smiles


# class VocSeq:
#     def __init__(self, max_len=1000):
#         self.chars = ['_'] + [r for r in utils.AA]
#         self.size = len(self.chars)
#         self.max_len = max_len
#         self.tk2ix = dict(zip(self.chars, range(len(self.chars))))
#
#     def encode(self, seqs):
#         """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
#         output = torch.zeros(len(seqs), self.max_len).long()
#         for i, seq in enumerate(seqs):
#             for j, res in enumerate(seq):
#                 if res not in self.chars:
#                     res = '_'
#                 output[i, j] = self.tk2ix[res]
#         return output

    
# class VocTgt:
#     def __init__(self, max_len=1000):
#         self.chars = ['_'] + [r for r in utils.AA]
#         self.size = len(self.chars)
#         self.max_len = max_len
#         self.tk2ix = dict(zip(self.chars, range(len(self.chars))))
#
#     def encode(self, seqs):
#         """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
#         output = torch.zeros(len(seqs), self.max_len).long()
#         for i, seq in enumerate(seqs):
#             for j, res in enumerate(seq):
#                 if res not in self.chars:
#                     res = '_'
#                 output[i, j] = self.tk2ix[res]
#         return output


class TgtData():
    def __init__(self, seqs, ix, max_len=100):
        self.max_len = max_len
        self.index = np.array(ix)
        self.map = {idx: i for i, idx in enumerate(self.index)}
        self.seq = seqs

    def __getitem__(self, i):
        seq = self.seq[i]
        return i, seq

    def __len__(self):
        return len(self.seq)

    def collate_fn(self, arr):
        collated_ix = np.zeros(len(arr), dtype=int)
        collated_seq = torch.zeros(len(arr), self.max_len).long()
        for i, (ix, tgt) in enumerate(arr):
            collated_ix[i] = ix
            collated_seq[i, :] = tgt
        return collated_ix, collated_seq


#THESE FUNCTIONS WERE ADDED BY HELLE FROM:
#   https://github.com/BenevolentAI/guacamol/blob/8247bbd5e927fbc3d328865d12cf83cb7019e2d6/guacamol/utils/data.py#L11
# to solve AttributeError: module 'utils' has no attribute 'canonicalize_list'
def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.
    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543
    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string
    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None

def remove_duplicates(list_with_duplicates):
    """
    Removes the duplicates and keeps the ordering of the original list.
    For duplicates, the first occurrence is kept and the later occurrences are ignored.
    Args:
        list_with_duplicates: list that possibly contains duplicates
    Returns:
        A list with no duplicates.
    """

    unique_set = set()
    unique_list = []
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)

    return unique_list

def canonicalize_list(smiles_list: Iterable[str], include_stereocenters=True) -> List[str]:
    """
    Canonicalize a list of smiles. Filters out repetitions and removes corrupted molecules.
    Args:
        smiles_list: molecules as SMILES strings
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings
    Returns:
        The canonicalized and filtered input smiles.
    """

    canonicalized_smiles = [canonicalize(smiles, include_stereocenters) for smiles in smiles_list]

    # Remove None elements
    canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]

    return remove_duplicates(canonicalized_smiles)
