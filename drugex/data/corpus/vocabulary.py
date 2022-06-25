"""
vocabulary

Created by: Martin Sicho
On: 26.04.22, 13:16
"""
import re

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from drugex.data.corpus.interfaces import VocabularySequence, Vocabulary
from drugex.molecules.converters.standardizers import CleanSMILES


class VocSmiles(VocabularySequence):
    """A class for handling encoding/decoding from SMILES to an array of indices"""

    defaultWords = ('#','%','(',')','-','0','1','2','3','4','5','6','7','8','9','=','B','C','F','I','L','N','O','P','R','S','[Ag-3]','[As+]','[As]','[B-]','[BH-]','[BH2-]','[BH3-]','[B]','[C+]','[C-]','[CH-]','[CH2]','[CH2-]','[CH]','[I+]','[IH2]','[N+]','[N-]','[NH+]','[NH-]','[NH2+]','[N]','[O+]','[O-]','[OH+]','[O]','[P+]','[PH]','[S+]','[S-]','[SH+]','[SH2]','[SH]','[Se+]','[SeH]','[Se]','[SiH2]','[SiH]','[Si]','[Te]','[b-]','[c+]','[c-]','[cH-]','[n+]','[n-]','[nH+]','[nH]','[o+]','[s+]','[se+]','[se]','[te+]',"[te]",'b','c','n','o','p','s'
    )

    def __init__(self, words=defaultWords, max_len=100, min_len=10):
        super().__init__(words, max_len=max_len, min_len=min_len)

    def encode(self, tokens, frags=None):
        """
        Takes a list of tokens (eg '[NH]') and encodes to array of indices
        Args:
            input: a list of SMILES squence represented as a series of tokens

        Returns:
            output (torch.LongTensor): a long tensor containing all of the indices of given tokens.
        """

        output = torch.zeros(len(tokens), self.max_len).long()
        for i, seq in enumerate(tokens):
            # print(i, len(seq))
            for j, char in enumerate(seq):
                output[i, j] = self.tk2ix[char]
        return output

    def decode(self, tensor, is_tk=True, is_smiles=True):
        """Takes an array of indices and returns the corresponding SMILES
        Args:
            tensor(torch.LongTensor): a long tensor containing all of the indices of given tokens.

        Returns:
            smiles (str): a decoded smiles sequence.
        """
        tokens = []
        for token in tensor:
            if not is_tk:
                token = self.ix2tk[int(token)]
            if token == 'EOS': break
            if token in self.control: continue
            tokens.append(token)
        seqs = "".join(tokens)
        if is_smiles:
            seqs = self.parseDecoded(seqs)
        else:
            seqs = seqs.replace('|', '')
        return seqs

    def parseDecoded(self, smiles):
        return smiles.replace('L', 'Cl').replace('R', 'Br')

    def splitSequence(self, smile):
        """Takes a SMILES and return a list of characters/tokens
        Args:
            smile (str): a decoded smiles sequence.

        Returns:
            tokens (List): a list of tokens decoded from the SMILES sequence.
        """
        regex = '(\[[^\[\]]{1,6}\])'
        smile = smile.replace('Cl', 'L').replace('Br', 'R')
        tokens = []
        for word in re.split(regex, smile):
            if word == '' or word is None: continue
            if word.startswith('['):
                tokens.append(word)
            else:
                for i, char in enumerate(word):
                    tokens.append(char)
        return tokens + ['EOS']

    @staticmethod
    def fromFile(path, min_len=10, max_len=100):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(path, 'r') as f:
            words = f.read().split()
            return VocSmiles(words, max_len=max_len, min_len=min_len)

    def calc_voc_fp(self, smiles, prefix=None):
        fps = np.zeros((len(smiles), self.max_len), dtype=np.long)
        for i, smile in enumerate(smiles):
            smile = CleanSMILES()(smile)
            token = self.splitSequence(smile)
            if prefix is not None: token = [prefix] + token
            if len(token) > self.max_len: continue
            if {'C', 'c'}.isdisjoint(token): continue
            if not {'[Na]', '[Zn]'}.isdisjoint(token): continue
            fps[i, :] = self.encode(token)
        return fps

class VocGPT(VocSmiles):

    def __init__(self, words, src_len=1000, trg_len=100, max_len=100, min_len=10):
        super(VocGPT, self).__init__(words, max_len=max_len, min_len=min_len)
        self.src_len = src_len
        self.trg_len = trg_len

    def encode(self, input, is_smiles=True):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        seq_len = self.trg_len if is_smiles else self.src_len
        output = torch.zeros(len(input), seq_len).long()
        for i, seq in enumerate(input):
            # print(i, len(seq))
            for j, char in enumerate(seq):
                output[i, j] = self.tk2ix[char] if is_smiles else self.tk2ix['|' + char]
        return output

    def decode(self, matrix, is_smiles=True, is_tk=False):
        """
        Takes an array of indices and returns the corresponding SMILES.
        """
        chars = super(VocGPT, self).decode(matrix, is_tk)
        seqs = "".join(chars)
        if is_smiles:
            seqs = self.parseDecoded(seqs)
        else:
            seqs = seqs.replace('|', '')
        return seqs


    @staticmethod
    def fromFile(path, src_len=1000, trg_len=100, max_len=100, min_len=10):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(path, 'r') as f:
            words = f.read().split()
            return VocGPT(words, src_len=src_len, trg_len=trg_len, max_len=max_len, min_len=min_len)

class VocGraph(Vocabulary):

    defaultWords=('2O','3O+','1O-','4C','3C+','3C-','3N','4N+','2N-','1Cl','2S','6S','4S','3S+','5S+','1S-','1F','1I','5I','2I+','1Br','5P','3P','4P+','2Se','6Se','4Se','3Se+','4Si','3B','4B-','5As','3As','4As+','2Te','4Te','3Te+',)

    def __init__(self, words=defaultWords, max_len=80, n_frags=4):
        super().__init__(words)
        self.control = ('EOS', 'GO')
        words = [x for x in words if x not in self.control]
        words_unique = []
        for word in words:
            if word not in words_unique:
                words_unique.append(word)
        words = words_unique
        self.n_frags = n_frags
        self.max_len = max_len
        self.tk2ix = {'EOS': 0, 'GO': 1}
        self.ix2nr = {0: 0, 1: 0}
        self.ix2ch = {0: 0, 1: 0}
        self.E = {0: '', 1: '+', -1: '-'}

        # init words
        self.words = []
        self.wordsParsed = [self.parseWord(word) for word in words]
        self.words = list(self.control) + list(words)
        if '*' not in words:
            self.words.append('*')
            self.wordsParsed.append(('*',0,0,0,'*'))
        self.size = len(self.words)
        self.masks = torch.zeros(len(self.wordsParsed) + len(self.control)).long()
        for i,item in enumerate(self.wordsParsed):
            self.masks[i + len(self.control)] = item[1]
            ix = i + len(self.control)
            self.tk2ix[item[4]] = ix
            self.ix2nr[ix] = item[3]
            self.ix2ch[ix] = item[2]
        assert len(set(self.words)) == len(self.words)

    @staticmethod
    def parseWord(word):
        if word == '*':
            return '*',0,0,0,'*'
        valence = re.search(r'[0-9]', word).group(0)
        charge = re.search(r'[+-]', word)
        charge_num = 0
        if charge:
            charge = charge.group(0)
            charge_num = 1 if charge == '+' else -1
        else:
            charge = ''
        element = re.search(r'[a-zA-Z]+', word).group(0)
        return element + charge, int(valence), charge_num, Chem.Atom(element).GetAtomicNum(), word

    @staticmethod
    def fromFile(path, word_col='Word', max_len=80, n_frags=4):
        df = pd.read_table(path)
        return VocGraph.fromDataFrame(df, word_col, max_len=80, n_frags=4)

    @staticmethod
    def fromDataFrame(df, word_col='Word', max_len=80, n_frags=4):
        return VocGraph(df[word_col].tolist(), max_len=max_len, n_frags=n_frags)

    def toFile(self, path):
        self.toDataFrame().to_csv(path, index=False, sep='\t')

    def toDataFrame(self):
        return pd.DataFrame(self.wordsParsed, columns=['Ele', 'Val', 'Ch', 'Nr', 'Word'])

    def get_atom_tk(self, atom):
        sb = atom.GetSymbol() + self.E[atom.GetFormalCharge()]
        val = atom.GetExplicitValence() + atom.GetImplicitValence()
        tk = str(val) + sb
        return self.tk2ix[tk]

    def encode(self, smiles, subs=None):
        if not subs:
            raise RuntimeError(f'Fragments must be specified, got {subs} instead')

        output = np.zeros([len(smiles), self.max_len - self.n_frags - 1, 5], dtype=np.compat.long)
        connect = np.zeros([len(smiles), self.n_frags + 1, 5], dtype=np.compat.long)
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            sub = Chem.MolFromSmiles(subs[i])
            # Chem.Kekulize(sub)
            sub_idxs = mol.GetSubstructMatches(sub)
            for sub_idx in sub_idxs:
                sub_bond = [mol.GetBondBetweenAtoms(
                    sub_idx[b.GetBeginAtomIdx()],
                    sub_idx[b.GetEndAtomIdx()]).GetIdx() for b in sub.GetBonds()]
                sub_atom = [mol.GetAtomWithIdx(ix) for ix in sub_idx]
                split_bond = {b.GetIdx() for a in sub_atom for b in a.GetBonds() if b.GetIdx() not in sub_bond}
                single = sum([int(mol.GetBondWithIdx(b).GetBondType()) for b in split_bond])
                if single == len(split_bond): break
            frags = Chem.FragmentOnBonds(mol, list(split_bond))

            Chem.MolToSmiles(frags)
            rank = eval(frags.GetProp('_smilesAtomOutputOrder'))
            mol_idx = list(sub_idx) + [idx for idx in rank if idx not in sub_idx and idx < mol.GetNumAtoms()]
            frg_idx = [i+1 for i, f in enumerate(Chem.GetMolFrags(sub)) for _ in f]

            Chem.Kekulize(mol)
            m, n, c = [(self.tk2ix['GO'], 0, 0, 0, 1)], [], [(self.tk2ix['GO'], 0, 0, 0, 0)]
            mol2sub = {ix: i for i, ix in enumerate(mol_idx)}
            for j, idx in enumerate(mol_idx):
                atom = mol.GetAtomWithIdx(idx)
                bonds = sorted(atom.GetBonds(), key=lambda x: mol2sub[x.GetOtherAtomIdx(idx)])
                bonds = [b for b in bonds if j > mol2sub[b.GetOtherAtomIdx(idx)]]
                n_split = sum([1 if b.GetIdx() in split_bond else 0 for b in bonds])
                tk = self.get_atom_tk(atom)
                for k, bond in enumerate(bonds):
                    ix2 = mol2sub[bond.GetOtherAtomIdx(idx)]
                    is_split = bond.GetIdx() in split_bond
                    if idx in sub_idx:
                        is_connect = is_split
                    elif len(bonds) == 1:
                        is_connect = False
                    elif n_split == len(bonds):
                        is_connect = is_split and k != 0
                    else:
                        is_connect = False
                    if bond.GetIdx() in sub_bond:
                        bin, f = m, frg_idx[j]
                    elif is_connect:
                        bin, f = c, 0
                    else:
                        bin, f = n, 0
                    if bond.GetIdx() in sub_bond or not is_connect:
                        tk2 = tk
                        tk = self.tk2ix['*']
                    else:
                        tk2 = self.tk2ix['*']
                    bin.append((tk2, j, ix2, int(bond.GetBondType()), f))
                if tk != self.tk2ix['*']:
                    bin, f = (m, frg_idx[j]) if idx in sub_idx else (n, f)
                    bin.append((tk, j, j, 0, f))
            output[i, :len(m+n), :] = m+n
            if len(c) > 0:
                connect[i, :len(c)] = c
        return np.concatenate([output, connect], axis=1)

    def decode(self, matrix):
        frags, smiles = [], []
        for m, adj in enumerate(matrix):
            # print('decode: ', m)
            emol = Chem.RWMol()
            esub = Chem.RWMol()
            try:
                for atom, curr, prev, bond, frag in adj:
                    atom, curr, prev, bond, frag = int(atom), int(curr), int(prev), int(bond), int(frag)
                    if atom == self.tk2ix['EOS']: continue
                    if atom == self.tk2ix['GO']: continue
                    if atom != self.tk2ix['*']:
                        a = Chem.Atom(self.ix2nr[atom])
                        a.SetFormalCharge(self.ix2ch[atom])
                        emol.AddAtom(a)
                        if frag != 0: esub.AddAtom(a)
                    if bond != 0:
                        b = Chem.BondType(bond)
                        emol.AddBond(curr, prev, b)
                        if frag != 0: esub.AddBond(curr, prev, b)
                Chem.SanitizeMol(emol)
                Chem.SanitizeMol(esub)
            except Exception as e:
                print(adj)
                # raise e
            frags.append(Chem.MolToSmiles(esub))
            smiles.append(Chem.MolToSmiles(emol))
        return frags, smiles

DEFAULT_GRAPH = VocGraph(VocGraph.defaultWords)
DEFAULT_SMILES = VocSmiles(VocSmiles.defaultWords)
DEFAULT_GPT = VocGPT(VocSmiles.defaultWords)