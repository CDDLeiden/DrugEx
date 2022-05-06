"""
vocabulary

Created by: Martin Sicho
On: 26.04.22, 13:16
"""
import re

import numpy as np
import torch

from drugex.corpus.interfaces import VocabularySequence
from drugex.molecules.converters.standardizers import CleanSMILES


class VocSmiles(VocabularySequence):
    """A class for handling encoding/decoding from SMILES to an array of indices"""

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

    def decode(self, tensor, is_tk=True):
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
        smiles = "".join(tokens)
        smiles = smiles.replace('L', 'Cl').replace('R', 'Br')
        return smiles

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
        words = []
        with open(path, 'r') as f:
            words += f.read().split()
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