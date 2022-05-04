"""
interfaces

Created by: Martin Sicho
On: 26.04.22, 13:12
"""
import logging
from abc import ABC, abstractmethod

from drugex.molecules.interfaces import MolSupplier


class Vocabulary(ABC):

    @abstractmethod
    def encode(self, tokens, frags=None):
        pass

    @abstractmethod
    def decode(self, representation):
        pass

    @staticmethod
    @abstractmethod
    def fromFile(path):
        pass

    @abstractmethod
    def toFile(self, path):
        pass

class VocabularySequence(Vocabulary, ABC):

    def __init__(self, words=None, max_len=100, min_len=10):
        """
        Args:
            words: iterable of words in this vocabulary
            max_len: the maximum number of tokens contained in one SMILES
        """

        self.control = ('_', 'GO', 'EOS')
        if words:
            self.words = words
        else:
            self.words = list(self.control) + ['.']
        self.updateIndex()
        self.max_len = max_len
        self.min_len = min_len

    @abstractmethod
    def splitSequence(self, seq):
        pass

    def toFile(self, path):
        log = open(path, 'w')
        log.write('\n'.join(self.words))
        log.close()

    def addWordsFromSeq(self, seq):
        token = self.splitSequence(seq)
        if self.min_len < len(token) <= self.max_len:
            diff = set(token) - self.words_set
            if len(diff) > 0:
                self.words.extend(diff)
                self.updateIndex()
            return token
        else:
            logging.info(f"Molecule does not meet min/max words requirements (min: {self.min_len}, max: {self.max_len}). Words found: {token} (count: {len(token)}). It will be ignored.")
            return None

    def updateIndex(self):
        self.words_set = set(self.words)
        self.size = len(self.words)
        self.tk2ix = dict(zip(self.words, range(len(self.words))))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}

class CorpusWriter(ABC):

    @abstractmethod
    def write(self, data):
        pass

    @abstractmethod
    def close(self):
        pass

class Corpus(MolSupplier, ABC):

    def __init__(self, molecules, out_writer=None):
        self.molecules = molecules if hasattr(molecules, "__next__") else iter(molecules)
        self.outWriter = out_writer

    def next(self):
        try:
            molecule = next(self.molecules)
            ret = self.processMolecule(molecule)
            if ret:
                if self.outWriter:
                    self.outWriter.write(ret)
                return ret
            else:
                return next(self)
        except StopIteration as exp:
            if self.outWriter:
                self.outWriter.close()
            raise exp

    def convertMol(self, representation):
        return representation

    def annotateMol(self, mol, key, value):
        return mol

    @abstractmethod
    def processMolecule(self, molecule):
        pass



