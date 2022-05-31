"""
interfaces

Created by: Martin Sicho
On: 26.04.22, 13:12
"""
from abc import ABC, abstractmethod

from drugex.logs import logger
from drugex.molecules.interfaces import MolSupplier


class Vocabulary(ABC):

    def __init__(self, words):
        self.words = words

    def __add__(self, other):
        return type(self)(other.words + self.words)

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
        self.special = list(self.control) + ['.']
        self.wordSet = set()
        if words:
            self.wordSet = set(x for x in words if x not in self.special)
        self.updateIndex()
        self.max_len = max_len
        self.min_len = min_len

    @abstractmethod
    def splitSequence(self, seq):
        pass

    def toFile(self, path):
        log = open(path, 'w')
        log.write('\n'.join([x for x in self.words if x not in self.special]))
        log.close()

    def addWordsFromSeq(self, seq, ignoreConstraints=False):
        token = self.splitSequence(seq)
        if ignoreConstraints or (self.min_len < len(token) <= self.max_len):
            diff = set(token) - self.wordSet
            if len(diff) > 0:
                self.wordSet.update(diff)
                self.updateIndex()
            return token
        else:
            logger.warning(f"Molecule does not meet min/max words requirements (min: {self.min_len}, max: {self.max_len}). Words found: {set(token)} (occurrence count: {len(token)}). It will be ignored.")
            return None

    def updateIndex(self):
        self.words = self.special + [x for x in sorted(self.wordSet) if x not in self.special]
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
            try:
                ret = self.processMolecule(molecule)
            except Exception as exp:
                logger.warning(f'Exception occurred when processing data for molecule: {molecule}')
                logger.exception(exp)
                return next(self)
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

    @abstractmethod
    def getVoc(self):
        pass



